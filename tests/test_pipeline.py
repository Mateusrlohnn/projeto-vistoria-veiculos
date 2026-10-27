"""Testes minimos do pipeline com imagens sinteticas (nao dependem do dataset).

    python -m pytest tests
"""
import cv2
import numpy as np
import pytest
import yaml

from src.avaliacao import avaliar_imagem, iou_dice, resumir
from src.pipeline import aplicar_morfologia, processar, regiao_aprovada


def imagem_sintetica():
    """Lataria cinza com textura leve, um risco claro fino e um reflexo branco circular."""
    rng = np.random.default_rng(0)
    img = np.full((480, 640, 3), (90, 90, 95), np.uint8)
    img += rng.integers(0, 12, img.shape, dtype=np.uint8)
    cv2.line(img, (100, 100), (400, 180), (170, 170, 175), 2)
    cv2.circle(img, (500, 350), 40, (255, 255, 255), -1)
    ref = np.zeros((480, 640), np.uint8)
    cv2.line(ref, (100, 100), (400, 180), 1, 5)
    return img, ref


def carregar_cfg(nome):
    with open(f"configs/{nome}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_iou_dice_caso_conhecido():
    a = np.zeros((10, 10), np.uint8)
    b = np.zeros((10, 10), np.uint8)
    a[:, :4] = 1   # 40 px
    b[:, 2:6] = 1  # 40 px, intersecao 20
    iou, dice = iou_dice(a, b)
    assert iou == pytest.approx(20 / 60)
    assert dice == pytest.approx(0.5)


def test_iou_imagem_negativa_sem_predicao():
    vazio = np.zeros((5, 5), np.uint8)
    assert iou_dice(vazio, vazio) == (1.0, 1.0)


def test_avaliar_conta_falso_positivo():
    ref = np.zeros((50, 50), np.uint8)
    ref[10, 5:40] = 1
    pred = ref.copy()
    pred[40:45, 40:45] = 1  # regiao longe da referencia
    m = avaliar_imagem(pred, [ref])
    assert (m["detectadas"], m["n_pred"], m["vp"], m["fp"]) == (1, 2, 1, 1)
    r = resumir([m])
    assert r["precisao"] == pytest.approx(0.5) and r["recall"] == pytest.approx(1.0)


def test_fechamento_reconecta_risco_fragmentado():
    m = np.zeros((20, 40), np.uint8)
    m[10, 5:18] = 1
    m[10, 20:35] = 1  # buraco de 2 px
    out = aplicar_morfologia(m, [{"op": "fechar", "kernel": 5, "forma": "quadrado"}])
    assert out[10, 18] == 1 and out[10, 19] == 1


def test_regra_rejeita_regiao_compacta_e_juncao():
    base = {"area_px": 200, "area_frac": 0.001, "alongamento": 1.2, "solidez": 0.9,
            "frac_reflexo": 0.0, "textura_entorno": 5.0, "toca_borda": False, "comprimento_frac": 0.05}
    f = {"area_min": 15, "alongamento_min": 2.5, "juncao_comp_frac": 0.15}
    assert regiao_aprovada(base, f) == (False, "pouco alongada")
    juncao = {**base, "alongamento": 30, "toca_borda": True, "comprimento_frac": 0.6}
    assert regiao_aprovada(juncao, f) == (False, "juncao de pecas")
    risco = {**base, "alongamento": 8}
    assert regiao_aprovada(risco, f) == (True, "arranhao")


def test_config_b_ignora_reflexo_e_acha_risco():
    img, ref = imagem_sintetica()
    r = processar(img, carregar_cfg("B_ajustado"))
    m = avaliar_imagem(r["predicao"], [ref])
    assert m["detectadas"] == 1
    assert m["fp"] == 0
    # o reflexo foi detectado pela mascara HSV
    assert r["reflexo"][350, 500] == 1
