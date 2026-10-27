"""Metricas de comparacao entre a predicao e a referencia.

Nivel de pixel (por imagem, depois media nas imagens com arranhao):
  - IoU  = |P ∩ R| / |P ∪ R|
  - Dice = 2|P ∩ R| / (|P| + |R|)

Nivel de regiao (contagem acumulada no conjunto):
  - instancia de referencia DETECTADA se pelo menos `cobertura_min` dos seus
    pixels sao cobertos pela predicao dilatada em `tolerancia_px`
    (arranhoes sao finos; poucos pixels de deslocamento nao devem contar como erro);
  - regiao predita CORRETA (VP) se pelo menos `pureza_min` dos seus pixels caem
    dentro da referencia dilatada em `tolerancia_px`; caso contrario e falso positivo.
  - precisao = VP / regioes preditas; recall = detectadas / instancias; F1.

Imagens sem arranhao (referencia vazia) entram so nas contagens de falso positivo.
"""
import cv2
import numpy as np
from skimage import measure


def _dilatar(m, t):
    if t <= 0:
        return m
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * t + 1, 2 * t + 1)))


def iou_dice(pred, ref):
    inter = int((pred & ref).sum())
    uniao = int((pred | ref).sum())
    soma = int(pred.sum() + ref.sum())
    if soma == 0:  # imagem negativa corretamente sem predicao
        return 1.0, 1.0
    return inter / uniao, 2 * inter / soma


def avaliar_imagem(pred, instancias, tolerancia_px=3, cobertura_min=0.3, pureza_min=0.5):
    """`pred`: mascara 0/1. `instancias`: lista de mascaras 0/1 no mesmo tamanho."""
    ref = np.zeros_like(pred)
    for m in instancias:
        ref |= m
    iou, dice = iou_dice(pred, ref)

    pred_d = _dilatar(pred, tolerancia_px)
    detectadas = sum(
        1 for m in instancias if m.sum() and pred_d[m == 1].mean() >= cobertura_min)

    ref_d = _dilatar(ref, tolerancia_px)
    rot = measure.label(pred, connectivity=2)
    n_pred = int(rot.max())
    vp = sum(1 for r in measure.regionprops(rot)
             if ref_d[r.coords[:, 0], r.coords[:, 1]].mean() >= pureza_min)

    return {"iou": iou, "dice": dice, "n_ref": len(instancias), "detectadas": detectadas,
            "n_pred": n_pred, "vp": vp, "fp": n_pred - vp}


def resumir(linhas):
    """Agrega metricas por imagem em indicadores do conjunto."""
    n_ref = sum(l["n_ref"] for l in linhas)
    det = sum(l["detectadas"] for l in linhas)
    n_pred = sum(l["n_pred"] for l in linhas)
    vp = sum(l["vp"] for l in linhas)
    prec = vp / n_pred if n_pred else 0.0
    rec = det / n_ref if n_ref else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    pos = [l for l in linhas if l["n_ref"] > 0]
    return {
        "imagens": len(linhas),
        "iou_medio": float(np.mean([l["iou"] for l in pos])) if pos else 0.0,
        "dice_medio": float(np.mean([l["dice"] for l in pos])) if pos else 0.0,
        "instancias_ref": n_ref, "detectadas": det, "regioes_preditas": n_pred,
        "vp": vp, "fp": n_pred - vp,
        "fp_por_imagem": (n_pred - vp) / len(linhas) if linhas else 0.0,
        "precisao": prec, "recall": rec, "f1": f1,
    }
