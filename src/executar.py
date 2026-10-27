"""Executa uma configuracao do pipeline sobre um conjunto e avalia contra a referencia.

Uso:
    python -m src.executar --config configs/B_proposto.yaml --dados data/car_scratch \
        --conjunto teste --saida results/B_teste --figuras 6

Saidas em --saida:
    metricas_por_imagem.csv   IoU, Dice, detectadas, VP, FP por imagem
    descritores.csv           uma linha por regiao candidata (aprovada ou nao, com motivo)
    resumo.json               indicadores agregados + configuracao usada
    figuras/<id>.png          painel com saidas intermediarias (primeiras N imagens)
    mascaras/<id>.png         mascara final predita
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from .avaliacao import avaliar_imagem, pureza_regioes, resumir
from .data import carregar, ler_imagem
from .pipeline import processar
from .visualizacao import salvar_painel


def ler_split(arq, conjunto):
    """IDs do conjunto pedido segundo `split.csv` (gerado por scripts/preparar_dados.py)."""
    arq = Path(arq)
    if conjunto == "todos":
        return None
    if not arq.exists():
        raise SystemExit(f"{arq} nao encontrado; rode scripts/preparar_dados.py ou use --conjunto todos")
    with open(arq, encoding="utf-8") as f:
        return {l["id"] for l in csv.DictReader(f) if l["conjunto"] == conjunto}


def redimensionar_mascara(m, shape):
    return cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def executar(cfg, dados, conjunto, saida, n_figuras=0, categoria=None, split="data/split.csv",
             apenas=None):
    saida = Path(saida)
    (saida / "mascaras").mkdir(parents=True, exist_ok=True)
    if n_figuras:
        (saida / "figuras").mkdir(exist_ok=True)

    ids = ler_split(split, conjunto)
    amostras = [a for a in carregar(dados, categoria)
                if (ids is None or a.id in ids) and (not apenas or a.id in apenas)]
    if not amostras:
        raise SystemExit(f"Nenhuma imagem para o conjunto '{conjunto}' em {dados}")

    linhas, regioes = [], []
    for i, a in enumerate(amostras):
        r = processar(ler_imagem(a.caminho), cfg)
        shape = r["predicao"].shape
        insts = [redimensionar_mascara(m, shape) for m in a.instancias]

        met = avaliar_imagem(r["predicao"], insts)
        linhas.append({"id": a.id, **met})
        ref = np.zeros(shape, np.uint8)
        for m in insts:
            ref |= m
        pureza = pureza_regioes(r["rotulos"], ref)
        for d in r["descritores"]:
            regioes.append({"id": a.id, **d, "pureza_ref": round(pureza[d["rotulo"]], 3)})

        cv2.imwrite(str(saida / "mascaras" / f"{a.id}.png"), r["predicao"] * 255)
        if i < n_figuras:
            salvar_painel(r, ref, met, saida / "figuras" / f"{a.id}.png", titulo=cfg.get("nome", ""))

    _csv(saida / "metricas_por_imagem.csv", linhas)
    _csv(saida / "descritores.csv", regioes)
    resumo = resumir(linhas)
    with open(saida / "resumo.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "conjunto": conjunto, "resumo": resumo}, f, indent=2, ensure_ascii=False)
    return resumo, linhas


def _csv(caminho, linhas):
    if not linhas:
        return
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--dados", required=True, help="pasta com imagens + _annotations.coco.json ou anotacoes.json")
    p.add_argument("--conjunto", default="todos", choices=["ajuste", "teste", "todos"])
    p.add_argument("--saida", required=True)
    p.add_argument("--figuras", type=int, default=0, help="quantas imagens salvar como painel")
    p.add_argument("--categoria", default=None, help="filtra categoria COCO pelo nome (ex.: scratch)")
    p.add_argument("--split", default="data/split.csv", help="CSV id,conjunto")
    p.add_argument("--ids", nargs="+", default=None, help="processa so estas imagens (ex.: IMG_5502)")
    a = p.parse_args()

    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    resumo, _ = executar(cfg, a.dados, a.conjunto, a.saida, a.figuras, a.categoria, a.split, a.ids)
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
