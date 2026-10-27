"""Busca em grade dos parametros da configuracao B usando SOMENTE o conjunto de ajuste.

Duas camadas para nao repetir trabalho:
  1. parametros de segmentacao (kernel do top-hat, limiar por percentil, fechamento)
     -> executa o pipeline uma vez por combinacao, sem filtros;
  2. parametros da regra de decisao (area, alongamento, textura do entorno, juncao)
     -> reaproveita rotulos e descritores e so recalcula a mascara final.

Criterio de escolha: maior F1 por regiao no ajuste (desempate: maior IoU medio).
Saida: tabela completa em CSV e a melhor configuracao em YAML.

Uso:
    python scripts/ajustar_parametros.py --base configs/B_proposto.yaml \
        --saida results/ajuste_B --melhor configs/B_ajustado.yaml
"""
import argparse
import copy
import csv
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.avaliacao import avaliar_imagem, resumir  # noqa: E402
from src.data import carregar, ler_imagem  # noqa: E402
from src.executar import ler_split  # noqa: E402
from src.pipeline import processar, regiao_aprovada  # noqa: E402

GRADE_SEGMENTACAO = {
    "tophat_kernel": [9, 15, 21],
    "limiar": ["p96", "p98", "p99"],
    "fechar": [3, 5, 7],
}
GRADE_FILTROS = {
    "area_min": [15, 30, 60],
    "alongamento_min": [1.5, 2.5],
    "textura_entorno_max": [15, 20, 30, 256],
    "juncao_comp_frac": [0.15, 0.3],
}


def combinacoes(grade):
    chaves = list(grade)
    for valores in itertools.product(*grade.values()):
        yield dict(zip(chaves, valores))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="configs/B_proposto.yaml")
    p.add_argument("--dados", default="data/car_scratch")
    p.add_argument("--split", default="data/split.csv")
    p.add_argument("--saida", default="results/ajuste_B")
    p.add_argument("--melhor", default="configs/B_ajustado.yaml")
    a = p.parse_args()

    base = yaml.safe_load(open(a.base, encoding="utf-8"))
    ids = ler_split(a.split, "ajuste")
    amostras = [s for s in carregar(a.dados) if s.id in ids]
    imagens = {s.id: ler_imagem(s.caminho) for s in amostras}
    print(f"{len(amostras)} imagens de ajuste")

    linhas = []
    for seg in combinacoes(GRADE_SEGMENTACAO):
        cfg = copy.deepcopy(base)
        cfg["tophat_kernel"] = seg["tophat_kernel"]
        cfg["limiar"] = seg["limiar"]
        for op in cfg["morfologia"]:
            if op["op"] == "fechar":
                op["kernel"] = seg["fechar"]
        cfg["filtros"] = {}

        cache = []
        for s in amostras:
            r = processar(imagens[s.id], cfg)
            shape = r["rotulos"].shape
            insts = [cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
                     for m in s.instancias]
            cache.append((r["rotulos"], r["descritores"], insts))

        for filt in combinacoes(GRADE_FILTROS):
            f = {**base.get("filtros", {}), **filt}
            met = []
            for rotulos, descs, insts in cache:
                ok = [d["rotulo"] for d in descs if regiao_aprovada(d, f)[0]]
                pred = np.isin(rotulos, ok).astype(np.uint8)
                met.append(avaliar_imagem(pred, insts))
            res = resumir(met)
            linhas.append({**seg, **filt, **{k: round(v, 4) if isinstance(v, float) else v
                                              for k, v in res.items()}})
        print(f"segmentacao {seg} ok")

    linhas.sort(key=lambda l: (l["f1"], l["iou_medio"]), reverse=True)
    saida = Path(a.saida)
    saida.mkdir(parents=True, exist_ok=True)
    with open(saida / "grade.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    m = linhas[0]
    melhor = copy.deepcopy(base)
    melhor["nome"] = "B_ajustado"
    melhor["tophat_kernel"], melhor["limiar"] = m["tophat_kernel"], m["limiar"]
    for op in melhor["morfologia"]:
        if op["op"] == "fechar":
            op["kernel"] = m["fechar"]
    melhor["filtros"] = {**base.get("filtros", {}), **{k: m[k] for k in GRADE_FILTROS}}
    if melhor["filtros"]["textura_entorno_max"] == 256:
        del melhor["filtros"]["textura_entorno_max"]
    with open(a.melhor, "w", encoding="utf-8") as fh:
        fh.write("# Gerado por scripts/ajustar_parametros.py a partir de "
                 f"{a.base}; escolhido por F1 no conjunto de AJUSTE ({len(amostras)} imagens).\n")
        yaml.safe_dump(melhor, fh, sort_keys=False, allow_unicode=True)

    print(f"{len(linhas)} combinacoes avaliadas. Melhor: "
          f"{ {k: m[k] for k in list(GRADE_SEGMENTACAO) + list(GRADE_FILTROS)} }")
    print({k: m[k] for k in ("precisao", "recall", "f1", "iou_medio", "fp_por_imagem")})


if __name__ == "__main__":
    main()
