"""Analise complementar do experimento A x B no conjunto de teste.

Le as saidas de `src.executar` (mascaras e descritores) e gera `analise.md` com:
  1. resumo lado a lado das configuracoes;
  2. recall separado por tipo de anotacao original (scratch x tiny-scratch);
  3. regioes candidatas removidas por cada regra de decisao e quantas delas
     eram arranhao real (pureza_ref >= 0.5) -> custo de cada regra;
  4. imagens com maior e menor IoU da config B (candidatas a caso de sucesso/falha).

Uso:
    python scripts/analisar_resultados.py --resultados results/teste \
        --configs A_baseline_m1 B_ajustado
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import _mascara_poligonos  # noqa: E402

TOLERANCIA = 3
COBERTURA = 0.3


def instancias_por_tipo(dados):
    coco = json.loads((Path(dados) / "_annotations.coco.json").read_text(encoding="utf-8"))
    imgs = {i["id"]: i for i in coco["images"]}
    por_img = defaultdict(list)
    for a in coco["annotations"]:
        im = imgs[a["image_id"]]
        m = _mascara_poligonos(a["segmentation"], im["height"], im["width"])
        por_img[Path(im["file_name"]).stem].append((a["tipo_original"], m))
    return por_img


def recall_por_tipo(pasta, por_img):
    det, tot = Counter(), Counter()
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOLERANCIA + 1,) * 2)
    for arq in (pasta / "mascaras").glob("*.png"):
        pred = (cv2.imread(str(arq), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        pred_d = cv2.dilate(pred, se)
        for tipo, m in por_img.get(arq.stem, []):
            m = cv2.resize(m, pred.shape[::-1], interpolation=cv2.INTER_NEAREST)
            if not m.any():
                continue
            tot[tipo] += 1
            det[tipo] += pred_d[m == 1].mean() >= COBERTURA
    return {t: (int(det[t]), tot[t]) for t in sorted(tot)}


def remocoes_por_regra(pasta):
    cont = defaultdict(lambda: [0, 0])
    with open(pasta / "descritores.csv", encoding="utf-8") as f:
        for l in csv.DictReader(f):
            c = cont[l["motivo"]]
            c[0] += 1
            c[1] += float(l["pureza_ref"]) >= 0.5
    return dict(cont)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resultados", default="results/teste")
    p.add_argument("--configs", nargs="+", default=["A_baseline_m1", "B_ajustado"])
    p.add_argument("--dados", default="data/car_scratch")
    a = p.parse_args()

    raiz = Path(a.resultados)
    por_img = instancias_por_tipo(a.dados)
    out = ["# Analise do experimento no conjunto de teste", ""]

    out += ["## 1. Resumo", "",
            "| config | imagens | precisao | recall | F1 | IoU medio | Dice medio | regioes preditas | FP/imagem |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in a.configs:
        r = json.loads((raiz / c / "resumo.json").read_text(encoding="utf-8"))["resumo"]
        out.append(f"| {c} | {r['imagens']} | {r['precisao']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
                   f"{r['iou_medio']:.3f} | {r['dice_medio']:.3f} | {r['regioes_preditas']} | {r['fp_por_imagem']:.2f} |")

    out += ["", "## 2. Recall por tipo de anotacao", "",
            "| config | " + " | ".join(["tipo", "detectadas", "total", "recall"]) + " |", "|---|---|---:|---:|---:|"]
    for c in a.configs:
        for t, (d, n) in recall_por_tipo(raiz / c, por_img).items():
            out.append(f"| {c} | {t} | {d} | {n} | {d / n:.3f} |")

    out += ["", "## 3. Regioes candidatas por motivo de decisao", "",
            "Pureza >= 0.5 = regiao sobre arranhao real. Para regras de rejeicao, "
            "'reais' sao arranhoes perdidos pela regra.", "",
            "| config | motivo | regioes | reais | % reais |", "|---|---|---:|---:|---:|"]
    for c in a.configs:
        for mot, (n, reais) in sorted(remocoes_por_regra(raiz / c).items(), key=lambda x: -x[1][0]):
            out.append(f"| {c} | {mot} | {n} | {reais} | {100 * reais / n:.1f} |")

    ultima = raiz / a.configs[-1]
    with open(ultima / "metricas_por_imagem.csv", encoding="utf-8") as f:
        linhas = sorted(csv.DictReader(f), key=lambda l: float(l["iou"]), reverse=True)
    out += ["", f"## 4. Imagens extremas ({a.configs[-1]})", "",
            "| id | IoU | detectadas/ref | VP | FP |", "|---|---:|---:|---:|---:|"]
    for l in linhas[:5] + linhas[-5:]:
        out.append(f"| {l['id']} | {float(l['iou']):.3f} | {l['detectadas']}/{l['n_ref']} | {l['vp']} | {l['fp']} |")

    (raiz / "analise.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
