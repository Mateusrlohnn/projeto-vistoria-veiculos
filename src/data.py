"""Leitura das imagens e das referencias (mascaras de arranhao).

Dois formatos aceitos:

- COCO Segmentation (export da Roboflow): pasta com imagens e
  `_annotations.coco.json`. Cada anotacao (poligono ou bbox) vira uma
  instancia de referencia.
- Caixas proprias: pasta com imagens e `anotacoes.json` no formato
  {"arquivo.jpg": [[x, y, w, h], ...]}, gerado por `scripts/anotar_caixas.py`.
  Lista vazia = imagem sem arranhao (negativo).
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

EXTENSOES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


@dataclass
class Amostra:
    id: str
    caminho: Path
    instancias: list = field(default_factory=list)  # mascaras binarias (uint8 0/1), uma por arranhao

    def mascara_referencia(self, shape):
        ref = np.zeros(shape[:2], np.uint8)
        for m in self.instancias:
            ref |= m
        return ref


def ler_imagem(caminho):
    img = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Nao foi possivel carregar {caminho}")
    return img


def _mascara_poligonos(segmentacao, h, w):
    m = np.zeros((h, w), np.uint8)
    for poli in segmentacao:
        pts = np.array(poli, np.float32).reshape(-1, 2).round().astype(np.int32)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
    return m


def _mascara_caixa(x, y, bw, bh, h, w):
    m = np.zeros((h, w), np.uint8)
    x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
    x1, y1 = min(int(round(x + bw)), w), min(int(round(y + bh)), h)
    m[y0:y1, x0:x1] = 1
    return m


def carregar_coco(pasta, categoria=None):
    """Le um split exportado em COCO. `categoria` filtra pelo nome (ex.: 'scratch')."""
    pasta = Path(pasta)
    with open(pasta / "_annotations.coco.json", encoding="utf-8") as f:
        coco = json.load(f)

    nomes = {c["id"]: c["name"].lower() for c in coco["categories"]}
    por_imagem = {}
    for a in coco["annotations"]:
        if categoria and categoria.lower() not in nomes[a["category_id"]]:
            continue
        por_imagem.setdefault(a["image_id"], []).append(a)

    amostras = []
    for im in sorted(coco["images"], key=lambda i: i["file_name"]):
        h, w = im["height"], im["width"]
        insts = []
        for a in por_imagem.get(im["id"], []):
            seg = a.get("segmentation")
            if isinstance(seg, list) and seg:
                insts.append(_mascara_poligonos(seg, h, w))
            else:
                insts.append(_mascara_caixa(*a["bbox"], h, w))
        amostras.append(Amostra(Path(im["file_name"]).stem, pasta / im["file_name"], insts))
    return amostras


def carregar_caixas(pasta):
    """Le fotos proprias anotadas com `scripts/anotar_caixas.py`."""
    pasta = Path(pasta)
    with open(pasta / "anotacoes.json", encoding="utf-8") as f:
        anot = json.load(f)

    amostras = []
    for nome in sorted(anot):
        caminho = pasta / nome
        h, w = ler_imagem(caminho).shape[:2]
        insts = [_mascara_caixa(*cx, h, w) for cx in anot[nome]]
        amostras.append(Amostra(Path(nome).stem, caminho, insts))
    return amostras


def carregar(pasta, categoria=None):
    pasta = Path(pasta)
    if (pasta / "_annotations.coco.json").exists():
        return carregar_coco(pasta, categoria)
    if (pasta / "anotacoes.json").exists():
        return carregar_caixas(pasta)
    raise FileNotFoundError(
        f"{pasta} nao tem _annotations.coco.json nem anotacoes.json")
