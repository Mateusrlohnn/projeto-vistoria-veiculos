"""Prepara o subconjunto de avaliacao a partir do export Roboflow "Car scratch" v10.

Origem: https://universe.roboflow.com/prismauto/car-scratch-pizj9-uu3hr (versao 10,
CC BY 4.0, export "COCO Segmentation"). O dataset NAO e versionado; ver README.

O que faz:
  1. junta os splits train/ e valid/ do export;
  2. mantem so as fotos IMG_xxxx (mesma camera, close-ups da lataria) e descarta
     as demais (origem incerta: nomes no padrao do CarDD, Adobe Stock, WhatsApp...);
  3. recorta as faixas pretas inseridas pelo resize "Fit (black edges)" da Roboflow
     e desloca os poligonos de acordo;
  4. une as categorias 'scratch' e 'tiny-scratch' em uma unica classe 'arranhao';
  5. divide ajuste/teste pela ordem de captura (numero do IMG), para que fotos
     consecutivas do mesmo carro nao fiquem nos dois conjuntos;
  6. grava data/car_scratch/ (imagens + _annotations.coco.json),
     data/split.csv e data/catalogo.csv (estes dois versionados).

Uso:
    python scripts/preparar_dados.py --origem data/raw/car_scratch_v10
"""
import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np

# fronteira ajuste/teste: maior salto de numeracao no inicio da sequencia (5493 -> 5498)
ULTIMO_AJUSTE = 5493


def area_util(img, limiar=10):
    """Bounding box do conteudo, ignorando faixas pretas de padding."""
    mx = img.max(axis=2)
    linhas = np.where(mx.max(axis=1) > limiar)[0]
    colunas = np.where(mx.max(axis=0) > limiar)[0]
    return int(colunas[0]), int(linhas[0]), int(colunas[-1] + 1), int(linhas[-1] + 1)  # x0, y0, x1, y1


def caracterizar(img, mascara):
    """Atributos simples para o catalogo (condicoes de aquisicao)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    reflexo = (hsv[..., 2] >= 230) & (hsv[..., 1] <= 40)
    return {
        "brilho_medio": round(float(hsv[..., 2].mean()), 1),
        "saturacao_media": round(float(hsv[..., 1].mean()), 1),
        "frac_reflexo": round(float(reflexo.mean()), 4),
        "frac_area_anotada": round(float(mascara.mean()), 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--origem", default="data/raw/car_scratch_v10")
    p.add_argument("--destino", default="data/car_scratch")
    p.add_argument("--meta", default="data", help="onde gravar split.csv e catalogo.csv")
    a = p.parse_args()

    origem, destino, meta = Path(a.origem), Path(a.destino), Path(a.meta)
    destino.mkdir(parents=True, exist_ok=True)

    imagens, anotacoes, catalogo, descartadas = [], [], [], []
    for split in ("train", "valid", "test"):
        arq = origem / split / "_annotations.coco.json"
        if not arq.exists():
            continue
        coco = json.loads(arq.read_text(encoding="utf-8"))
        nomes_cat = {c["id"]: c["name"] for c in coco["categories"]}
        por_img = {}
        for an in coco["annotations"]:
            por_img.setdefault(an["image_id"], []).append(an)

        for im in coco["images"]:
            original = re.sub(r"_(jpg|jpeg|png)\.rf\..*", "", im["file_name"], flags=re.I)
            m = re.fullmatch(r"IMG_(\d+)", original)
            if not m:
                descartadas.append(original)
                continue

            img = cv2.imread(str(origem / split / im["file_name"]))
            x0, y0, x1, y1 = area_util(img)
            rec = img[y0:y1, x0:x1]
            h, w = rec.shape[:2]
            novo_id = len(imagens) + 1
            nome = f"{original}.jpg"
            cv2.imwrite(str(destino / nome), rec, [cv2.IMWRITE_JPEG_QUALITY, 95])
            imagens.append({"id": novo_id, "file_name": nome, "width": w, "height": h})

            mascara = np.zeros((h, w), np.uint8)
            n_tiny = 0
            for an in por_img.get(im["id"], []):
                segs = []
                for poli in an["segmentation"]:
                    pts = np.array(poli, np.float32).reshape(-1, 2) - (x0, y0)
                    pts = np.clip(pts, 0, (w - 1, h - 1))
                    segs.append(pts.round(1).ravel().tolist())
                    cv2.fillPoly(mascara, [pts.round().astype(np.int32)], 1)
                bx, by, bw, bh = an["bbox"]
                n_tiny += "tiny" in nomes_cat[an["category_id"]]
                anotacoes.append({
                    "id": len(anotacoes) + 1, "image_id": novo_id, "category_id": 1,
                    "segmentation": segs, "bbox": [bx - x0, by - y0, bw, bh],
                    "area": an.get("area", bw * bh), "iscrowd": 0,
                    "tipo_original": nomes_cat[an["category_id"]],
                })

            numero = int(m.group(1))
            n = len(por_img.get(im["id"], []))
            catalogo.append({
                "id": original, "numero": numero,
                "conjunto": "ajuste" if numero <= ULTIMO_AJUSTE else "teste",
                "largura": w, "altura": h, "n_arranhoes": n, "n_tiny": n_tiny,
                **caracterizar(rec, mascara),
            })

    coco_saida = {
        "info": {"descricao": "Subconjunto IMG_ do Roboflow 'Car scratch' v10 (PRISMAUTO, CC BY 4.0), "
                              "faixas pretas recortadas, classes unidas em 'arranhao'"},
        "categories": [{"id": 1, "name": "arranhao"}],
        "images": imagens, "annotations": anotacoes,
    }
    (destino / "_annotations.coco.json").write_text(json.dumps(coco_saida), encoding="utf-8")

    catalogo.sort(key=lambda c: c["numero"])
    meta.mkdir(exist_ok=True)
    with open(meta / "catalogo.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(catalogo[0].keys()))
        wr.writeheader()
        wr.writerows(catalogo)
    with open(meta / "split.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["id", "conjunto"])
        wr.writerows((c["id"], c["conjunto"]) for c in catalogo)

    n_aj = sum(c["conjunto"] == "ajuste" for c in catalogo)
    print(f"mantidas: {len(catalogo)} (ajuste {n_aj}, teste {len(catalogo) - n_aj}); "
          f"anotacoes: {len(anotacoes)}; descartadas: {len(descartadas)}")


if __name__ == "__main__":
    main()
