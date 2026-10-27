"""Descritores de forma, intensidade e cor por regiao (componente conexo).

Unidades: pixels da imagem redimensionada (maior lado = `lado_max`).

| atributo         | unidade | calculo |
|------------------|---------|---------|
| area_px          | px^2    | numero de pixels da regiao |
| area_frac        | -       | area_px / area da imagem |
| perimetro_px     | px      | perimetro do contorno (skimage, vizinhanca 8) |
| eixo_maior_px    | px      | eixo maior da elipse com mesmos momentos de 2a ordem |
| eixo_menor_px    | px      | eixo menor da mesma elipse |
| alongamento      | -       | eixo_maior / max(eixo_menor, 1) |
| excentricidade   | -       | excentricidade da elipse (0 = circulo, ~1 = linha) |
| solidez          | -       | area / area do fecho convexo |
| circularidade    | -       | 4*pi*area / perimetro^2 (1 = circulo) |
| orientacao_graus | graus   | angulo do eixo maior em relacao ao eixo horizontal |
| intensidade_media| 0-255   | media do nivel de cinza dentro da regiao |
| comprimento_frac | -       | eixo_maior / diagonal da imagem |
| contraste_local  | 0-255   | |media dentro - media do anel de 5 px ao redor| |
| textura_entorno  | 0-255   | desvio-padrao do nivel de cinza no anel de 5 px ao redor |
| saturacao_media  | 0-255   | media do canal S (HSV) dentro da regiao |
| frac_reflexo     | -       | fracao dos pixels da regiao dentro da mascara de reflexo |
| toca_borda       | bool    | bounding box encosta na borda da imagem |
"""
import math

import cv2
import numpy as np


def extrair_descritores(reg, cinza, hsv, reflexo, shape):
    h, w = shape[:2]
    minr, minc, maxr, maxc = reg.bbox
    area = int(reg.area)
    perim = float(reg.perimeter) if reg.perimeter > 0 else 1.0
    eixo_maior = float(reg.axis_major_length)
    eixo_menor = float(reg.axis_minor_length)

    # recorte com margem para calcular o contraste com a vizinhanca
    m = 5
    r0, c0 = max(minr - m, 0), max(minc - m, 0)
    r1, c1 = min(maxr + m, h), min(maxc + m, w)
    local = np.zeros((r1 - r0, c1 - c0), np.uint8)
    rr, cc = reg.coords[:, 0] - r0, reg.coords[:, 1] - c0
    local[rr, cc] = 1
    anel = cv2.dilate(local, np.ones((2 * m + 1, 2 * m + 1), np.uint8)) & (1 - local)
    rec = cinza[r0:r1, c0:c1]
    media_dentro = float(rec[local == 1].mean())
    media_anel = float(rec[anel == 1].mean()) if anel.any() else media_dentro
    textura_anel = float(rec[anel == 1].std()) if anel.any() else 0.0

    return {
        "rotulo": int(reg.label),
        "area_px": area,
        "area_frac": area / (h * w),
        "perimetro_px": round(perim, 1),
        "eixo_maior_px": round(eixo_maior, 1),
        "eixo_menor_px": round(eixo_menor, 1),
        "alongamento": round(eixo_maior / max(eixo_menor, 1.0), 2),
        "excentricidade": round(float(reg.eccentricity), 3),
        "solidez": round(float(reg.solidity), 3),
        "circularidade": round(4 * math.pi * area / perim ** 2, 3),
        "orientacao_graus": round(math.degrees(reg.orientation), 1),
        "intensidade_media": round(media_dentro, 1),
        "comprimento_frac": round(eixo_maior / math.hypot(h, w), 3),
        "contraste_local": round(abs(media_dentro - media_anel), 1),
        "textura_entorno": round(textura_anel, 1),
        "saturacao_media": round(float(hsv[..., 1][reg.coords[:, 0], reg.coords[:, 1]].mean()), 1),
        "frac_reflexo": round(float(reflexo[reg.coords[:, 0], reg.coords[:, 1]].mean()), 3),
        "toca_borda": bool(minr == 0 or minc == 0 or maxr == h or maxc == w),
        "bbox_xywh": (int(minc), int(minr), int(maxc - minc), int(maxr - minr)),
    }
