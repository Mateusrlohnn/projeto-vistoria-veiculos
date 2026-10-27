"""Pipeline de deteccao de arranhoes: entrada -> realce -> segmentacao ->
morfologia -> componentes conexos -> descritores -> filtragem por regra.

Todas as etapas sao controladas por um dicionario de configuracao
(ver `configs/*.yaml`). A funcao `processar` devolve as saidas
intermediarias para inspecao e avaliacao.
"""
import cv2
import numpy as np
from skimage import measure

from .descritores import extrair_descritores

# ---------------------------------------------------------------- pre-processamento


def redimensionar(img, lado_max):
    """Reduz a imagem para que o maior lado tenha `lado_max` px (mantem proporcao).

    Padroniza a escala para que parametros em pixels (kernels, areas) tenham
    o mesmo significado em todas as imagens.
    """
    h, w = img.shape[:2]
    escala = lado_max / max(h, w)
    if escala >= 1:
        return img, 1.0
    return cv2.resize(img, (round(w * escala), round(h * escala)), interpolation=cv2.INTER_AREA), escala


def preprocessar(img_bgr, cfg):
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    if cfg.get("clahe"):
        # equalizacao local de contraste: compensa iluminacao desigual na lataria
        cinza = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cinza)
    k = cfg.get("blur_ksize", 5)
    suave = cv2.GaussianBlur(cinza, (k, k), 0) if k > 1 else cinza
    return cinza, suave, hsv

# ---------------------------------------------------------------- segmentacao


def realce_canny(suave, cfg):
    """Mapa binario de bordas (abordagem da M1)."""
    return (cv2.Canny(suave, cfg["canny_low"], cfg["canny_high"]) > 0).astype(np.uint8)


def realce_tophat(suave, cfg):
    """Realce de estruturas finas por top-hat + black-hat morfologicos.

    Black-hat (fechamento - imagem) destaca estruturas finas mais escuras que o
    entorno; top-hat (imagem - abertura) destaca as mais claras. Um arranhao
    pode aparecer claro (verniz removido) ou escuro (sulco), por isso usa-se o
    maximo dos dois. Objetos maiores que o elemento estruturante (portas,
    reflexos largos) sao suprimidos.
    """
    k = cfg["tophat_kernel"]
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(suave, cv2.MORPH_BLACKHAT, se)
    th = cv2.morphologyEx(suave, cv2.MORPH_TOPHAT, se)
    resp = np.maximum(bh, th)
    limiar = cfg.get("limiar", "otsu")
    if limiar == "otsu":
        _, binaria = cv2.threshold(resp, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif isinstance(limiar, str) and limiar.startswith("p"):
        # percentil: mantem os (100 - p)% pixels de maior resposta. Otsu falha
        # quando o histograma e quase todo fundo (arranhoes ocupam < 1% da imagem).
        binaria = (resp > np.percentile(resp, float(limiar[1:]))).astype(np.uint8)
    else:
        binaria = (resp >= cfg["limiar"]).astype(np.uint8)
    return binaria, resp


def mascara_reflexo(hsv, cfg):
    """Regioes de reflexo especular: muito claras (V alto) e pouco saturadas (S baixo).

    Reflexo de sol/lampada na pintura gera bordas fortes que imitam arranhao.
    A mascara e dilatada para cobrir tambem a borda do reflexo.
    """
    v, s = hsv[..., 2], hsv[..., 1]
    m = ((v >= cfg["v_min"]) & (s <= cfg["s_max"])).astype(np.uint8)
    d = cfg.get("dilatar", 0)
    if d:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    return m

# ---------------------------------------------------------------- morfologia


def aplicar_morfologia(binaria, operacoes):
    """Aplica em sequencia operacoes do tipo {op, kernel, iter, forma}."""
    ops = {"dilatar": cv2.MORPH_DILATE, "erodir": cv2.MORPH_ERODE,
           "abrir": cv2.MORPH_OPEN, "fechar": cv2.MORPH_CLOSE}
    formas = {"elipse": cv2.MORPH_ELLIPSE, "quadrado": cv2.MORPH_RECT, "cruz": cv2.MORPH_CROSS}
    out = binaria.copy()
    for o in operacoes or []:
        k = o["kernel"]
        se = cv2.getStructuringElement(formas[o.get("forma", "quadrado")], (k, k))
        out = cv2.morphologyEx(out, ops[o["op"]], se, iterations=o.get("iter", 1))
    return out

# ---------------------------------------------------------------- regra de decisao


def regiao_aprovada(d, f):
    """Regra verificavel que transforma uma regiao candidata em 'arranhao'.

    Cada limiar e opcional; ausente = criterio desligado.
    """
    if d["area_px"] < f.get("area_min", 0):
        return False, "area pequena"
    if "area_max_frac" in f and d["area_frac"] > f["area_max_frac"]:
        return False, "area grande"
    if d["alongamento"] < f.get("alongamento_min", 0):
        return False, "pouco alongada"
    if d["solidez"] > f.get("solidez_max", 1.01):
        return False, "compacta"
    if d["frac_reflexo"] > f.get("frac_reflexo_max", 1.01):
        return False, "reflexo"
    if d["textura_entorno"] > f.get("textura_entorno_max", 256):
        # pneu, roda, asfalto, grade: entorno texturizado; arranhao fica sobre pintura lisa
        return False, "entorno texturizado"
    if "juncao_comp_frac" in f and d["toca_borda"] and d["comprimento_frac"] > f["juncao_comp_frac"]:
        # vaos entre portas/paineis atravessam a foto e encostam na borda
        return False, "juncao de pecas"
    if "borda_imagem" in f and d["toca_borda"] and d["area_frac"] > f["borda_imagem"]:
        return False, "contorno da peca"
    return True, "arranhao"

# ---------------------------------------------------------------- fluxo completo


def processar(img_bgr, cfg):
    """Executa o pipeline completo. Devolve dict com saidas intermediarias.

    Chaves: img, cinza, realce (0..255), binaria, reflexo, morfologia,
    rotulos, descritores (lista de dicts), predicao (mascara final 0/1).
    """
    img, escala = redimensionar(img_bgr, cfg.get("lado_max", 640))
    cinza, suave, hsv = preprocessar(img, cfg)

    met = cfg["metodo"]
    if met == "canny":
        binaria = realce_canny(suave, cfg)
        realce = binaria * 255
    elif met == "tophat":
        binaria, realce = realce_tophat(suave, cfg)
    else:
        raise ValueError(f"metodo desconhecido: {met}")

    reflexo = mascara_reflexo(hsv, cfg["reflexo"]) if cfg.get("reflexo") else np.zeros_like(binaria)
    if cfg.get("reflexo", {}).get("remover_antes"):
        binaria = binaria & (1 - reflexo)

    morf = aplicar_morfologia(binaria, cfg.get("morfologia"))
    rotulos = measure.label(morf, connectivity=2)

    descritores, predicao = [], np.zeros_like(morf)
    for reg in measure.regionprops(rotulos, intensity_image=cinza):
        d = extrair_descritores(reg, cinza, hsv, reflexo, img.shape)
        d["aprovada"], d["motivo"] = regiao_aprovada(d, cfg.get("filtros", {}))
        descritores.append(d)
        if d["aprovada"]:
            predicao[rotulos == reg.label] = 1

    return {"img": img, "escala": escala, "cinza": cinza, "realce": realce,
            "binaria": binaria, "reflexo": reflexo, "morfologia": morf,
            "rotulos": rotulos, "descritores": descritores, "predicao": predicao}
