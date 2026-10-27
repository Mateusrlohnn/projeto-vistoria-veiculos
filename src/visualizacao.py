"""Paineis de saidas intermediarias para relatorio e video."""
import cv2
import numpy as np
from matplotlib.figure import Figure  # sem pyplot: nao altera o backend do notebook
from skimage.color import label2rgb


def sobrepor(img_bgr, pred, ref):
    """Verde = referencia, vermelho = predicao, amarelo = acerto (sobreposicao)."""
    out = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    cor = np.zeros_like(out)
    cor[ref == 1] = (0, 255, 0)
    cor[pred == 1] = (255, 0, 0)
    cor[(ref == 1) & (pred == 1)] = (255, 255, 0)
    m = (ref | pred) == 1
    out[m] = (0.4 * out[m] + 0.6 * cor[m]).astype(np.uint8)
    return out


def salvar_painel(r, ref, met, caminho, titulo=""):
    rgb = cv2.cvtColor(r["img"], cv2.COLOR_BGR2RGB)
    aprov = sum(d["aprovada"] for d in r["descritores"])
    paineis = [
        (rgb, "1. entrada"),
        (r["realce"], "2. realce"),
        (r["binaria"] * 255, "3. segmentacao binaria"),
        (r["reflexo"] * 255, "4. mascara de reflexo (HSV)"),
        (r["morfologia"] * 255, "5. apos morfologia"),
        (label2rgb(r["rotulos"], bg_label=0), f"6. componentes ({r['rotulos'].max()})"),
        (r["predicao"] * 255, f"7. regioes aprovadas ({aprov})"),
        (sobrepor(r["img"], r["predicao"], ref),
         f"8. vs referencia  IoU={met['iou']:.2f} FP={met['fp']}"),
    ]
    fig = Figure(figsize=(20, 9))
    eixos = fig.subplots(2, 4)
    for ax, (im, t) in zip(eixos.ravel(), paineis):
        ax.imshow(im, cmap=None if im.ndim == 3 else "gray")
        ax.set_title(t, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{titulo} - {caminho.stem}", fontsize=13)
    fig.tight_layout()
    fig.savefig(caminho, dpi=90)
