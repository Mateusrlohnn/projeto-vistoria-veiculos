# Analise do experimento no conjunto de teste

## 1. Resumo

| config | imagens | precisao | recall | F1 | IoU medio | Dice medio | regioes preditas | FP/imagem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A_baseline_m1 | 54 | 0.071 | 0.287 | 0.114 | 0.011 | 0.022 | 494 | 8.50 |
| B_ajustado | 54 | 0.170 | 0.030 | 0.050 | 0.017 | 0.027 | 100 | 1.54 |

## 2. Recall por tipo de anotacao

| config | tipo | detectadas | total | recall |
|---|---|---:|---:|---:|
| A_baseline_m1 | scratch | 132 | 385 | 0.343 |
| A_baseline_m1 | tiny-scratch | 81 | 357 | 0.227 |
| B_ajustado | scratch | 21 | 385 | 0.055 |
| B_ajustado | tiny-scratch | 1 | 357 | 0.003 |

## 3. Regioes candidatas por motivo de decisao

Pureza >= 0.5 = regiao sobre arranhao real. Para regras de rejeicao, 'reais' sao arranhoes perdidos pela regra.

| config | motivo | regioes | reais | % reais |
|---|---|---:|---:|---:|
| A_baseline_m1 | arranhao | 494 | 35 | 7.1 |
| A_baseline_m1 | area pequena | 34 | 4 | 11.8 |
| B_ajustado | area pequena | 3732 | 85 | 2.3 |
| B_ajustado | entorno texturizado | 877 | 14 | 1.6 |
| B_ajustado | pouco alongada | 818 | 16 | 2.0 |
| B_ajustado | arranhao | 100 | 17 | 17.0 |
| B_ajustado | reflexo | 24 | 0 | 0.0 |
| B_ajustado | juncao de pecas | 6 | 0 | 0.0 |

## 4. Imagens extremas (B_ajustado)

| id | IoU | detectadas/ref | VP | FP |
|---|---:|---:|---:|---:|
| IMG_5502 | 0.377 | 1/1 | 1 | 1 |
| IMG_5521 | 0.223 | 2/20 | 2 | 0 |
| IMG_5498 | 0.222 | 7/15 | 7 | 6 |
| IMG_5547 | 0.051 | 9/60 | 3 | 11 |
| IMG_5522 | 0.022 | 1/7 | 1 | 14 |
| IMG_5579 | 0.000 | 0/2 | 0 | 1 |
| IMG_5581 | 0.000 | 0/25 | 0 | 0 |
| IMG_5620 | 0.000 | 0/5 | 0 | 0 |
| IMG_5621 | 0.000 | 0/7 | 0 | 0 |
| IMG_5667 | 0.000 | 0/8 | 0 | 1 |
