# Vistoria Automática de Veículos — Detecção de Arranhões na Lataria

> Projeto Aplicado Longitudinal — Processamento de Imagens (PDI), 2026-02
> Etapa atual: **M2** (implementação parcial e experimentos)

## 1. Integrante

- Mateus Rachadel Lohn — [@Mateusrlohnn](https://github.com/Mateusrlohnn) (trabalho individual)

## 2. Problema

Locadoras precisam verificar, na devolução, se o carro voltou com avarias;
hoje isso é manual. O projeto investiga se técnicas clássicas de PDI
conseguem **localizar arranhões em uma foto próxima da lataria**,
distinguindo-os de reflexos, vãos entre peças e texturas.

Detalhes: [`docs/proposta.md`](docs/proposta.md) (revisada na M2).

## 3. Objetivos

- **Geral:** dada uma foto próxima de um painel, produzir a máscara das
  regiões compatíveis com arranhão e a descrição de cada região.
- **Específicos (M2):** pipeline executável de segmentação, morfologia,
  componentes e descritores; tratamento de reflexos e junções; comparação
  controlada entre o pipeline da M1 (A) e o da M2 (B).
- **M3:** classificador sobre os descritores e experimento antes/depois.

## 4. Entradas e saídas

| | Descrição |
|---|---|
| Entrada | foto colorida de um painel da lataria (reduzida a 640 px no maior lado) |
| Saída | máscara binária dos arranhões (PNG), CSV com descritores de cada região candidata (aprovada ou não, com o motivo), métricas contra a referência e painel com 8 saídas intermediárias |

## 5. Estágio atual

**M2 concluída como implementação parcial.** O fluxo entrada →
processamento → saída verificável funciona e foi avaliado:

| Config (teste, 54 imagens) | Precisão | Recall | F1 | IoU | FP/imagem |
|---|---:|---:|---:|---:|---:|
| A — pipeline da M1 | 0,071 | 0,287 | 0,114 | 0,011 | 8,50 |
| B — pipeline da M2 | 0,170 | 0,030 | 0,050 | 0,017 | 1,54 |

B reduz os falsos positivos em 82 % e dobra a precisão, mas perde recall.
As regras de reflexo e de junção não removeram nenhum arranhão real. A
análise completa, com casos de sucesso e falha, está no
[relatório da M2](docs/relatorio_m2.md).

**Pendente (M3):** corrigir a fragmentação dos traços, classificador
supervisionado sobre os descritores, experimento antes/depois com fotos
próprias e saída em bounding boxes.

## 6. Tecnologias

Python 3.14 · OpenCV 5.0 · scikit-image 0.26 · NumPy 2.4 · Matplotlib 3.10
· PyYAML · pytest · Jupyter. Versões fixas em
[`requirements.txt`](requirements.txt).

## 7. Estrutura

```text
projeto-vistoria-veiculos/
├── README.md
├── requirements.txt
├── configs/                  # configurações do pipeline (YAML)
│   ├── A_baseline_m1.yaml    #   A: Canny + dilatação + área (M1)
│   ├── B_proposto.yaml       #   B: valores iniciais (ponto de partida da busca)
│   └── B_ajustado.yaml       #   B: parâmetros escolhidos no conjunto de ajuste
├── src/
│   ├── pipeline.py           # pré-processamento, realce, reflexo, morfologia, regra
│   ├── descritores.py        # descritores por região (unidades e cálculo)
│   ├── data.py               # leitura de imagens e polígonos (COCO)
│   ├── avaliacao.py          # IoU/Dice, precisão/recall/F1 por região
│   ├── executar.py           # CLI: roda uma config e avalia
│   └── visualizacao.py       # painéis de saídas intermediárias
├── scripts/
│   ├── preparar_dados.py     # filtra, recorta e divide o dataset
│   ├── ajustar_parametros.py # busca em grade (só no ajuste)
│   └── analisar_resultados.py
├── data/
│   ├── catalogo.csv          # características de cada imagem usada
│   ├── split.csv             # ajuste / teste
│   ├── raw/                  # (não versionado) export da Roboflow
│   └── car_scratch/          # (não versionado) gerado por preparar_dados.py
├── results/
│   ├── ajuste_B/             # grade de parâmetros e execução no ajuste
│   ├── teste/                # experimento A × B: métricas, descritores, máscaras, analise.md
│   └── casos/                # painéis de sucesso e falha
├── notebooks/
│   ├── 01_experimento_preliminar.ipynb  # M1 (usava imagens removidas; ver commit 3f42cb6)
│   └── 02_pipeline_m2.ipynb             # demonstração da M2
├── tests/test_pipeline.py    # testes com imagens sintéticas
└── docs/
    ├── proposta.md           # proposta revisada
    ├── relatorio_m2.md       # relatório parcial da M2
    └── diario_m2.md          # decisões tomadas durante a M2
```

## 8. Dados

- **Dataset:** "Car scratch" v10 — PRISMAUTO, Roboflow Universe,
  licença **CC BY 4.0** —
  <https://universe.roboflow.com/prismauto/car-scratch-pizj9-uu3hr>.
- **Usado:** 67 fotos próximas da lataria (`IMG_5481`–`IMG_5667`, mesma
  câmera) com 921 polígonos de arranhão como referência. As outras 79
  imagens do export foram descartadas por origem incerta.
- **Divisão:** 13 imagens de ajuste e 54 de teste, pela ordem de captura
  (`data/split.csv`).
- **Não versionado.** As imagens da M1 (Google Imagens, sem licença) foram
  removidas do repositório.

### Como obter

1. Criar conta gratuita em <https://roboflow.com> e abrir o link acima.
2. Aba **Dataset** → versão **v10** → **Download Dataset** → formato
   **COCO Segmentation** → *Download zip to computer*.
3. Extrair em `data/raw/car_scratch_v10/` (deve conter `train/` e `valid/`).
4. `python scripts/preparar_dados.py`

## 9. Instalação

```bash
git clone https://github.com/Mateusrlohnn/projeto-vistoria-veiculos.git
cd projeto-vistoria-veiculos
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests          # 6 testes; não precisam do dataset
```

## 10. Execução

```bash
# 1. preparar os dados (após baixar o zip; ver seção 8)
python scripts/preparar_dados.py

# 2. experimento A x B no conjunto de teste
python -m src.executar --config configs/A_baseline_m1.yaml --dados data/car_scratch --conjunto teste --saida results/teste/A_baseline_m1
python -m src.executar --config configs/B_ajustado.yaml   --dados data/car_scratch --conjunto teste --saida results/teste/B_ajustado
python scripts/analisar_resultados.py

# 3. (opcional) refazer a busca de parâmetros no ajuste (~10 min)
python scripts/ajustar_parametros.py
```

Ou abrir `notebooks/02_pipeline_m2.ipynb` e executar *Restart & Run All*.

### Exemplo mínimo

```bash
python -m src.executar --config configs/B_ajustado.yaml --dados data/car_scratch \
    --conjunto teste --ids IMG_5498 --figuras 1 --saida results/exemplo
```

Saída esperada no terminal (trecho):

```text
"instancias_ref": 15, "detectadas": 7, "regioes_preditas": 13, "vp": 7, "fp": 6
```

e em `results/exemplo/`: `figuras/IMG_5498.png` (painel com as 8 etapas),
`mascaras/IMG_5498.png`, `descritores.csv`, `metricas_por_imagem.csv`,
`resumo.json`.

## 11. Resultados

- Relatório: [`docs/relatorio_m2.md`](docs/relatorio_m2.md) — pipeline,
  justificativas, experimento, casos de sucesso/falha, limitações e plano
  da M3.
- Tabelas: [`results/teste/analise.md`](results/teste/analise.md).
- Painéis: `results/casos/`.

## 12. Vídeos

- M1: <https://youtu.be/IxA_tRpA2tM>
- M2: *(link a adicionar)*

## 13. Documentação

- [`docs/proposta.md`](docs/proposta.md) — proposta técnica revisada.
- [`docs/relatorio_m2.md`](docs/relatorio_m2.md) — relatório parcial da M2.
- [`docs/diario_m2.md`](docs/diario_m2.md) — registro das decisões da M2.

## 14. Uso de Inteligência Artificial

Na M1, usei uma ferramenta de IA generativa para estruturar a
documentação e organizar o repositório. Na M2, usei o **Claude Code
(Anthropic)** para planejamento, busca do dataset, código, execução dos
experimentos e redação da documentação. Verifiquei tudo com testes
automatizados, execução real dos comandos e inspeção das saídas, e revisei
e aprovei cada etapa. Detalhes no
[relatório, seção 12](docs/relatorio_m2.md#12-declaração-de-uso-de-ia-generativa).
