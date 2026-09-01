# Vistoria Automática de Veículos — Detecção de Danos Antes/Depois

> Projeto Aplicado Longitudinal — Disciplina de Processamento de Imagens (PDI)
> Etapa atual: **M1**

## 1. Integrantes

- [Nome completo 1] — [usuário GitHub]
- [Nome completo 2] — [usuário GitHub]
- [Nome completo 3] — [usuário GitHub]

## 2. Problema investigado

Locadoras de veículos precisam comparar o estado do carro **antes** e **depois**
de um período de aluguel para identificar danos novos (riscos, amassados,
trincas) causados durante o uso pelo cliente. Hoje essa comparação costuma ser
feita manualmente, o que é lento e sujeito a erro humano.

Este projeto investiga como técnicas de Processamento Digital de Imagens podem
apoiar essa vistoria, comparando um par de fotos do mesmo veículo (antes e
depois) para destacar regiões onde surgiram divergências visuais compatíveis
com dano.

*(Detalhe completo em [`docs/proposta.md`](docs/proposta.md))*

## 3. Contexto de aplicação

Locadoras de veículos e sistemas de vistoria de frotas, no momento de
devolução do carro pelo cliente. Ver detalhamento em `docs/proposta.md`.

## 4. Objetivo geral

Desenvolver um pipeline de PDI capaz de, a partir de um par de imagens
(antes/depois) do mesmo veículo, indicar se há regiões com divergência visual
compatível com dano novo, e localizar essas regiões na imagem.

*(Objetivos específicos em `docs/proposta.md`, seção 3)*

## 5. Visão resumida da solução proposta

Pipeline: pré-processamento → alinhamento/registro das duas imagens →
comparação estrutural (diferença/SSIM) → segmentação das regiões divergentes
→ filtragem de falsos positivos (sombra, reflexo, sujeira) → resultado
(mapa de regiões candidatas a dano).

Ver pipeline detalhado em `docs/proposta.md`, seção 4.

## 6. Conjunto ou origem das imagens

- Dataset público **CarDD** (4.000 imagens de danos anotados, licença
  Flickr/Shutterstock — ver instruções de obtenção em `docs/proposta.md`).
- Fotos próprias capturadas pelo grupo (pares antes/depois, com dano
  simulado de forma reversível) — em `images/input/`.

Detalhes de origem, quantidade e licenciamento em `docs/proposta.md`, seção 5.

## 7. Estágio atual do projeto

Em andamento (etapa M1): definição do problema, levantamento inicial de
imagens e primeiros experimentos de alinhamento/diferença.

## 8. Organização do repositório

```text
projeto-vistoria-veiculos/
├── README.md              # este arquivo
├── docs/
│   └── proposta.md        # proposta técnica detalhada
├── images/
│   ├── input/              # imagens de entrada (antes/depois)
│   └── results/            # resultados de experimentos
├── src/                    # código-fonte (quando houver)
├── notebooks/               # notebooks exploratórios
├── tests/                  # testes (quando aplicável)
├── requirements.txt        # dependências Python
└── .gitignore
```

## 9. Tecnologias previstas

- Python
- OpenCV (pré-processamento, alinhamento ORB/SIFT, morfologia)
- scikit-image (SSIM, métricas de comparação estrutural)
- scikit-learn (etapa futura de classificação de tipo de dano, M2/M3)
- Jupyter Notebook (experimentos exploratórios)

## 10. Instruções para reproduzir experimentos

```bash
pip install -r requirements.txt
# instruções detalhadas de execução serão adicionadas
# conforme os experimentos forem implementados
```

## 11. Vídeo da M1

[Link para o vídeo não listado no YouTube] — a adicionar.

## 12. Documentação adicional

- [`docs/proposta.md`](docs/proposta.md) — proposta técnica completa
  (problema, contexto, objetivos, pipeline, viabilidade).
