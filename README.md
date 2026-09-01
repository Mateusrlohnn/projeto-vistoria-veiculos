# Vistoria Automática de Veículos — Detecção de Avarias na Lataria

> Projeto Aplicado Longitudinal — Disciplina de Processamento de Imagens (PDI)
> Etapa atual: **M1**

## 1. Integrante

- Mateus Rachadel Lohn — [usuário GitHub]

## 2. Problema investigado

Locadoras de veículos precisam verificar, no momento da devolução, se o carro
retornou com alguma avaria (risco, amassado, trinca). Hoje essa verificação
costuma ser feita manualmente, o que é lento e sujeito a erro humano.

Este projeto investiga como técnicas de Processamento Digital de Imagens podem
reconhecer e segmentar diretamente padrões de avaria na lataria do veículo a
partir de uma única foto tirada na inspeção, sem depender de comparação com
uma foto anterior.

*(Detalhe completo em [`docs/proposta.md`](docs/proposta.md))*

## 3. Contexto de aplicação

Locadoras de veículos e sistemas de vistoria de frotas, no momento de
devolução do carro pelo cliente. Ver detalhamento em `docs/proposta.md`.

## 4. Objetivo geral

Desenvolver um pipeline de PDI capaz de, a partir de uma foto do veículo no
momento da inspeção, localizar regiões com padrão visual compatível com
avaria (bounding box) e atribuir uma classificação inicial ao tipo de padrão
encontrado.

*(Objetivos específicos em `docs/proposta.md`, seção 3)*

## 5. Visão resumida da solução proposta

Pipeline: pré-processamento (escala de cinza + HSV) → filtro de bordas
(Canny/Sobel) → segmentação das regiões candidatas → destaque do dano
(bounding box + classificação inicial heurística).

Ver pipeline detalhado em `docs/proposta.md`, seção 4.

## 6. Conjunto ou origem das imagens

- 20 a 50 imagens de veículos com avarias (arranhões, amassados),
  coletadas manualmente via pesquisa no Google Imagens, para uso
  exclusivamente educacional — em `images/input/`.
- Dataset público **CarDD** como candidato para ampliar o conjunto em
  M2/M3 (licença Flickr/Shutterstock — ver `docs/proposta.md`).

Detalhes de origem, quantidade e licenciamento em `docs/proposta.md`, seção 5.

## 7. Estágio atual do projeto

Em andamento (etapa M1): definição do problema, levantamento inicial de
imagens e primeiros experimentos de detecção de bordas/segmentação.

## 8. Organização do repositório

```text
projeto-vistoria-veiculos/
├── README.md              # este arquivo
├── docs/
│   └── proposta.md        # proposta técnica detalhada
├── images/
│   ├── input/              # imagens de entrada (fotos de veículos para inspeção)
│   └── results/            # resultados de experimentos
├── src/                    # código-fonte (quando houver)
├── notebooks/               # notebooks exploratórios
├── tests/                  # testes (quando aplicável)
├── requirements.txt        # dependências Python
└── .gitignore
```

## 9. Tecnologias previstas

- Python
- OpenCV (pré-processamento, detecção de bordas Canny/Sobel, morfologia)
- scikit-image (segmentação, métricas de imagem)
- scikit-learn (etapa futura de classificação de tipo de dano, M2/M3)
- Jupyter Notebook (experimentos exploratórios)

## 10. Instruções para reproduzir experimentos

```bash
pip install -r requirements.txt
# instruções detalhadas de execução serão adicionadas
# conforme os experimentos forem implementados
```

## 11. Vídeo da M1

[https://youtu.be/IxA_tRpA2tM](https://youtu.be/IxA_tRpA2tM)

## 12. Documentação adicional

- [`docs/proposta.md`](docs/proposta.md) — proposta técnica completa
  (problema, contexto, objetivos, pipeline, viabilidade).

## 13. Uso de Inteligência Artificial

Este projeto conta com apoio de ferramentas de Inteligência Artificial
generativa para estruturação da documentação e organização do repositório,
com revisão e validação minhas.
