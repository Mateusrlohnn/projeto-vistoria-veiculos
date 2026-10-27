# Diário de desenvolvimento — M2

Registro das decisões tomadas durante a M2 (27/10/2026), em ordem. Os
commits citados estão na branch `m2`.

| # | Situação / problema | Decisão | Commit |
|---|---|---|---|
| 1 | Devolutiva da M1: imagens sem licença, escopo amplo, sem referência | Plano da M2 refeito em torno das 5 correções essenciais | — |
| 2 | CarDD exige formulário de licença e espera pelo link | Descartado pelo prazo; busca por dataset CC BY 4.0 com segmentação | — |
| 3 | Primeiro candidato (`scratchseg`) não tinha fotos de carro | Escolhido "Car scratch" (PRISMAUTO), conferido página a página: instance segmentation, CC BY 4.0, close-ups de lataria | — |
| 4 | Pipeline precisava rodar antes dos dados chegarem | Código escrito independente da fonte (COCO ou caixas próprias) e testado com imagem sintética | `a7cb093` |
| 5 | v14 do dataset: 3 cópias aumentadas por foto, faixas pretas | Trocado pela **v10** (sem augmentation, colorida). v9 rejeitada por estar em cinza e equalizada | — |
| 6 | v10 mistura origens (CarDD, Adobe Stock, WhatsApp…) | Mantidas só as 67 `IMG_` de mesma câmera; 79 descartadas | `0356edf` |
| 7 | Fotos consecutivas são do mesmo carro | Split ajuste/teste pela ordem de captura, no salto 5493 → 5498 | `0356edf` |
| 8 | B com parâmetros iniciais pior que A no ajuste (Otsu deixa passar textura do pneu; vão de porta passa como arranhão) | Adicionados limiar por percentil, descritor de textura do entorno e regra de junção | `5fc13ee` |
| 9 | Commit `5fc13ee` incluiu também a remoção de `images/` (estava preparada pelo `git rm` e entrou no primeiro commit seguinte) | Mantido como está e registrado aqui, para não reescrever histórico já publicado | `5fc13ee` |
| 10 | Escolha de parâmetros sem olhar o teste | Busca em grade (1296 combinações) só no ajuste, por F1 | `073435e` |
| 11 | Ao escrever o teste de fechamento: elipse 5 × 5 não reconecta traço de 1 px | Não alterado depois do teste (evita ajuste olhando o resultado); registrado como limitação e tarefa da M3 | `7f2db94` |
| 12 | Resultado no teste: FP/imagem 8,5 → 1,5, mas recall 0,29 → 0,03 | Reportado integralmente, com análise por tipo e por regra; plano da M3 ataca a fragmentação e usa classificador | `3ef34cc` |
