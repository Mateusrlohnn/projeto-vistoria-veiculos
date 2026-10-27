# Roteiro — vídeo da M2 (~10 min)

Formato sugerido: gravação de tela (OBS ou Xbox Game Bar `Win+Alt+R`) com
sua voz e, se possível, sua webcam num canto — o enunciado pede
"participação identificável". Fale com suas palavras; as falas abaixo são
um guia, não para ler.

## Antes de gravar (checklist)

- [ ] Terminal aberto na pasta do projeto, fonte grande (Ctrl + roda do mouse).
- [ ] Rodar uma vez antes, para não esperar no vídeo:
      `python -m pytest tests` e o comando do exemplo mínimo (bloco 3).
- [ ] Abas abertas no navegador:
      1. devolutiva da M1 (`grupo_16.md`)
      2. página do dataset na Roboflow (aba *Dataset*, versão v10)
      3. repositório no GitHub, branch `m2`
- [ ] No VS Code, deixar abertos: `configs/B_ajustado.yaml`,
      `src/pipeline.py` (função `regiao_aprovada`), `docs/relatorio_m2.md`,
      `results/teste/analise.md`.
- [ ] Painéis abertos num visualizador de imagens:
      `results/exemplo/figuras/IMG_5498.png`,
      `results/casos/A_baseline_m1/figuras/IMG_5498.png`,
      `results/casos/B_ajustado/figuras/IMG_5581.png`,
      `results/casos/B_ajustado/figuras/IMG_5522.png`.

---

## 1. Abertura — 0:00 a 0:40

**Tela:** README no GitHub (branch `m2`).

> "Olá, sou o Mateus Rachadel Lohn, e este é o vídeo da M2 do projeto de
> vistoria automática de veículos. Faço o trabalho individualmente. Vou
> mostrar o que mudou desde a M1, os dados, o pipeline rodando, o
> experimento que compara o método da M1 com o da M2, onde ele acerta e
> onde falha, e o plano para a M3."

## 2. Evolução desde a M1 e resposta à devolutiva — 0:40 a 2:10

**Tela:** devolutiva → depois tabela da seção 2.1 do relatório.

> "Na M1 o problema era amplo demais: qualquer avaria, em qualquer foto, e
> com imagens do Google sem licença. O professor apontou cinco correções."

Passe pelos 5 itens, apontando a tabela:

1. **Licença:** "tirei as imagens do Google do repositório e troquei por um
   dataset CC BY 4.0."
2. **Escopo:** "reduzi para um tipo de dano — arranhão — em fotos próximas
   de um painel."
3. **Referência:** "o dataset já tem polígonos marcando cada arranhão; é a
   minha referência."
4. **Comparação e negativos difíceis:** "agora meço contra a referência, e
   criei regras específicas para reflexo e para junção entre portas."
5. **Antes/depois:** "esse eu não consegui testar com dados na M2 — vou
   explicar no final por que os resultados reforçam essa ideia e como
   entra na M3."

> "Todas as decisões do dia estão registradas no diário da M2."
> (mostre `docs/diario_m2.md` por 2 segundos)

## 3. Dados — 2:10 a 3:20

**Tela:** página da Roboflow → `data/catalogo.csv`.

> "O dataset é o 'Car scratch', publicado na Roboflow com licença CC BY
> 4.0. Escolhi a versão 10 porque é a única colorida e sem
> augmentation — a versão 14 tinha cada foto repetida três vezes,
> girada e com ruído, o que estragaria a avaliação."

> "Analisando os nomes dos arquivos, vi que o export mistura origens:
> imagens com padrão do CarDD, do Adobe Stock, de WhatsApp. Para não
> repetir o problema de licença da M1, fiquei só com as 67 fotos da mesma
> câmera, numeradas IMG_5481 a 5667."

> "Como fotos com números próximos são do mesmo carro, dividi pela ordem
> de captura: as 13 primeiras para ajustar parâmetros, as 54 seguintes
> só para o teste final. O dataset não vai pro repositório; o README
> explica como baixar e o script `preparar_dados.py` monta tudo."

## 4. Pipeline rodando — 3:20 a 5:40

**Tela:** terminal. Rode ao vivo:

```bash
python -m src.executar --config configs/B_ajustado.yaml --dados data/car_scratch --conjunto teste --ids IMG_5498 --figuras 1 --saida results/exemplo
```

> "Esse comando processa uma foto e já compara com a referência: 15
> arranhões anotados, 7 detectados, 6 falsos positivos."

**Tela:** abra `results/exemplo/figuras/IMG_5498.png` e explique os 8
quadros, apontando cada um:

1. **Entrada.**
2. **Realce top-hat + black-hat:** "realça só o que é mais fino que o
   elemento estruturante de 9 pixels. Black-hat pega traços escuros,
   top-hat traços claros. Por isso troquei o Canny da M1: o Canny marca
   qualquer borda, inclusive pneu e porta."
3. **Segmentação:** "mantenho os 2% de pixels com maior resposta. Testei
   Otsu, mas como o arranhão ocupa 0,2% da imagem, o Otsu deixava passar
   a textura toda."
4. **Máscara de reflexo:** "no HSV, reflexo é muito claro e pouco
   saturado. Aqui ela cobre a borda iluminada do vão da porta."
5. **Morfologia:** "abertura tira pontinhos; fechamento tenta unir pedaços
   do mesmo risco."
6. **Componentes conexos:** "cada cor é uma região candidata."
7. **Regiões aprovadas pela regra.**
8. **Comparação:** "verde é a referência, vermelho a predição, amarelo
   acerto."

**Tela:** `descritores.csv` (ou seção 4 do notebook) e depois
`regiao_aprovada` em `src/pipeline.py`.

> "Para cada região calculo descritores: área, alongamento, solidez,
> contraste, textura ao redor, saturação. A regra aprova como arranhão se
> for alongada, de tamanho certo, fora de reflexo, sobre pintura lisa e se
> não for uma linha que atravessa a foto — que é o vão entre portas."

> "Os limiares foram escolhidos por uma busca em grade de 1296
> combinações, usando só as 13 imagens de ajuste."

## 5. Experimento A × B — 5:40 a 7:30

**Tela:** `results/teste/analise.md` (ou a tabela do README).

> "A pergunta do experimento: essas regras novas reduzem os falsos
> positivos sem perder arranhões? A é exatamente o método da M1; B é o da
> M2. Mesmas 54 imagens de teste, mesma referência, mesmas métricas —
> rodado uma única vez."

Leia a tabela:

> "B reduziu os falsos positivos de 8,5 para 1,5 por imagem e dobrou a
> precisão. Mas o recall caiu de 29% para 3%. Ou seja: a resposta é sim
> para os falsos positivos e não para 'sem perder arranhões'."

**Tela:** tabela "regiões por motivo".

> "Um resultado bom: as regras de reflexo e de junção removeram 30
> regiões e nenhuma era arranhão real. O problema está antes: a
> segmentação quebra cada arranhão em pedacinhos, e 3700 pedaços caem no
> filtro de área. Descobri também que o fechamento com elipse não
> consegue reconectar traços de 1 pixel — percebi isso ao escrever o
> teste automático. Não mudei depois de ver o teste, pra não viciar o
> resultado; fica como tarefa da M3."

> "E o recall da M1 é em parte ilusório: a dilatação gera manchas
> grossas que cobrem arranhões por acaso — o IoU dela é quase zero."

## 6. Sucesso e falhas — 7:30 a 8:50

**Tela:** painéis, um de cada vez.

- **IMG_5498, A × B:** "na mesma foto, a M1 marca o vão da porta e a caixa
  de roda e não acha nenhum arranhão. A M2 acha 7 dos 15."
- **IMG_5581:** "falha de segmentação: para-choque curvo, muito reflexo e
  asfalto. Nenhum candidato cai sobre os arranhões."
- **IMG_5522:** "falha da regra: a pintura reflete céu e galhos. Essas
  bordas são finas, alongadas e sobre fundo liso — idênticas a um
  arranhão para a regra. Como não são brancas, a máscara de reflexo não
  pega."

## 7. Limitações, antes/depois e M3 — 8:50 a 9:45

**Tela:** seção 7 e 8 do relatório.

> "Limitações: ajustei em só 13 imagens e os parâmetros não generalizaram
> bem; metade das anotações são marcas de poucos pixels; os carros são
> quase todos claros."

> "Sobre o antes/depois que o professor sugeriu: os resultados apoiam a
> ideia. O reflexo do céu aparece igual nas duas fotos e some na
> diferença; o arranhão novo não. Na M3 vou fazer esse experimento com
> fotos minhas."

> "Plano da M3: corrigir a fragmentação, validar com mais imagens,
> treinar um classificador sobre os descritores e fazer o antes/depois."

## 8. Fechamento — 9:45 a 10:10

**Tela:** GitHub, branch `m2` / testes passando no terminal.

> "Tudo está na branch m2, com instruções de instalação, testes e um
> notebook que reproduz o experimento. Usei o Claude Code como apoio no
> código e na documentação — está declarado no relatório — e verifiquei
> com testes e execução real. Obrigado."

---

## Depois de gravar

1. YouTube → *Criar* → *Enviar vídeo* → visibilidade **Não listado**.
2. Abrir o link numa janela anônima para confirmar que abre sem login.
3. Me passar o link: coloco no README, faço o commit, o merge na `main` e
   a tag `m2` (com a sua autorização).
