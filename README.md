# Projeto 4 — Traço a traço

**Reconhecimento progressivo de desenhos: GRU temporal versus MLP convencional sobre imagem.**

[Laboratório interativo](https://martinelipedro.github.io/projeto4-irn/) · [Relatório único em PDF](relatorio.pdf) · [Experimento secundário de previsão](https://martinelipedro.github.io/projeto4-irn/forecast/)

A rede recebe somente os traços já observados e estima a categoria do desenho entre 12 possibilidades. O rótulo verdadeiro é o alvo supervisionado, **não uma entrada**. O site permite desenhar livremente ou reproduzir exemplos reais lentamente, acompanhando as hipóteses top-3 e as curvas de probabilidade das duas redes.

## O que comparamos

| Rede | Apresentação do prefixo | Arquitetura |
|---|---|---|
| GRU | Eventos ordenados: Δx, Δy, pen-up, log(1+Δt/0,1), x e y | GRU → densa → 12 classes |
| MLP convencional | Imagem binária 32×32 achatada em 1.024 pixels | Densa → ReLU → densa → ReLU → 12 classes |

A MLP principal não usa Deep Sets, convolução, agregação de eventos, relógio ou canal de recência. A imagem preserva a geometria observada e remove a ordem em que os traços foram executados. Uma MLP sobre pontos concatenados ainda teria a ordem representada pelas posições do vetor e responderia a outra pergunta.

Ambas as redes podem reconhecer antes do fim. A GRU dispõe de movimentos, intervalos e levantamentos da caneta; isso não garante melhor resultado em todo prefixo nem imunidade a rabiscos. As duas apresentações usam a caixa do próprio prefixo para normalizar posição e escala. Como essa normalização muda com o prefixo, reexecutamos a GRU sobre todos os eventos disponíveis a cada atualização.

Arquitetura **e representação** mudam juntas: a comparação avalia dois sistemas completos, sem atribuir toda diferença exclusivamente à recorrência. O raster de baixa resolução também é uma limitação do comparador.

## Resultados de reconhecimento

<!-- RESULTS_START -->
Acurácia e F1 macro no mesmo teste de 3.000 desenhos; médias das três inicializações 42/43/44, sem ensemble.

| Observado | Acurácia GRU | Acurácia MLP | F1 GRU | F1 MLP |
|---|---:|---:|---:|---:|
| 25% | 63,62% | 51,47% | 0,633 | 0,505 |
| 50% | 82,87% | 69,81% | 0,828 | 0,697 |
| 75% | 90,50% | 81,10% | 0,905 | 0,810 |
| 100% | 93,03% | 83,71% | 0,931 | 0,838 |
<!-- RESULTS_END -->

A busca e o checkpoint usam somente a validação desta comparação. O teste já havia sido observado em estudos anteriores, portanto não é confirmação externa inédita. Os exemplos demonstrativos pertencem exclusivamente à validação, com regras de seleção registradas, incluindo casos favoráveis a cada rede.

## Protocolo

- 36 mil desenhos de treino, 3 mil de validação e 3 mil de teste, com IDs e geometrias separados por desenho; 12 categorias e somente exemplos reconhecidos pelo jogo.
- Mesmos prefixos e alvos para as duas redes: 25%, 50%, 75% e 100% dos eventos. A fração final é usada para escolher o corte de avaliação; não é atributo de entrada.
- Busca por largura e taxa de aprendizagem: GRU H128/192, MLP H128/256, taxas 0,003/0,001. AdamW, dropout 0,1 e 32 épocas por execução.
- Triagem de quatro candidatos por família na seed 42; dois finalistas repetidos com 43; escolha pela média da entropia cruzada de validação. Seed 44 apenas replica a configuração escolhida. As duas famílias são congeladas antes do teste.
- Todos os classificadores são inicializados do zero. Um piloto de 16 épocas foi interrompido por subajuste observado na validação; seus pesos não inicializam os modelos principais.

O relatório reúne curvas de aprendizagem e reconhecimento, acurácia/F1/top-3, qualidade probabilística, confusões por categoria, intervalos pareados e controles de ordem, pausas, cruzamentos e normalização. [Detalhamento do protocolo](docs/RECONHECIMENTO.md).

## Executar e verificar

```bash
uv sync --group dev
uv run pytest -q
PYTHONPATH=src uv run python scripts/audit_recognition.py
node site/tests/check.mjs
node site/forecast/tests/check.mjs
python3 -m http.server 8765 --directory site
```

Abra http://127.0.0.1:8765. Os dois classificadores rodam num Web Worker local, sem backend e sem enviar desenhos a um servidor. O teste Node compara a inferência JavaScript a referências Python dos checkpoints efetivos.

Para reconstruir os diagnósticos, os exemplos do site e o PDF em CPU:

```bash
PYTHONPATH=src uv run python -m projeto4.drawing_recognition_compare.export
PYTHONPATH=src uv run python -m projeto4.drawing_recognition_compare.report
```

O PDF requer `pdflatex`. A versão gerada fica em `results/drawing_recognition_compare_v2/report/relatorio.pdf`; as cópias de entrega estão na raiz e em `site/relatorio.pdf`.

## Reproduzir o treinamento do zero

O dataset preparado de 42 mil desenhos acompanha o repositório em `results/drawing_forecast_compare_v1/data/`. O reconhecimento reutiliza essas partições, sem checkpoints. Os caches de sequências normalizadas e imagens são reconstruídos automaticamente e não são versionados.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src uv run python -m projeto4.drawing_recognition_compare.study all --config configs/recognition_reproduce.json
```

O comando usa um novo diretório de resultados e requer PyTorch com CUDA. O estudo de referência usou uma RTX 4050 de 6 GB, float32, determinismo habilitado e TF32 desligado; as versões estão em `results/drawing_recognition_compare_v2/environment.json`. Igualdade bit a bit não é garantida entre hardware e versões diferentes.

## Estudo secundário: prever os próximos pontos

A previsão de trajetória permanece disponível em `site/forecast/`, separada do reconhecimento. Nela a categoria é fornecida e o horizonte é de 5, 15 ou 30 eventos futuros. Seu comparador é **Deep Sets**, identificado como tal: não é a MLP convencional do estudo principal. ADE/FDE desse experimento não são métricas de reconhecimento. Seus dados, treinos, fontes e resultados permanecem em `results/drawing_forecast_compare_v1/`; o relatório único resume a tarefa secundária.

## Organização

| Caminho | Conteúdo |
|---|---|
| `src/projeto4/drawing_recognition_compare/` | Apresentação dos dados, modelos, busca, avaliação, exportação e relatório principal |
| `configs/recognition*.json` | Configuração de referência e reprodução do zero |
| `results/drawing_recognition_compare_v2/` | Históricos, checkpoints, seleções, previsões, métricas, diagnósticos e figuras |
| `site/` | Reconhecimento interativo e modelos reais no navegador |
| `site/forecast/` | Demonstração secundária de previsão, com GRU e Deep Sets |
| `tests/` e `scripts/audit_recognition.py` | Invariantes e auditoria científica dos resultados |

O workflow confere as duas demonstrações antes de publicar `site/` no GitHub Pages. O treinamento não ocorre no deploy.

Dados: [Google Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset), **CC BY 4.0**; [atribuição](DATA_ATTRIBUTION.md). Código e redação elaborados com auxílio de IA, acompanhados de artefatos, testes e verificações visuais.
