# Projeto 4 — O próximo traço

**Previsão de séries temporais de desenhos com GRU, comparada a um MLP DeepSets.**

[Abrir laboratório interativo](https://martinelipedro.github.io/projeto4-irn/) · [Relatório completo em PDF](relatorio.pdf) · [Relatório em Markdown](results/drawing_forecast_compare_v1/report/relatorio.md)

O modelo recebe os eventos já observados de um desenho e a categoria fornecida pelo usuário. Prevê as posições, os estados da caneta e os intervalos dos próximos **5, 15 ou 30 eventos**. O horizonte é contado em pontos, não em segundos. Este trabalho avalia exclusivamente previsão, sem reconhecedor de categorias.

## Resultados

Erro médio de trajetória (ADE), em pixels, no mesmo teste e com categoria fornecida. Médias das três inicializações 42/43/44; a demonstração executa a seed 42.

| Horizonte | GRU | MLP DeepSets |
|---|---:|---:|
| 5 eventos | 44.67 | 49.96 |
| 15 eventos | 94.83 | 110.47 |
| 30 eventos | 138.57 | 161.19 |

As redes recebem os mesmos dados, atributos, condição, âncoras, decoder, perdas, augmentações e orçamento. A GRU atualiza um estado recorrente; o comparador aplica uma MLP por evento e agrega por média, sem recorrência. O decoder comum usa a última posição e direção; o MLP também recebe deslocamentos, posições e tempo locais. A comparação não demonstra superioridade sobre qualquer rede possível.

O teste já foi observado em estudos anteriores. A seleção desta comparação usa somente a validação; não é uma confirmação externa inédita. Os exemplos demonstrativos são de validação e incluem casos em que cada rede ganha, com regras registradas.

## O que o relatório investiga

- Busca de hiperparâmetros no início, checkpoints e curvas de aprendizagem.
- Entrada de seis atributos por evento, embedding da categoria e saídas previstas.
- Horizontes, contextos crescentes e janelas com origem e alvos fixos.
- Erros por categoria, cruzamentos e levantamentos da caneta.
- Perturbações de ordem, escala, translação, tempo e estado da caneta.
- Diferenças pareadas, intervalos por desenho, caudas de erro e exemplos 2D/X/Y.

Translação é removida pela representação relativa compartilhada. Deslocamentos não removem a magnitude, portanto **escala não é invariância garantida**. A GRU é sensível à ordem, mas não separa perfeitamente componentes nem é imune a rabiscos. As hipóteses, evidências e limites estão no relatório.

## Executar o site

```bash
python3 -m http.server 8765 --directory site
```

Abra http://127.0.0.1:8765. Os dois modelos rodam localmente em JavaScript num Web Worker. Não há backend Python nem envio do desenho a um servidor. Escolha um exemplo real ou desenhe, indique a categoria e compare previsões, curvas e perturbações. A reprodução lenta não altera timestamps.

## Conferir resultados e gerar o relatório

```bash
uv sync --group dev
uv run pytest -q
PYTHONPATH=src uv run python scripts/audit_results.py
node site/tests/check.mjs
PYTHONPATH=src uv run python -m projeto4.drawing_forecast_compare.diagnostics
PYTHONPATH=src uv run python -m projeto4.drawing_forecast_compare.report
```

Os diagnósticos e o relatório usam CPU. A geração do PDF requer `pdflatex`. O ambiente e a GPU de referência estão registrados nos artefatos. As figuras existem em PNG e PDF. O site usa caminhos relativos para funcionar sob `/projeto4-irn/`.

## Reproduzir os treinos do zero

O dataset preparado de 42 mil desenhos acompanha o repositório (36 mil de treino, 3 mil de validação e 3 mil de teste). Não é necessário baixar os recortes brutos para treinar. O manifesto por IDs, hashes e URLs permite reconstruí-lo; veja `data_sources/raw_manifest.json` e `data.py`.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src uv run python -m projeto4.drawing_forecast_compare all --config configs/reproduce.json
```

Esse comando exige PyTorch com CUDA e uma GPU acessível. Usa um novo diretório de resultados e não reutiliza checkpoints neurais anteriores. Os dados e o banco comum continuam restritos ao treino. O estudo de referência utilizou uma RTX 4050 de 6 GB, float32, determinismo habilitado e sem TF32. Reprodutibilidade numérica entre hardware/versões diferentes não implica igualdade bit a bit.

A busca faz quatro candidatos por família na seed 42, repete dois finalistas com 43 e escolhe pela média da validação. A vencedora é repetida com 44, fora da decisão. O orçamento é 20 épocas. Na execução registrada, quatro treinos GRU completos e compatíveis já existentes foram reutilizados com hashes; eles próprios haviam sido treinados do zero. Essa reutilização não inicializa outro modelo com pesos treinados.

## Organização

| Caminho | Conteúdo |
|---|---|
| `src/projeto4/drawing_forecast_compare/` | Dados, encoders, treino, avaliação, exportação, diagnósticos e relatório |
| `configs/` | Configuração congelada e configuração de reprodução do zero |
| `results/drawing_forecast_compare_v1/` | Dados preparados, proveniência, históricos, checkpoints, previsões e métricas |
| `site/` | Aplicativo estático, pesos exportados e testes numéricos/UI |
| `tests/` | Invariantes da comparação, representação e inferência |

O workflow publica somente `site/` no GitHub Pages após conferir os artefatos. O treinamento não ocorre no deploy.

Dados: [Google Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset), **CC BY 4.0**. Veja [DATA_ATTRIBUTION.md](DATA_ATTRIBUTION.md). O relatório cita GRU, Deep Sets e a inspiração em modos de trajetória; a implementação não reproduz Sketch-RNN ou MultiPath. Código e redação elaborados com auxílio de IA e revisados por verificações numéricas e visuais.
