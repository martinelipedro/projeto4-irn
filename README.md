# Projeto 4 — Séries temporais de desenhos

[Relatório em PDF](relatorio.pdf) · [Reconhecimento no navegador](https://martinelipedro.github.io/projeto4-irn/) · [Previsão no navegador](https://martinelipedro.github.io/projeto4-irn/forecast/)

Usamos 42 mil desenhos de 12 categorias do Quick, Draw!, divididos em treino, validação e teste por desenho.

| Tarefa | Entrada | Saída |
| --- | --- | --- |
| Previsão com GRU | Sequência observada e categoria fornecida | Próximos 5, 15 ou 30 eventos da caneta |
| Reconhecimento com GRU | Sequência observada de movimentos | Probabilidades das 12 categorias |
| Reconhecimento com MLP | Imagem binária do trecho observado, com 32 × 32 pixels | Probabilidades das 12 categorias |

## Rodar as demonstrações

Na raiz do repositório, execute com Python 3.

```bash
python3 -m http.server 8765 --directory site
```

Abra [reconhecimento](http://127.0.0.1:8765/) ou [previsão](http://127.0.0.1:8765/forecast/). Os pesos já estão em `site/` e a inferência roda no navegador. Você pode usar os exemplos incluídos ou desenhar.

## Repetir o treinamento e a avaliação

Use Python 3.11 ou superior, `uv` e PyTorch com acesso a uma GPU NVIDIA via CUDA. O recorte preparado e as partições usadas no relatório acompanham o repositório.

```bash
uv sync
CUBLAS_WORKSPACE_CONFIG=:4096:8 uv run python -m projeto4.drawing_forecast_compare all
uv run python -m projeto4.drawing_recognition_compare.study all
```

Cada comando prepara a entrada, executa a busca de hiperparâmetros na validação e avalia as configurações escolhidas no teste. A previsão usa o banco de trajetórias extraído do treino e inicializa os pesos da GRU do zero. O reconhecimento inicializa GRU e MLP do zero. O PDF documenta os critérios e os resultados. As execuções novas gravam seus resultados em `results/drawing_forecast_reproduction_v1/` e `results/drawing_recognition_reproduction_v1/`.

## Conferir a execução

```bash
uv run pytest -q
node site/tests/check.mjs
node site/forecast/tests/check.mjs
```

Os testes JavaScript comparam a inferência no navegador com referências calculadas em Python. As duas demos usam os pesos da seed 42. Os checkpoints escolhidos, históricos e métricas das três seeds ficam em `results/`.

O código dos modelos e do treino fica em `src/projeto4/`, e as configurações em `configs/`. A versão completa, com análises auxiliares e scripts de geração do PDF, está na [tag arquivo-experimentos](https://github.com/martinelipedro/projeto4-irn/tree/arquivo-experimentos).

Dados do Google Quick, Draw! sob CC BY 4.0. [Atribuição e proveniência](DATA_ATTRIBUTION.md)
