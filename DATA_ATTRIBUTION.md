# Dados e atribuição

Os desenhos são um recorte do [Google Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset), disponibilizado sob [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Neste projeto os vetores foram amostrados causalmente, normalizados e organizados em partições por desenho. Os tempos originais dos pontos foram preservados. O arquivo `results/drawing_forecast_compare_v1/data_sources/raw_manifest.json` registra os endereços de origem, os recortes e os hashes. `data/records.json` identifica os desenhos, categorias e partições utilizados; `data/metadata.json` registra as transformações.

Os exemplos do site pertencem à validação e sua escolha é documentada nos diagnósticos. Os resultados agregados utilizam o teste congelado. A categoria é fornecida aos modelos, e este estudo não avalia reconhecimento de categorias.

A implementação de previsão por modos é uma adaptação própria inspirada em [MultiPath](https://proceedings.mlr.press/v100/chai20a.html). O comparador por agregação de eventos se inspira em [Deep Sets](https://arxiv.org/abs/1703.06114). O projeto não é uma reprodução de MultiPath nem do Sketch-RNN.
