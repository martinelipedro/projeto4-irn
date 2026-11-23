# Dados e atribuição

Os desenhos são um recorte do [Google Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset), disponibilizado sob [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Neste projeto os vetores foram amostrados causalmente, normalizados e organizados em partições por desenho. Os tempos originais dos pontos foram preservados. O arquivo `results/drawing_forecast_compare_v1/data_sources/raw_manifest.json` registra os endereços de origem, os recortes e os hashes. Nessa mesma pasta de experimento, `data/records.json` identifica os desenhos, categorias e partições; `data/metadata.json` registra as transformações.

O experimento de reconhecimento reutiliza as mesmas partições, com proveniência em `results/drawing_recognition_compare_v2/data_manifest.json`. Ele produz duas apresentações do mesmo prefixo: uma sequência normalizada para a GRU e uma imagem binária para a MLP convencional. A categoria é somente o alvo supervisionado e nunca uma entrada desses classificadores. Os exemplos do site pertencem à validação e sua escolha é documentada. Os resultados agregados utilizam o teste fixo, que já foi observado em estudos anteriores.

Na previsão dos próximos traços, a GRU recebe a categoria como condição. O decoder de trajetórias usa modos inspirados em MultiPath. [Artigo de Chai et al.](https://proceedings.mlr.press/v100/chai20a.html)
