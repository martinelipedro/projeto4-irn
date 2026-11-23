# Reconhecimento progressivo: protocolo e motivação

A comparação de reconhecimento é entre uma **GRU temporal** e uma **MLP convencional sobre imagem**, ambas classificadoras de 12 categorias. O relatório tem duas macroseções: previsão dos traços e reconhecimento da categoria. As métricas de previsão não são métricas de reconhecimento.

## Por que usar pixels na MLP

Uma MLP sobre pontos concatenados também não remove a ordem: cada posição do vetor representa um instante. Para realmente remover a ordem de execução da entrada, renderizamos os traços numa imagem binária e achatamos seus pixels.

## Experimento principal

- **Entrada GRU:** eventos observados `(dx,dy,pen_up,log1p(dt/0.1),x,y)`, na ordem de execução e com comprimento real. Padding à direita não altera a saída correspondente ao último evento observado.
- **Entrada MLP:** os mesmos traços observados, renderizados numa imagem binária 32×32 e achatados em 1.024 valores. Duas camadas densas ocultas, ReLU e dropout; nenhuma convolução, recorrência, pooling de eventos, relógio ou canal de recência.
- **Saída de ambas:** 12 probabilidades de classe. O rótulo verdadeiro é somente o alvo supervisionado. Nenhuma categoria é fornecida como condição à rede.
- **Normalização:** centrar a caixa do prefixo e dividir pela maior extensão observada, nas duas apresentações. Não usar caixa, duração ou comprimento final do desenho para construir uma entrada parcial.
- **Partições:** mesmos 36 mil/3 mil/3 mil desenhos inteiros já empacotados, com IDs fixos e sem sobreposição entre partições. O teste já foi observado em outros estudos; não é confirmação externa inédita.
- **Prefixos:** 25%, 50%, 75%, 100% dos eventos para avaliação. A fração final é uma escolha do protocolo de avaliação; não é atributo de entrada. Treino escolhe uma dessas vistas por desenho a cada época, igualmente para ambas as famílias.
- **Busca:** GRU H128/192, MLP H128/256, taxas 0,003/0,001, AdamW, dropout 0,1, 32 épocas. Quatro candidatos por família na seed 42; dois finalistas repetidos com 43; decisão pela média da entropia cruzada na validação, calculada nos quatro prefixos. Seed 44 replica a vencedora. Congelar ambas as decisões antes do teste.
- **Piloto:** uma triagem inicial de 16 épocas foi interrompida após evidência de subajuste na validação. Nenhum teste foi consultado nessa decisão. Aumentamos o orçamento para ambas as famílias e inicializamos novamente todos os pesos. Os históricos do piloto não compõem os resultados principais.

## O que os resultados permitem concluir

A comparação mede dois sistemas completos: arquitetura e apresentação dos dados mudam juntas. Uma vantagem não pode ser atribuída exclusivamente à recorrência. Controlar somente a arquitetura exigiria também uma MLP sobre a mesma sequência concatenada; esse controle ainda preservaria informação de ordem e responderia a outra pergunta.

Trocar a ordem de traços inteiros conserva a imagem binária e a previsão da MLP. Pode alterar a GRU; sensibilidade não significa melhor acurácia. As duas ordens usam intervalos uniformes e fechamento do último trecho parcial (pen-up final igual a 1 em ambas), conservando também a quantidade de flags. Essa referência composta não é uma ablação isolada de tempo. É uma perturbação fora da distribuição, sem novo treinamento.

Ambas as redes podem reconhecer antes do fim. A hipótese é que façam isso em momentos e com erros diferentes. Cruzamentos não garantem vantagem da GRU. Pausas e pen-up são informação disponível, não garantia de interpretação perfeita. Invariância a translação e escala vem da normalização explícita compartilhada, não de uma propriedade automática de uma RNN.

Referências: [Google Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset), [GRU](https://arxiv.org/abs/1406.1078).
