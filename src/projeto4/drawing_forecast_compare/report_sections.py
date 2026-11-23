"""Seções de previsão do relatório de apresentação."""
import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..io import read_json
from .successful_examples import INK, PREDICTED, make_pages, plot_ink


def number(value, digits=1):
    return f'{value:.{digits}f}'.replace('.', ',')


def learning_figure(gru, output, run_id):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout='constrained')
    for seed in [42, 43, 44]:
        with (gru/'runs'/f'{run_id}_seed{seed}'/'history.csv').open() as stream:
            history = list(csv.DictReader(stream))
        epochs = [int(row['epoch']) for row in history]
        axes[0].plot(epochs, [float(row['train_objective']) for row in history], label=f'Seed {seed}')
        scores = [float(row['val_rollout_objective']) for row in history]
        line, = axes[1].plot(epochs, scores)
        best = min(range(len(scores)), key=scores.__getitem__)
        axes[1].scatter(epochs[best], scores[best], marker='*', color=line.get_color(), s=85)
    axes[0].set(title='Treino', ylabel='Perda de treino')
    axes[1].set(title='Validação', ylabel='Erro usado na seleção')
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel('Época')
        ax.grid(alpha=.17)
    path = output/'figures'/'forecast_gru_learning.pdf'
    for target in [path, path.with_suffix('.png')]:
        fig.savefig(target, dpi=190)
    plt.close(fig)
    return path


def compact_examples(output, cases):
    fig, axes = plt.subplots(len(cases), 2, figsize=(7.2, 7.5), layout='constrained')
    for row, case in enumerate(cases):
        cut, horizon = case['cut'], case['k']
        points = np.asarray(case['points'])
        past = points[:cut]
        predicted = np.vstack([past[-1], case['forecast']['gru']])
        geometry = np.vstack([past[:, :2], predicted[:, :2]])
        low, high = geometry.min(0), geometry.max(0)
        center = (low+high)/2
        radius = max(*(high-low), 1)*.57
        for column, ax in enumerate(axes[row]):
            plot_ink(ax, past, INK, lw=1.4)
            if column == 1:
                plot_ink(ax, predicted, PREDICTED, lw=2.1)
            ax.scatter(*past[-1, :2], s=9, color=INK, zorder=3)
            ax.set(xlim=(center[0]-radius, center[0]+radius),
                   ylim=(center[1]+radius, center[1]-radius), aspect='equal')
            ax.axis('off')
        category = case['category']
        axes[row, 0].set_title(f'{category} · observado', fontsize=9)
        axes[row, 1].set_title(f'{category} · GRU · k={horizon}', fontsize=9)
    handles = [Line2D([], [], color=color, lw=2, label=label) for color, label in
               [(INK, 'Passado recebido'), (PREDICTED, 'Previsão da GRU')]]
    fig.legend(handles=handles, loc='outside lower center', ncol=2, fontsize=8)
    path = output/'figures'/'forecast_apresentacao_casos.pdf'
    for target in [path, path.with_suffix('.png')]:
        fig.savefig(target, dpi=190)
    plt.close(fig)
    return path


def build_sections(source, output):
    gru = source/'models/gru'
    selection = read_json(gru/'selection.json')
    chosen = selection['best']['spec']
    metrics = read_json(gru/'test_metrics.json')['horizons']
    cases = read_json(source/'successful_examples/selected_cases.json')['cases']
    other_cases = compact_examples(output, cases[:-1])
    fish_page = make_pages(source, output)[-1]
    fish_page['title'] = 'Exemplo em detalhe · peixe, k=30'
    fish_page['caption'] = ('A linha escura mostra os 16 eventos observados. A linha verde-azulada mostra a continuação da GRU para os 30 eventos seguintes. As curvas detalham as coordenadas X e Y.')
    rows = [[k, number(metrics[str(k)]['ade_pixels']), number(metrics[str(k)]['fde_pixels'])] for k in [5, 15, 30]]
    learning = learning_figure(gru, output, chosen['id'])
    return [
        {'macro_title': 'Previsão dos próximos traços (t+k)',
         'title': 'O que a GRU prevê',
         'text': 'Entregamos à GRU até 64 eventos observados e a categoria do desenho. Com t=32 e k=15, por exemplo, ela recebe os primeiros 32 movimentos e devolve os eventos 33 a 47. Cada previsão inclui a posição, o estado da caneta e o intervalo desde o ponto anterior. O horizonte k pode ser de 5, 15 ou 30 eventos.\n\n'
                 'Cada ponto observado é representado por seis valores. São os deslocamentos X/Y, o indicador de fim de traço (pen-up), o intervalo de tempo comprimido com logaritmo e a posição X/Y relativa ao primeiro ponto. O pen-up marca quando a pessoa levanta a caneta ao terminar um trecho. A GRU resume a sequência em um estado. O decoder usa esse estado para escolher e ajustar uma entre 64 trajetórias-base da categoria, aprendidas no treino.\n\n'
                 'Testamos 128 e 192 unidades ocultas e taxas de aprendizagem de 0,001 e 0,0005. Treinamos com AdamW por 20 épocas. Os dois melhores candidatos da primeira inicialização foram repetidos com outra inicialização. A escolha usou o erro de posição e a capacidade de seguir até o fim do horizonte na validação, além do acerto de pen-up. '
                 f'A validação selecionou {chosen["hidden"]} unidades ocultas e taxa {number(chosen["lr"],3)}. Os resultados do teste usam a média de três inicializações.'},
        {'title': 'Aprendizagem e erro de previsão',
         'start_on_new_page': True,
         'text': 'Na tabela, o ADE e o FDE aumentam conforme a GRU prevê mais eventos. Os valores são médias de três inicializações e incluem todas as janelas elegíveis do teste para cada horizonte.',
         'tables': [('Resultados da GRU no teste', ['Horizonte k', 'ADE (pixels)', 'FDE (pixels)'], rows)],
         'figures': [(learning, 'As estrelas marcam as épocas escolhidas na validação para cada inicialização.')],
         'pages': [
             {'title': 'Cinco continuações dos exemplos de validação',
              'text': 'Os exemplos vêm da validação. Selecionamos trechos com baixo erro de posição e uma continuação visualmente coerente.',
              'figure': other_cases,
              'caption': 'À esquerda, a entrada. À direita, essa mesma entrada com os pontos previstos pela GRU em verde-azulado.'},
             fish_page]},
    ]
