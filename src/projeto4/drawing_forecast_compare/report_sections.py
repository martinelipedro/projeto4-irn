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
    winner = selection['best']
    chosen = winner['spec']
    finalist_scores = {
        spec['id']: min(row['score'] for row in selection['finalist_scores'] if row['spec'] == spec)
        for spec in selection['finalists']
    }
    search_rows = []
    for row in selection['screening']:
        spec = row['spec']
        mean = finalist_scores.get(spec['id'])
        search_rows.append([spec['hidden'], number(spec['lr'], 4), number(row['score'], 4),
                            number(mean, 4) if mean is not None else '—'])
    metrics = read_json(gru/'test_metrics.json')['horizons']
    cases = read_json(source/'successful_examples/selected_cases.json')['cases']
    other_cases = compact_examples(output, cases[:-1])
    fish_page = make_pages(source, output)[-1]
    fish_page['title'] = 'Exemplo em detalhe · peixe, k=30'
    fish_page['caption'] = ('A linha escura mostra os 16 eventos observados. A linha verde-azulada mostra a continuação da GRU para os 30 eventos seguintes. As curvas detalham as coordenadas X e Y.')
    rows = [[k, number(metrics[str(k)]['ade_pixels']), number(metrics[str(k)]['fde_pixels'])] for k in [5, 15, 30]]
    learning = learning_figure(gru, output, chosen['id'])
    checkpoint_epochs = [read_json(gru/'runs'/f'{chosen["id"]}_seed{seed}'/'summary.json')['best_epoch']
                         for seed in (42, 43, 44)]
    checkpoints = ', '.join(map(str, checkpoint_epochs))
    return [
        {'macro_title': 'Previsão dos próximos traços (t+k)',
         'title': 'Como continuar um desenho em andamento?',
         'text': 'Imagine que alguém começou o contorno de um peixe. Conhecemos a categoria e vemos por onde a caneta passou. Queremos estimar por onde ela vai passar a seguir, incluindo onde levanta e quanto tempo leva entre os pontos. Por exemplo, com t=32 e k=15, a GRU recebe os eventos 1 a 32 e prevê os eventos 33 a 47. Testamos horizontes de 5, 15 e 30 eventos.\n\n'
                 'A entrada tem até 64 eventos observados. Cada um contém deslocamentos X/Y, fim de traço (pen-up), intervalo comprimido com logaritmo e posição X/Y relativa ao primeiro ponto. A GRU lê esses movimentos junto com a categoria fornecida. Seu estado final orienta o decoder, que escolhe e ajusta uma entre 64 trajetórias-base da categoria, aprendidas no treino.'},
        {'title':'Ajuste dos hiperparâmetros da previsão', 'level':2,
         'text':'Cruzamos duas larguras da GRU com duas taxas iniciais de aprendizagem. Cada treino usou AdamW por 20 épocas. Avaliamos os quatro candidatos com a inicialização 42 e repetimos os dois melhores com a 43. O escore de validação considera distância em pixels, encerramentos antes do horizonte e erros de pen-up. Para cada finalista, também testamos limiares de pen-up de 0,2, 0,35 e 0,5 e escolhemos o menor escore médio.\n\n'
                f'A configuração selecionada tem {chosen["hidden"]} unidades ocultas, taxa de {number(chosen["lr"],3)} e limiar de {number(winner["pen_threshold"],1)}. A inicialização 44 entra na média final do teste, depois da seleção.',
         'tables':[('Busca da GRU na validação. Menor escore é melhor. A média é calculada para os finalistas.',
                    ['Unidades','Taxa inicial','Escore 42','Média 42 e 43'],search_rows)]},
        {'title': 'Arquitetura final e épocas da previsão', 'level': 2,
         'text': f'Os 6 atributos de cada evento e os 16 valores da categoria entram em uma camada GRU com {chosen["hidden"]} unidades. Uma camada densa gera {chosen["n_modes"]} escores para escolher a trajetória-base. O estado final da GRU e o vetor de 24 valores do modo escolhido alimentam o decodificador denso 216 → 128 (tanh) → 180. As 180 saídas são 30 grupos de 6 ajustes para posição, caneta e tempo.\n\n'
                 f'Foram {chosen["epochs"]} épocas por inicialização. Os checkpoints escolhidos na validação ficaram nas épocas {checkpoints}, na ordem das seeds 42, 43 e 44.'},
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
