"""Primeira macroseção do relatório único: previsão dos traços e bons exemplos."""
import csv

import matplotlib.pyplot as plt

from ..io import read_json
from .successful_examples import make_pages


def number(value,digits=2):return f'{value:.{digits}f}'.replace('.',',')


def build_sections(source,output):
    metadata=read_json(source/'data/metadata.json')
    gru=source/'models/gru'
    selection=read_json(gru/'selection.json')
    metrics=read_json(gru/'test_metrics.json')['horizons']
    chosen=selection['best']
    search=[[row['spec']['hidden'],number(row['spec']['lr'],4),row['checkpoint_epoch'],
             number(row['score'],4)] for row in selection['screening']]
    finals=[]
    for spec in selection['finalists']:
        best=min((r for r in selection['finalist_scores'] if r['spec']==spec),key=lambda r:r['score'])
        finals.append([spec['hidden'],number(spec['lr'],4),number(best['seed_scores'][0],4),
                       number(best['seed_scores'][1],4),number(best['score'],4)])
    results=[[k,row['n_prefixes'],number(row['ade_pixels']),number(row['fde_pixels']),
              number(row['pen_f1'],3),number(row['clock_mae_seconds'],3)]
             for k,row in metrics.items()]
    selected=read_json(source/'successful_examples/selected_cases.json')
    examples=[]
    for row in selected['cases']:
        m=row['metrics']['gru']
        examples.append([row['category'],row['cut'],row['k'],number(m['ade']),number(m['fde'])])
    pages=make_pages(source,output)
    learning, errors = gru_figures(gru, output, chosen['spec']['id'], metrics)
    return [
        {'macro_title':'Previsão dos próximos traços (t+k)',
         'title':'O que significa prever t+k?',
         'text':'A GRU observa uma janela de movimentos até o evento t e prevê onde a caneta estará nos eventos t+1 até t+k. O alvo é uma sequência futura de posições, estados da caneta e intervalos de tempo. Por exemplo, com t=32 e k=15, entregamos somente os primeiros 32 eventos e comparamos a saída com os eventos reais 33 a 47. O modelo não recebe esses 15 eventos durante a inferência.\n\n'
                'A categoria do desenho é fornecida nesta tarefa: ao pedir uma continuação de Peixe, já informamos Peixe à rede. Reconhecer a categoria é uma tarefa diferente, tratada na segunda parte, em que esse rótulo deixa de ser entrada e passa a ser a saída. Os pesos de previsão e de reconhecimento são treinados separadamente, embora os desenhos e as partições de origem sejam os mesmos.\n\n'
                'Os horizontes são k=5,15,30 eventos amostrados, e não segundos. Os instantes são irregulares; o relógio também é estimado. O objetivo é aproximar o movimento humano que realmente aconteceu dentro desse horizonte, não completar qualquer desenho inteiro de uma vez.',
         'equations':[r'(\mathbf u_{t-w+1},\ldots,\mathbf u_t,\ c)\longmapsto(\widehat{\mathbf y}_{t+1},\ldots,\widehat{\mathbf y}_{t+k}),\qquad k\in\{5,15,30\}.'],
         'tables':[('Contrato do experimento de previsão', ['Elemento','Conteúdo','Significado'],[
             ['Passado','Até 64 eventos observados','Só dados disponíveis até t'],
             ['Condição','Uma das 12 categorias','Informada pelo usuário'],
             ['Futuro','5,15 ou 30 eventos','Posição, caneta e intervalo'],
             ['Conferência','Futuro humano registrado','Usado apenas em perda e avaliação']])]},
        {'title':'Como os dados entram e os traços saem',
         'text':'Reutilizamos as partições de 36 mil desenhos de treino, 3 mil de validação e 3 mil de teste. A amostragem espacial preserva limites de traços e timestamps; os desenhos preparados têm entre 12 e 180 eventos. Durante o treinamento, prefixos de 8 a 64 eventos variam conforme a disponibilidade do desenho e do lote. O contexto é causal e começa no início do desenho.\n\n'
                f'Cada evento fornece seis atributos: deslocamentos X/Y divididos pela escala de treino de {number(metadata["delta_scale"])} pixels, pen-up, log(1+Δt/0,1), e posições relativas ao primeiro ponto divididas por 256. A categoria vira um embedding de 16 dimensões, formando 22 atributos por evento. Nesta tarefa não usamos a normalização pela caixa do prefixo adotada no reconhecimento. A representação remove translação, mas não garante invariância de escala.\n\n'
                'A GRU atualiza seu estado a cada evento. O último estado resume a janela e alimenta o decoder, que estima a continuação usando a última posição e a direção recente da caneta.\n\n'
                'O decoder escolhe uma entre 64 trajetórias-base por categoria, aprendidas somente no treino, e estima correções de posição, probabilidades de caneta/fim e intervalos positivos. Na inferência determinística, o modo é escolhido pela própria rede; não escolhemos depois o modo mais próximo do futuro real. Nenhum ponto real novo é realimentado para corrigir a trajetória prevista.'},
        {'title':'Ajuste da topologia e das taxas por validação',
         'text':'Para a GRU, testamos 128 e 192 unidades ocultas e taxas iniciais de 0,001 e 0,0005. O orçamento é de 20 épocas, lote 128, AdamW, weight decay 0,00001, gradientes limitados em 1 e taxa cossenoidal até 10% da inicial. A escala isotrópica 0,8–1,2 varia somente no treino. Examinamos quatro candidatos com a seed 42 e repetimos os dois finalistas com a seed 43. A decisão usa a média dessas duas inicializações. A seed 44 replica a vencedora sem participar da escolha.\n\n'
                'O checkpoint minimiza a média de ADE/256 +0,1 ×fim precoce nos três horizontes, usando os primeiros 100 desenhos de validação por categoria. A escolha externa usa a validação completa, acrescenta 0,02 ×(1−F1 de pen-up) e compara limiares de caneta 0,2/0,35/0,5. A perda de treino combina classificação do modo, Huber de posição e velocidade, classificação da caneta, erro do relógio e suavidade das correções.\n\n'
                f'A validação escolheu {chosen["spec"]["hidden"]} unidades ocultas, taxa inicial de {number(chosen["spec"]["lr"],3)} e limiar de pen-up de {number(chosen["pen_threshold"],1)}. As quatro execuções de triagem foram reaproveitadas de um estudo com o mesmo protocolo. Todos os candidatos e réplicas partiram de pesos novos. A partição de teste já havia sido observada em estudos anteriores.' ,
         'tables':[('Triagem de previsão: seed 42, menor escore é melhor', ['H','Taxa','Época','Validação'],search),
                   ('Finalistas de previsão: seleção pela média 42/43', ['H','Taxa','Seed 42','Seed 43','Média'],finals)]},
        {'title':'Curvas de aprendizagem da previsão',
         'text':'As curvas acompanham a configuração final da GRU nas três inicializações. A perda composta de treino e o objetivo de trajetória na validação têm fórmulas e populações diferentes; sua escala não deve ser comparada diretamente. A estrela identifica o checkpoint selecionado, preservado para a avaliação e para os exemplos. A escolha não usa os desenhos que aparecem nas páginas seguintes como um novo conjunto de treinamento.',
         'figures':[(learning,'Aprendizagem da GRU ao longo das 20 épocas. Cada estrela marca o checkpoint escolhido na validação.')]},
        {'title':'Avaliação do conjunto e significado das métricas',
         'text':'ADE é a distância média, em pixels, entre cada ponto previsto e o ponto humano correspondente. FDE mede a distância no último evento do horizonte. O F1 de pen-up verifica a separação de traços e o MAE de Δt mede o erro por intervalo em segundos. Uma curva pode ficar próxima e ainda errar um levantamento ou uma pausa; por isso essas dimensões são separadas.\n\n'
                'A tabela preserva as médias das três inicializações sobre todo o teste elegível, com prefixos de 16 e 32 eventos e futuro real disponível até o horizonte avaliado. Horizontes distintos exigem quantidades distintas de futuro disponível. A seleção dos bons exemplos visuais não altera esses valores. A galeria mostra situações em que o modelo funcionou; não afirma que todos os desenhos tenham essa qualidade.',
         'equations':[r'\operatorname{ADE}_k=\frac{1}{k}\sum_{j=1}^{k}\|\widehat{\mathbf p}_{t+j}-\mathbf p_{t+j}\|_2,\qquad\operatorname{FDE}_k=\|\widehat{\mathbf p}_{t+k}-\mathbf p_{t+k}\|_2.'],
         'tables':[('Previsão no teste: médias das seeds 42/43/44', ['k','Janelas','ADE (px)','FDE (px)','F1 caneta','MAE Δt (s)'],results)],
         'figures':[(errors,'ADE e FDE da GRU por horizonte, na média das três inicializações.')]},
        {'title':'Galeria de previsões que fizeram sentido',
         'text':'As páginas seguintes apresentam seis situações positivas, todas da validação: dois casos em cada horizonte 5/15/30 e seis categorias diferentes. Primeiro verificamos erros espaciais baixos, ausência de fim precoce e correspondência de pen-up. Exigimos ainda um trecho contínuo e movimento real não trivial, evitando selecionar apenas uma caneta quase parada. Em seguida inspecionamos as sobreposições para conservar continuações geometricamente coerentes.\n\n'
                'Os limites de ADE foram 8/15/25 pixels e de FDE 15/25/40 pixels, respectivamente. A razão entre os comprimentos previsto e real ficou entre 0,6 e 1,4, com ADE de no máximo 7,5% da extensão observada. A revisão escolheu bons exemplos deliberadamente, sem reamostragem de sementes, troca do modo após ver o alvo ou retoques nos pontos previstos. As regras, as contagens da busca e as coordenadas estão nos artefatos do projeto.\n\n'
                'Em cada prancha, o passado é escuro, o futuro humano é laranja e a previsão da GRU é verde-azulada. A visão geral mostra a coerência da continuação; o zoom e as curvas X/Y revelam diferenças residuais. O restante do desenho além de t+k fica fora dessa avaliação. Esses casos ilustram acertos locais, sem estimar uma taxa geral de sucesso.',
         'tables':[('Casos positivos selecionados para apresentação', ['Categoria','t','k','ADE (px)','FDE (px)'],examples)],
         'pages':pages},
    ]


def gru_figures(gru, output, run_id, metrics):
    folder=output/'figures'
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    for seed in [42,43,44]:
        with (gru/'runs'/f'{run_id}_seed{seed}'/'history.csv').open() as stream:
            history=list(csv.DictReader(stream))
        epochs=[int(row['epoch']) for row in history]
        axes[0].plot(epochs,[float(row['train_objective']) for row in history],label=f'Seed {seed}')
        scores=[float(row['val_rollout_objective']) for row in history]
        line,=axes[1].plot(epochs,scores)
        best=min(range(len(scores)),key=scores.__getitem__)
        axes[1].scatter(epochs[best],scores[best],marker='*',color=line.get_color(),s=85)
    axes[0].set(title='Treino',ylabel='Perda composta')
    axes[1].set(title='Validação',ylabel='Objetivo de trajetória')
    axes[0].legend(fontsize=8)
    learning=folder/'forecast_gru_learning.pdf'
    for ax in axes:
        ax.set_xlabel('Época')
        ax.grid(alpha=.17)
    for path in [learning,learning.with_suffix('.png')]:fig.savefig(path,dpi=190)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,3),layout='constrained')
    horizons=[5,15,30]
    for metric,label in [('ade_pixels','ADE'),('fde_pixels','FDE')]:
        ax.plot(horizons,[metrics[str(k)][metric] for k in horizons],'-o',label=label)
    ax.set(xlabel='Eventos futuros k',ylabel='Erro da GRU (pixels)',xticks=horizons)
    ax.legend();ax.grid(alpha=.17)
    errors=folder/'forecast_gru_error.pdf'
    for path in [errors,errors.with_suffix('.png')]:fig.savefig(path,dpi=190)
    plt.close(fig)
    return learning, errors
