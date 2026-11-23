"""Relatório único em duas partes: previsão de traços e reconhecimento de desenhos."""
import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from ..io import ROOT, read_json, write_json
from .data import represent, permute_strokes
from .gallery import make_gallery
from ..drawing_forecast_compare.report_sections import build_sections as forecast_sections

NAMES = {'gru': 'GRU', 'mlp': 'MLP de imagem'}
COLORS = {'gru': '#087F8C', 'mlp': '#D96646'}
FRACTIONS = [.25, .5, .75, 1.]


def num(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'.replace('.', ',')


def pct(value, digits=1):
    return num(float(value)*100, digits)+'%'


def integer(value):
    return f'{int(value):,}'.replace(',', '.')


def escape(value):
    mapping = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '_': r'\_',
               '#': r'\#', '$': r'\$', '{': r'\{', '}': r'\}',
               '≥': r'$\geq$', '≤': r'$\leq$', '−': '-', '²': r'$^2$',
               '×': r'$\times$', '±': r'$\pm$', 'Δ': r'$\Delta$', '→': r'$\to$'}
    return ''.join(mapping.get(c, c) for c in str(value))


def history(path):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def save(fig, output, name):
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    for ext in ['pdf', 'png']:
        fig.savefig(folder/(name+'.'+ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    return folder/(name+'.pdf')


def draw_points(ax, points, color='#203448', linewidth=2., prefix=None):
    p = np.asarray(points)
    if prefix is not None:
        p = p[:prefix]
    ends = (np.flatnonzero(p[:, 2] > .5)+1).tolist()
    if not ends or ends[-1] != len(p):
        ends.append(len(p))
    start = 0
    for stop in ends:
        if stop-start > 1:
            ax.plot(p[start:stop, 0], p[start:stop, 1], color=color, lw=linewidth,
                    solid_capstyle='round')
        else:
            ax.scatter(p[start:stop, 0], p[start:stop, 1], color=color, s=8)
        start = stop
    ax.set_aspect('equal', adjustable='datalim')
    ax.invert_yaxis()
    ax.axis('off')


def predictions(output, family, seed=42):
    with np.load(output/family/f'test_seed{seed}.npz') as data:
        return data['truth'], data['probabilities']


def run_directory(output, family, selection, seed):
    spec = selection['best']['spec']
    return output/family/f'h{spec["hidden"]}_lr{spec["lr"]}_seed{seed}'


def make_figures(output, config, manifest, selections, metrics, diagnostics, examples):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'legend.frameon': False, 'axes.titlesize': 11})
    figures = {}
    def register(fig, name, caption):
        figures[name] = (save(fig, output, name), caption)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), layout='constrained')
    for ax, family in zip(axes, NAMES):
        rows = sorted(selections[family]['screening'], key=lambda x: x['score'])
        labels = [f"H{r['spec']['hidden']} / η={r['spec']['lr']:g}" for r in rows]
        ax.barh(labels, [r['score'] for r in rows], color=COLORS[family])
        ax.invert_yaxis()
        ax.set(title=NAMES[family]+' · triagem seed 42', xlabel='CE média de validação · menor é melhor')
        ax.grid(axis='x', alpha=.15)
    register(fig, '01_busca', 'Triagem dos quatro candidatos por família, todos inicializados do zero. A seleção utiliza a entropia cruzada média sobre os quatro prefixos de validação; a acurácia final não é o critério de escolha.')

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.1), layout='constrained')
    for col, family in enumerate(NAMES):
        for seed in config['seeds']:
            h = history(run_directory(output, family, selections[family], seed)/'history.csv')
            axes[0, col].plot(h['epoch'], h['train_ce'], label=f'Seed {seed}')
            line, = axes[1, col].plot(h['epoch'], h['val_ce'], label=f'Seed {seed}')
            i = int(np.argmin(h['val_ce']))
            axes[1, col].scatter(h['epoch'][i], h['val_ce'][i], marker='*', s=85, color=line.get_color())
        axes[0, col].set(title=NAMES[family]+' · treino', ylabel='Entropia cruzada')
        axes[1, col].set(title=NAMES[family]+' · validação', ylabel='CE média dos quatro prefixos')
    for ax in axes.flat:
        ax.set_xlabel('Época'); ax.legend(fontsize=8); ax.grid(alpha=.15)
    register(fig, '02_aprendizagem', 'Curvas das configurações escolhidas, nas três inicializações. A estrela marca o checkpoint de menor CE de validação. O treino amostra uma vista por desenho e inclui dropout; a validação usa todas as vistas, sem dropout, portanto as perdas não são estimativas idênticas da mesma distribuição.')

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout='constrained')
    for ax, family in zip(axes, NAMES):
        for seed in config['seeds']:
            h = history(run_directory(output, family, selections[family], seed)/'history.csv')
            ax.plot(h['epoch'], h['val_full_accuracy']*100, label=f'Seed {seed}')
        ax.set(title=NAMES[family], xlabel='Época', ylabel='Acurácia no desenho completo (%)', ylim=(0, 101))
        ax.legend(); ax.grid(alpha=.15)
    register(fig, '03_acuracia_validacao', 'Evolução da acurácia de validação com o desenho completo. Um checkpoint com maior acurácia final pode ter CE pior nos prefixos iniciais; a escolha privilegia o reconhecimento progressivo, e não somente o resultado ao terminar.')

    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    boxes = [(.01, .38, .19, .27, 'Mesmo desenho parcial\nsem categoria fornecida'),
             (.29, .68, .28, .22, 'Sequência de eventos\nL × 6 · ordem e pausas'),
             (.29, .13, .28, .22, 'Imagem binária 32 × 32\nflatten: 1.024 pixels'),
             (.66, .68, .31, .22, 'GRU H → densa 12\nsoftmax de categorias'),
             (.66, .13, .31, .22, 'MLP 1.024 → H → H/2 → 12\nReLU + dropout 0,1')]
    for x, y, w, h, label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.009',facecolor='#EDF4F6',edgecolor='#A8C4CD'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
    for start, stop in [((.2,.55),(.29,.79)),((.2,.45),(.29,.24)),((.57,.79),(.66,.79)),((.57,.24),(.66,.24))]:
        ax.annotate('',xy=stop,xytext=start,arrowprops={'arrowstyle':'->','lw':1.7,'color':'#087F8C'})
    ax.text(.5,.005,'Mudam a representação e a arquitetura. A comparação não isola somente a recorrência.',ha='center',fontsize=10)
    register(fig, '04_arquiteturas', 'Os dois classificadores observam exatamente o mesmo prefixo e recebem a mesma normalização geométrica. O rótulo supervisiona a função de perda; nunca integra a entrada. A MLP principal é uma rede densa convencional, sem convolução, agregação de eventos ou estado recorrente.')

    sample = examples[0]
    points = np.asarray(sample['points'])
    cut = max(2, int(np.floor(len(points)*.5+.5)))
    tokens, image = represent(points[:cut])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), layout='constrained', gridspec_kw={'width_ratios':[1,1,1.65]})
    draw_points(axes[0], points[:cut]); axes[0].set_title(f'Prefixo real · {cut} eventos')
    axes[1].imshow(image,cmap='gray_r',vmin=0,vmax=1);axes[1].axis('off');axes[1].set_title('MLP: somente pixels')
    shown=tokens[:min(cut,36)]
    im=axes[2].imshow(shown.T,aspect='auto',cmap='coolwarm',vmin=-1,vmax=1)
    axes[2].set(yticks=range(6),yticklabels=['Δx','Δy','pen-up','log(1+Δt/0,1)','x','y'],xlabel='Evento observado',title='GRU: sequência de seis atributos')
    fig.colorbar(im,ax=axes[2],shrink=.8,label='Valor (cor limitada a ±1)')
    register(fig, '05_representacoes', 'Duas apresentações do mesmo passado observado. O mapa mostra até 36 eventos e limita apenas a escala de cor; os tensores usados pelas redes não são truncados em valor. O raster é uma união binária dos segmentos de tinta e não revela a ordem, o relógio ou a recência.')

    source = ROOT/'results'/config['source_experiment']
    records = read_json(source/'data/records.json')
    fig, axes = plt.subplots(1,2,figsize=(11,3.5),layout='constrained')
    for split,color in [('train','#203448'),('validation','#087F8C'),('test','#D96646')]:
        lengths=[r['length'] for r in records if r['split']==split]
        axes[0].hist(lengths,bins=np.arange(10,191,10),density=True,histtype='step',lw=1.7,label=split,color=color)
    axes[0].set(xlabel='Eventos por desenho',ylabel='Densidade',title='Comprimentos nas partições fixas');axes[0].legend()
    fractions=np.asarray(FRACTIONS)
    lengths=np.asarray([r['length'] for r in records if r['split']=='test'])
    data=[np.floor(lengths*f+.5) for f in fractions]
    axes[1].boxplot(data,tick_labels=['25%','50%','75%','100%'],showfliers=False)
    axes[1].set(xlabel='Fração observada',ylabel='Eventos disponíveis',title='Mesmo percentual, quantidades diferentes')
    register(fig,'06_dados','Os desenhos inteiros são a unidade da partição. As quatro vistas do mesmo desenho pertencem à mesma partição. A porcentagem observada serve à avaliação; não fornece à rede o comprimento futuro ou a duração final.')

    fig, axes = plt.subplots(1,2,figsize=(11,3.7),layout='constrained')
    for ax,key,title in zip(axes,['accuracy','macro_f1'],['Acurácia','F1 macro']):
        for family in NAMES:
            values=np.array([[r['fractions'][str(f)][key] for f in FRACTIONS] for r in metrics[family]['rows']])
            mean=values.mean(0)*100;std=values.std(0,ddof=1)*100
            ax.errorbar(np.array(FRACTIONS)*100,mean,yerr=std,marker='o',capsize=4,color=COLORS[family],label=NAMES[family])
        ax.set(xlabel='Desenho observado (%)',ylabel=title+' (%)',xticks=[25,50,75,100],ylim=(0,101));ax.legend();ax.grid(alpha=.15)
    register(fig,'07_reconhecimento_progressivo','Desempenho progressivo no teste: média e desvio padrão entre as três inicializações. Ambas as redes podem reconhecer antes do fim. As barras medem variação entre seeds; não são intervalos de confiança populacionais.')

    fig, axes = plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    for ax,key,title in zip(axes,['ce','brier'],['Entropia cruzada','Brier multiclasse']):
        for family in NAMES:
            ax.plot(np.array(FRACTIONS)*100,[metrics[family]['mean'][str(f)][key] for f in FRACTIONS],'-o',color=COLORS[family],label=NAMES[family])
        ax.set(xlabel='Desenho observado (%)',ylabel=title,xticks=[25,50,75,100]);ax.legend();ax.grid(alpha=.15)
    register(fig,'08_probabilidades','Qualidade das probabilidades, além do acerto da classe mais provável. Entropia cruzada e Brier menores são melhores; a CE penaliza fortemente erros com confiança excessiva. O Brier usado soma os erros quadráticos das 12 classes, sem dividir por 12.')

    test={f:predictions(output,f) for f in NAMES}
    fig, axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for ax,fraction_index in zip(axes,[1,3]):
        ax.plot([0,1],[0,1],'--',color='#9CA9B1',lw=1)
        for family in NAMES:
            y,p=test[family];p=p[:,fraction_index];confidence=p.max(-1);correct=p.argmax(-1)==y
            xc=[];yc=[];counts=[]
            bins=np.minimum(np.floor(confidence*10).astype(np.int64),9)
            for bin_index in range(10):
                mask=bins==bin_index
                if mask.any():xc.append(confidence[mask].mean());yc.append(correct[mask].mean());counts.append(mask.sum())
            ax.plot(xc,yc,'-o',color=COLORS[family],label=NAMES[family]);ax.scatter(xc,yc,s=np.array(counts)/8+8,alpha=.25,color=COLORS[family])
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Confiança média do top-1',ylabel='Frequência de acerto',title=f'Calibração · {int(FRACTIONS[fraction_index]*100)}% observado');ax.legend();ax.grid(alpha=.15)
    register(fig,'09_calibracao','Diagramas de confiabilidade da seed 42 em 10 faixas de confiança. A diagonal representa confiança compatível com a frequência empírica de acerto. Marcadores maiores indicam grupos com mais desenhos; faixas vazias são omitidas. Não houve ajuste de temperatura após o treino.')

    fig,axes=plt.subplots(1,2,figsize=(12,5.7),layout='constrained')
    for ax,family in zip(axes,NAMES):
        cm=np.asarray(metrics[family]['rows'][0]['fractions']['1.0']['confusion'])
        normalized=cm/cm.sum(1,keepdims=True)
        im=ax.imshow(normalized,cmap='Blues',vmin=0,vmax=1)
        ax.set(xticks=range(12),xticklabels=manifest['labels'],yticks=range(12),yticklabels=manifest['labels'],title=NAMES[family]+' · 100%',xlabel='Categoria prevista',ylabel='Categoria verdadeira')
        plt.setp(ax.get_xticklabels(),rotation=65,ha='right',fontsize=8);plt.setp(ax.get_yticklabels(),fontsize=8)
        for i in range(12):
            for j in range(12):
                if normalized[i,j]>=.04:
                    ax.text(j,i,f'{normalized[i,j]*100:.0f}',ha='center',va='center',fontsize=8,color='white' if normalized[i,j]>.55 else '#203448')
    fig.colorbar(im,ax=axes,shrink=.7,label='Proporção dentro da categoria')
    register(fig,'10_confusao','Matrizes de confusão normalizadas por categoria, seed 42, no desenho completo. A diagonal reúne os acertos; cada linha soma 100%. Apenas células de pelo menos 4% recebem rótulo, evitando poluição visual. A escala de cor é compartilhada.')

    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained',sharey=True)
    for ax,fraction in zip(axes,[.5,1.]):
        ypos=np.arange(12)
        for family,offset in [('gru',-.18),('mlp',.18)]:
            matrix=np.asarray(metrics[family]['rows'][0]['fractions'][str(fraction)]['confusion'])
            recall=matrix.diagonal()/matrix.sum(1)
            ax.barh(ypos+offset,recall*100,height=.34,color=COLORS[family],label=NAMES[family])
        ax.set(yticks=ypos,yticklabels=manifest['labels'],xlabel='Recall por categoria (%)',xlim=(0,101),title=f'{int(fraction*100)}% observado');ax.invert_yaxis();ax.legend(fontsize=8);ax.grid(axis='x',alpha=.15)
    register(fig,'11_categorias','Sensibilidade por categoria na seed 42, com metade e com a totalidade do desenho. Médias globais escondem pares difíceis e ganhos desiguais. O teste tem 250 desenhos por categoria; estes resultados não indicam desempenho em categorias ausentes.')

    paired=diagnostics.get('paired_bootstrap',[])
    if paired:
        fig,ax=plt.subplots(figsize=(10,3.5),layout='constrained')
        x=np.array([r['fraction'] for r in paired])*100;y=np.array([r['delta_accuracy'] for r in paired])*100
        low=np.array([r['ci95'][0] for r in paired])*100;high=np.array([r['ci95'][1] for r in paired])*100
        ax.errorbar(x,y,yerr=[y-low,high-y],fmt='o-',capsize=6,color=COLORS['gru']);ax.axhline(0,color='#9CA9B1',ls='--')
        ax.set(xticks=[25,50,75,100],xlabel='Desenho observado (%)',ylabel='Acurácia GRU − MLP (pontos percentuais)',title='Diferença pareada · seed 42');ax.grid(alpha=.15)
        register(fig,'12_pareado','Diferença pareada de acurácia e intervalo bootstrap de 95%, reamostrando desenhos inteiros. Valores positivos favorecem a GRU. Esta incerteza amostral condiciona-se aos pesos da seed 42 e não incorpora toda a variabilidade do treino ou a dependência entre autores desconhecidos.')

    # A mesma imagem pode vir de ordens diferentes; não confundir sensibilidade com acerto.
    candidate=next((e for e in examples if np.sum(np.asarray(e['points'])[:,2]>.5)>2),sample)
    original=np.asarray(candidate['points']);cut=max(2,int(np.floor(len(original)*.75+.5)));original=original[:cut]
    original=original.copy();original[:,3]=np.arange(len(original))*.1;original[-1,2]=1
    reordered=permute_strokes(original)
    _,a=represent(original);_,b=represent(reordered)
    assert np.array_equal(a,b)
    fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
    for ax,p,title in zip(axes[:2],[original,reordered],['Ordem original · relógio + fechamento','Traços em ordem invertida']):
        draw_points(ax,p);ax.set_title(title,fontsize=10)
        starts=np.r_[0,np.flatnonzero(p[:-1,2]>.5)+1]
        for i,start in enumerate(starts):ax.annotate(str(i+1),p[start,:2],xytext=(4,4),textcoords='offset points',fontsize=8,color=COLORS['gru'])
    axes[2].imshow(a,cmap='gray_r',vmin=0,vmax=1);axes[2].axis('off');axes[2].set_title('MLP: imagem exatamente igual')
    register(fig,'13_ordem','Inverter traços inteiros preserva sua geometria e a imagem binária, mas altera a sequência recebida pela GRU. As numerações identificam a ordem de início dos traços. O relógio é uniformizado e o trecho parcial é encerrado em ambas as sequências, evitando atribuir ao reordenamento uma mudança do intervalo ou somente do pen-up final.')

    probes=diagnostics.get('probes',[])
    if probes:
        scenarios=['original','uniform_clock','reordered_strokes_uniform_clock','hidden_penup','zero_clock']
        labels=['Original','Relógio uniforme\n+ fechamento','Ordem de traços\n(relógio + fechamento)','Sem pen-up\nna GRU','Sem relógio\nna GRU']
        fig,axes=plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
        lookup={(r.get('family',r.get('model')),r['scenario']):r for r in probes}
        for family,offset in [('gru',-.18),('mlp',.18)]:
            rows=[lookup[(family,s)] for s in scenarios]
            axes[0].barh(np.arange(5)+offset,[r['accuracy']*100 for r in rows],height=.34,color=COLORS[family],label=NAMES[family])
            axes[1].barh(np.arange(5)+offset,[r['prediction_change_fraction']*100 for r in rows],height=.34,color=COLORS[family],label=NAMES[family])
        for ax in axes:
            ax.set(yticks=range(5),yticklabels=labels,xlim=(0,101));ax.invert_yaxis();ax.legend(fontsize=8);ax.grid(axis='x',alpha=.15)
        axes[0].set_xlabel('Acurácia (%)');axes[0].set_title('Qualidade sob perturbação')
        axes[1].set_xlabel('Classes top-1 alteradas (%)');axes[1].set_title('Sensibilidade da decisão')
        register(fig,'14_perturbacoes','Sondas exploratórias em 240 desenhos de validação balanceados, com 75% dos eventos. A alteração de classe em ordem invertida usa como referência a ordem original com relógio também uniforme e fechamento comum do trecho parcial; as demais alterações usam o original. Pen-up e relógio são ocultados apenas nos tokens da GRU, conservando o raster da MLP. As sondas são fora da distribuição, sem retreino, e não estimam efeitos causais.')

    if diagnostics.get('crossing_groups'):
        fig,ax=plt.subplots(figsize=(10,3.5),layout='constrained')
        rows=diagnostics['crossing_groups'];groups=list(dict.fromkeys(str(r['group']) for r in rows));lookup={(r.get('family',r.get('model')),str(r['group'])):r for r in rows}
        for family,offset in [('gru',-.18),('mlp',.18)]:
            chosen=[lookup[(family,g)] for g in groups]
            ax.bar(np.arange(len(groups))+offset,[r['accuracy']*100 for r in chosen],width=.34,color=COLORS[family],label=NAMES[family])
            for i,r in enumerate(chosen):ax.text(i+offset,r['accuracy']*100+1,f'n={r["n"]}',ha='center',fontsize=9)
        ax.set(xticks=range(len(groups)),xticklabels=['Sem cruzamentos' if g in ['0','none','sem','sem_cruzamentos'] else 'Com cruzamentos' for g in groups],ylabel='Acurácia (%)',ylim=(0,105),title='Desempenho e cruzamentos · 75% observado · seed 42');ax.legend();ax.grid(axis='y',alpha=.15)
        register(fig,'15_cruzamentos','Comparação descritiva entre desenhos com e sem cruzamentos no prefixo de teste. Os grupos podem diferir em categoria, número de pontos e complexidade; uma diferença não demonstra imunidade a sobreposição nem efeito isolado dos cruzamentos.')

    histories=[e for e in examples if len(e.get('history',[]))>=4]
    if histories:
        chosen=[]
        for prefix in ['favorece gru','favorece mlp']:
            found=next((e for e in histories if prefix in e.get('selection_rule','').lower() and e not in chosen),None)
            if found:chosen.append(found)
        for e in histories:
            if len(chosen)>=4:break
            if e not in chosen:chosen.append(e)
        fig,axes=plt.subplots(len(chosen),2,figsize=(11,2.45*len(chosen)),layout='constrained',gridspec_kw={'width_ratios':[1,2.1]},squeeze=False)
        for row,e in enumerate(chosen):
            draw_points(axes[row,0],e['points']);axes[row,0].set_title(manifest['labels'][e['label']]+' · validação',fontsize=10)
            hs=e['history'];x=[float(h['fraction'])*100 for h in hs]
            for family in NAMES:axes[row,1].plot(x,[h[family][e['label']] for h in hs],'-o',ms=3,color=COLORS[family],label=NAMES[family])
            axes[row,1].set(ylim=(0,1.02),xlabel='Desenho observado (%)',ylabel='P(categoria verdadeira)');axes[row,1].legend(fontsize=8);axes[row,1].grid(alpha=.15)
        register(fig,'16_exemplos','Desenhos reais de validação e probabilidade atribuída à categoria verdadeira à medida que os eventos chegam. A figura mostra o desenho completo apenas para orientar o leitor; cada ponto da curva é calculado com o prefixo correspondente. Os exemplos são escolhidos por regras determinísticas registradas, incluindo diferenças favoráveis a cada rede; não estimam a média do teste.')
    return figures


def render(output, abstract, sections, references):
    folder=output/'report';folder.mkdir(exist_ok=True)
    title='Previsão de traços e reconhecimento de desenhos com redes neurais'
    tex=r'''\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage[brazilian]{babel}
\usepackage[margin=2cm]{geometry}\usepackage{graphicx,booktabs,longtable,amsmath,amssymb,hyperref,float,xcolor,fancyhdr,array}
\definecolor{ink}{HTML}{203448}\definecolor{accent}{HTML}{087F8C}
\hypersetup{colorlinks=true,urlcolor=accent,linkcolor=ink,pdftitle={Previsão de traços e reconhecimento de desenhos},pdfauthor={Pedro Martineli}}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\setlength{\headheight}{14pt}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small Projeto 4 — Modelagem temporal de desenhos}\fancyhead[R]{\small Quick, Draw!}\fancyfoot[C]{\thepage}
\begin{document}\begin{center}{\Large\bfseries\color{ink} '''+escape(title)+r'''}\\[5pt]
Introdução às Redes Neurais Artificiais\\Pedro Martineli (RA: 193474)
\end{center}\textbf{Resumo.} '''+escape(abstract)+'\n'
    md='# '+title+'\n\n'+abstract+'\n\n'
    in_part=False
    for section in sections:
        heading=section['title'];text=section['text'];level=section.get('level',1)
        if section.get('macro_title'):
            tex+='\\clearpage\n\\part{'+escape(section['macro_title'])+'}\n'
            md+='## '+section['macro_title']+'\n\n'
            in_part=True
        elif section.get('start_on_new_page'):
            tex+='\\clearpage\n'
        tex+='\\'+('section' if level==1 else 'subsection')+('*' if section.get('unnumbered') else '')+'{'+escape(heading)+'}\n'+escape(text)+'\n'
        md+='#'*(level+1+int(in_part))+' '+heading+'\n\n'+text+'\n\n'
        for equation in section.get('equations',[]):
            tex+='\\[\n'+equation+'\n\\]\n';md+='$$\n'+equation+'\n$$\n\n'
        for caption,head,rows in section.get('tables',[]):
            md+='**'+caption+'**\n\n| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n\n'
            wide=len(head)==3 and any(len(str(c))>38 for r in rows for c in r)
            columns=r'p{.22\linewidth}p{.34\linewidth}p{.34\linewidth}' if wide else 'l'+'r'*(len(head)-1)
            tex+='\\begingroup\\small\\setlength{\\tabcolsep}{4pt}\\begin{longtable}{'+columns+'}\n\\caption{'+escape(caption)+r'}\\'+'\n\\toprule\n'
            header=' & '.join(map(escape,head))+r' \\'+'\n'
            tex+=header+'\\midrule\\endfirsthead\n\\toprule\n'+header+'\\midrule\\endhead\n\\bottomrule\\endfoot\n'
            tex+='\n'.join(' & '.join(map(escape,row))+r' \\' for row in rows)+'\n\\end{longtable}\\endgroup\n'
        for path,caption in section.get('figures',[]):
            path=Path(path)
            if not path.exists():raise FileNotFoundError(path)
            target=output/'figures'/path.name
            if target.resolve()!=path.resolve():shutil.copy2(path,target)
            relative='../figures/'+target.name
            tex+='\\begin{figure}[H]\\centering\\includegraphics[width=\\linewidth,height=.57\\textheight,keepaspectratio]{'+relative+'}\\caption{'+escape(caption)+'}\\end{figure}\n'
            png=path.with_suffix('.png')
            if png.exists():
                target_png=target.with_suffix('.png')
                if target_png.resolve()!=png.resolve():shutil.copy2(png,target_png)
                relative='../figures/'+target_png.name
            md+='!['+caption+']('+relative+')\n\n'+caption+'\n\n'
        for page in section.get('pages', []):
            tex+='\\clearpage\n\\subsection*{'+escape(page['title'])+'}\n'+escape(page['text'])+'\n'
            path=Path(page['figure']);relative='../figures/'+path.name
            tex+='\\begin{figure}[H]\\centering\\includegraphics[width=\\linewidth,height=.75\\textheight,keepaspectratio]{'+relative+'}\\caption{'+escape(page['caption'])+'}\\end{figure}\n'
            md+='#### '+page['title']+'\n\n'+page['text']+'\n\n!['+page['caption']+'](../figures/'+path.with_suffix('.png').name+')\n\n'+page['caption']+'\n\n'
        if section.get('pages'):
            tex+='\\clearpage\n'
    tex+='\\section*{Referências}\\begin{enumerate}\n';md+='## Referências\n\n'
    for label,url in references:
        tex+='\\item \\href{'+url+'}{'+escape(label)+'}\n';md+=f'- [{label}]({url})\n'
    tex+='\\end{enumerate}\\small Dados: Google Quick, Draw!, CC BY 4.0. Código e redação elaborados com auxílio de IA; resultados acompanhados de artefatos e verificações numéricas.\\end{document}'
    (folder/'relatorio.tex').write_text(tex);(folder/'relatorio.md').write_text(md)
    for _ in range(2):
        run=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','relatorio.tex'],cwd=folder,capture_output=True,text=True,encoding='utf-8',errors='replace')
        if run.returncode:
            (folder/'compile_error.txt').write_text(run.stdout+run.stderr)
            raise RuntimeError(str(folder/'compile_error.txt'))
    (folder/'compile_error.txt').unlink(missing_ok=True)


def accuracy_figure(output, metrics):
    fig, ax = plt.subplots(figsize=(8, 3.5), layout='constrained')
    horizontal = [25, 50, 75, 100]
    for family, offset in [('gru', 9), ('mlp', -16)]:
        scores = [100*metrics[family]['mean'][str(f)]['accuracy'] for f in FRACTIONS]
        ax.plot(horizontal, scores, '-o', color=COLORS[family], lw=2, label=NAMES[family])
        for x, y in zip(horizontal, scores):
            ax.annotate(num(y, 1)+'%', (x,y), xytext=(0,offset), textcoords='offset points',
                        ha='center', color=COLORS[family], fontsize=10)
    ax.set(xlabel='Eventos do desenho observados (%)', ylabel='Acurácia no teste (%)',
           xticks=horizontal, xlim=(18,107), ylim=(0,100))
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=.18)
    return save(fig,output,'reconhecimento_acuracia_apresentacao')


def build(output=None):
    output = Path(output or ROOT/'results/drawing_recognition_compare_v2')
    config = read_json(output/'config.json')
    manifest = read_json(output/'data_manifest.json')
    selections = {f:read_json(output/f/'frozen_selection.json') for f in NAMES}
    metrics = read_json(output/'test_metrics.json')
    diagnostics = read_json(output/'diagnostics.json')
    examples = read_json(ROOT/'site/assets/recognition/examples.json')['examples']
    figures = make_figures(output,config,manifest,selections,metrics,diagnostics,examples)
    full_gallery = make_gallery(output,examples,manifest['labels'])
    # Dois contrastes favoráveis à GRU, um acerto compartilhado e um contraste favorável à MLP.
    exhibit_ids = ['5421316351983616','6342047466258432','6511453274112000','6141759400706048']
    gallery_pages = [next(p for p in full_gallery if drawing_id in p['figure'].stem)
                     for drawing_id in exhibit_ids]
    captions = [
        ('Peixe · a GRU encontra a categoria',
         'Nos últimos eventos, a GRU reconhece o peixe e a MLP dá sua maior probabilidade a Cachorro.'),
        ('Flor · a GRU acerta com metade do desenho',
         'Os palpites dividem-se enquanto o contorno ainda está incompleto.'),
        ('Guarda-chuva · as duas redes reconhecem',
         'Os novos traços tornam a forma mais clara para as duas redes.'),
        ('Carro · a MLP reconhece mais cedo neste caso',
         'Neste desenho, com 50% dos pontos, a MLP acerta e a GRU ainda erra.'),
    ]
    for page,(title,description) in zip(gallery_pages,captions):
        page['title'] = title
        page['text'] = description
        page['caption'] = ('As colunas revelam 10%, 25%, 50%, 75% e 100% dos eventos. A imagem em preto e branco é a entrada da MLP. '
                           'As barras mostram os três palpites de cada rede. A curva mostra a probabilidade da categoria correta.')
    widths = [[NAMES[f],selections[f]['best']['spec']['hidden'],
               num(selections[f]['best']['spec']['lr'],3)] for f in NAMES]
    chart = accuracy_figure(output,metrics)
    half_gru,half_mlp = (metrics[f]['mean']['0.5']['accuracy'] for f in NAMES)
    full_gru,full_mlp = (metrics[f]['mean']['1.0']['accuracy'] for f in NAMES)
    overview = {
        'title':'O que fizemos?', 'unnumbered':True,
        'text':'Usamos o Quick, Draw! para estudar duas tarefas. Primeiro, a GRU prevê os próximos movimentos da caneta a partir do trecho observado e da categoria fornecida. Depois, a GRU e a MLP tentam reconhecer a categoria de um desenho parcial.\n\n'
               'Cada desenho guarda a categoria, as coordenadas dos pontos, seus tempos e os limites entre traços. Esses registros formam uma série temporal porque a ordem e os intervalos dos movimentos descrevem como o desenho foi feito. Trabalhamos com 42.000 desenhos de 12 categorias, divididos por desenho inteiro em 36.000 para treino, 3.000 para validação e 3.000 para teste. A validação orienta a escolha dos modelos. O desempenho é medido no teste.\n\n'
               'O teste usa as mesmas partições dos estudos anteriores do projeto. O recorte contém desenhos reconhecidos pelo próprio jogo.',
        'tables':[('As duas tarefas do projeto',['Tarefa','Entrada','Saída'],[
            ['Previsão t+k','Movimentos e categoria','Próximos pontos da caneta'],
            ['Reconhecimento','Movimentos ou imagem parcial','Probabilidades das 12 categorias']])],
    }
    metrics_guide = [
        {'title':'Como medimos os resultados', 'unnumbered':True, 'start_on_new_page':True,
         'text':'Na previsão, medimos a distância entre os pontos previstos e os pontos registrados. No reconhecimento, contamos quantas categorias as redes acertam.'},
        {'title':'Previsão — ADE e FDE', 'level':2, 'unnumbered':True,
         'text':'Para cada passo futuro, comparamos o ponto previsto com o ponto registrado no mesmo passo. Calculamos a distância em linha reta, em pixels, a partir das diferenças em X e Y. ADE (Average Displacement Error) é a média dessas distâncias ao longo dos k passos. FDE (Final Displacement Error) usa apenas a distância no último passo. Menor é melhor nas duas métricas.\n\n'
                'Exemplo com cinco passos. Se as distâncias são 2, 4, 4, 5 e 10 pixels, o ADE é (2 + 4 + 4 + 5 + 10) / 5 = 5 pixels e o FDE é 10 pixels. Nesse caso, o ponto final ficou mais distante do que a média do trajeto.',
         'equations':[
             r'd_j=\sqrt{(\widehat{x}_{t+j}-x_{t+j})^2+(\widehat{y}_{t+j}-y_{t+j})^2},\qquad'
             r'\mathrm{ADE}_k=\frac{1}{k}\sum_{j=1}^{k}d_j,\qquad\mathrm{FDE}_k=d_k.',
         ]},
        {'title':'Reconhecimento — acurácia', 'level':2, 'unnumbered':True,
         'text':'Para cada desenho, usamos como palpite a categoria com maior probabilidade. Se ela coincide com a categoria correta, contamos um acerto. A acurácia é a porcentagem de desenhos acertados. Calculamos essa porcentagem separadamente com 25%, 50%, 75% e 100% dos eventos disponíveis. As duas redes recebem o mesmo trecho em cada corte.\n\n'
                'Exemplo com dez desenhos pela metade. Se o primeiro palpite da rede acerta a categoria em oito deles, a acurácia nesse corte é 8 / 10 = 80%.',
         'equations':[r'\text{Acurácia (\%)}=100\,\frac{\text{número de desenhos acertados}}{\text{número de desenhos avaliados}}.']},
        {'title':'De onde vêm os valores do relatório?', 'level':2, 'unnumbered':True,
         'text':'Na tabela de previsão, calculamos ADE e FDE em cada janela elegível de teste e fazemos a média das janelas para aquele horizonte. No gráfico de reconhecimento, contamos os acertos nos 3.000 desenhos de teste em cada corte. Por fim, fazemos a média dos resultados das três inicializações em cada tarefa.'},
    ]
    forecast = forecast_sections(ROOT/'results/drawing_forecast_compare_v1',output)
    recognition = [
        {'macro_title':'Reconhecimento da categoria do desenho',
         'title':'O que cada rede observa',
         'text':'As duas redes classificam os mesmos desenhos parciais. A GRU recebe pontos em ordem, com movimentos, tempos e fins de traço. A MLP recebe a imagem desses mesmos pontos, com 32 × 32 pixels achatados em um vetor. Cada rede devolve 12 probabilidades, e a categoria com maior probabilidade é o palpite.\n\n'
                'Para cada rede, testamos duas larguras e taxas iniciais de 0,001 e 0,003. Os treinos duraram 32 épocas. Escolhemos a configuração pela perda de validação nos prefixos de 25%, 50%, 75% e 100% do desenho. As configurações vencedoras foram treinadas com três inicializações.',
         'tables':[('Configurações escolhidas na validação',['Rede','Largura principal','Taxa inicial'],widths)],
         'figures':[(figures['02_aprendizagem'][0], 'Curvas de treino e validação das duas redes. As estrelas marcam as épocas escolhidas com base na validação.')]},
        {'title':'Acurácia conforme o desenho aparece',
         'start_on_new_page':True,
         'text':f'Com metade do desenho, a GRU acertou {pct(half_gru)} e a MLP acertou {pct(half_mlp)}. Com o desenho completo, os resultados foram {pct(full_gru)} e {pct(full_mlp)}, respectivamente. Nos quatro cortes avaliados, a GRU teve maior acurácia. As redes usam arquiteturas e representações diferentes, então a comparação mede o efeito dessas escolhas em conjunto.\n\n'
                'As próximas páginas mostram quatro desenhos de validação com resultados diferentes. No peixe desenhado livremente no site, a GRU atribuiu 97,5% a Peixe e o palpite principal da MLP foi Cachorro (33,9%). O peixe mostrado a seguir é outro desenho, de validação, com uma confusão semelhante.',
         'figures':[(chart,'Acurácia depois de observar 25%, 50%, 75% e 100% dos eventos.')],
         'pages':gallery_pages},
        {'title':'O que os resultados mostram',
         'text':'A GRU consegue continuar trechos curtos da caneta em exemplos como os da galeria. No teste, a previsão fica mais difícil conforme o horizonte cresce. No reconhecimento, as duas redes acertam mais quando recebem uma parte maior do desenho. A GRU obteve a maior acurácia média nos quatro cortes, e a galeria também inclui um exemplo em que a MLP acerta mais cedo.'},
    ]
    abstract = ('Comparamos duas tarefas sobre 42 mil desenhos do Quick, Draw!. Uma GRU prevê os próximos '
                '5, 15 ou 30 movimentos da caneta. Para reconhecer desenhos parciais, comparamos uma GRU '
                'que acompanha a sequência de pontos com uma MLP que recebe a imagem. Com metade do desenho, '
                f'a acurácia foi de {pct(half_gru)} na GRU e {pct(half_mlp)} na MLP.')
    sections = [overview]+metrics_guide+forecast+recognition
    references = [
        ('Google Creative Lab — Quick, Draw! Dataset e licença CC BY 4.0','https://github.com/googlecreativelab/quickdraw-dataset'),
        ('Cho et al. (2014) — artigo que apresenta a GRU','https://arxiv.org/abs/1406.1078'),
        ('Laboratório de previsão da GRU','https://martinelipedro.github.io/projeto4-irn/forecast/'),
        ('Laboratório de reconhecimento da GRU e da MLP','https://martinelipedro.github.io/projeto4-irn/'),
        ('Código, galeria completa e resultados do projeto','https://github.com/martinelipedro/projeto4-irn'),
    ]
    render(output,abstract,sections,references)
    write_json(output/'report/report_manifest.json',{
        'task':'forecast_and_recognition','edition':'presentation',
        'macro_sections':[s['macro_title'] for s in sections if 'macro_title' in s],
        'forecast_horizons':[5,15,30],'forecast_success_cases':6,'forecast_real_future_shown':False,
        'metrics_explained':['ADE','FDE','accuracy'],
        'recognition_comparison_metrics':['accuracy'],'recognition_cases_in_pdf':len(gallery_pages),
        'gallery_cases':len(full_gallery),'gallery_snapshots':5*len(full_gallery),
        'figures':sum(len(s.get('figures',[]))+len(s.get('pages',[])) for s in sections),
        'models':{'forecast':['GRU'],'recognition':['GRU','conventional image MLP']},
        'test_previously_observed':True,
    })
    print('Relatório gerado:',output/'report/relatorio.pdf',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path);args=parser.parse_args();build(args.directory)
