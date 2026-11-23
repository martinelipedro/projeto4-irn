"""Relatório único: reconhecimento GRU/MLP convencional e previsão secundária."""
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
    title='Reconhecimento progressivo de desenhos: GRU temporal versus MLP de imagem'
    tex=r'''\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage[brazilian]{babel}
\usepackage[margin=2cm]{geometry}\usepackage{graphicx,booktabs,longtable,amsmath,amssymb,hyperref,float,xcolor,fancyhdr,array}
\definecolor{ink}{HTML}{203448}\definecolor{accent}{HTML}{087F8C}
\hypersetup{colorlinks=true,urlcolor=accent,linkcolor=ink,pdftitle={Reconhecimento progressivo: GRU temporal versus MLP de imagem},pdfauthor={Pedro Martineli}}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\setlength{\headheight}{14pt}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small Projeto 4 — Reconhecimento de séries temporais}\fancyhead[R]{\small Quick, Draw!}\fancyfoot[C]{\thepage}
\begin{document}\begin{center}{\Large\bfseries\color{ink} '''+escape(title)+r'''}\\[5pt]
Introdução às Redes Neurais Artificiais\\Pedro Martineli (RA: 193474)
\end{center}\textbf{Resumo.} '''+escape(abstract)+'\n'
    md='# '+title+'\n\n'+abstract+'\n\n'
    for section in sections:
        heading=section['title'];text=section['text'];level=section.get('level',1)
        if section.get('start_on_new_page'):
            tex+='\\clearpage\n'
        tex+='\\'+('section' if level==1 else 'subsection')+'{'+escape(heading)+'}\n'+escape(text)+'\n'
        md+='#'*(level+1)+' '+heading+'\n\n'+text+'\n\n'
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
            md+='### '+page['title']+'\n\n'+page['text']+'\n\n!['+page['caption']+'](../figures/'+path.with_suffix('.png').name+')\n\n'+page['caption']+'\n\n'
        if section.get('pages'):
            tex+='\\clearpage\n'
    tex+='\\section*{Referências}\\begin{enumerate}\n';md+='## Referências\n\n'
    for label,url in references:
        tex+='\\item \\href{'+url+'}{'+escape(label)+'}\n';md+=f'- [{label}]({url})\n'
    tex+='\\end{enumerate}\\small Dados: Google Quick, Draw!, CC BY 4.0. Código e redação elaborados com auxílio de IA; resultados acompanhados de artefatos e verificações numéricas.\\end{document}'
    (folder/'relatorio.tex').write_text(tex);(folder/'relatorio.md').write_text(md)
    for _ in range(2):
        run=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','relatorio.tex'],cwd=folder,capture_output=True,text=True)
        if run.returncode:
            (folder/'compile_error.txt').write_text(run.stdout+run.stderr)
            raise RuntimeError(str(folder/'compile_error.txt'))
    (folder/'compile_error.txt').unlink(missing_ok=True)


def build(output=None):
    output=Path(output or ROOT/'results/drawing_recognition_compare_v2')
    config=read_json(output/'config.json');manifest=read_json(output/'data_manifest.json')
    selections={f:read_json(output/f/'frozen_selection.json') for f in NAMES}
    metrics=read_json(output/'test_metrics.json')
    diagnostic_path=output/'diagnostics.json'
    if not diagnostic_path.exists():raise FileNotFoundError('Aguarde diagnostics.json antes de gerar o relatório final.')
    diagnostics=read_json(diagnostic_path)
    example_path=ROOT/'site/assets/recognition/examples.json'
    raw=read_json(example_path);examples=raw['examples'] if isinstance(raw,dict) else raw
    figures=make_figures(output,config,manifest,selections,metrics,diagnostics,examples)
    gallery_pages=make_gallery(output,examples,manifest['labels'])
    def fs(*names):return [figures[n] for n in names if n in figures]
    screen=[];finals=[];architectures=[]
    for family in NAMES:
        selection=selections[family]
        for r in selection['screening']:
            screen.append([NAMES[family],r['spec']['hidden'],num(r['spec']['lr'],4),r['best_epoch'],num(r['score'])])
        for r in selection['finalists']:
            finals.append([NAMES[family],r['spec']['hidden'],num(r['spec']['lr'],4),num(r['seed_scores'][0]),num(r['seed_scores'][1]),num(r['score'])])
        spec=selection['best']['spec']
        r=read_json(run_directory(output,family,selection,42)/'summary.json')
        architectures.append([NAMES[family],spec['hidden'],num(spec['lr'],4),integer(r['parameters']),num(r['seconds'],1),r['best_epoch']])
    recognition=[];probabilistic=[];seeds=[];paired=[];probes=[];stable_rows=[]
    for f in FRACTIONS:
        for family in NAMES:
            row=metrics[family]['mean'][str(f)]
            recognition.append([pct(f,0),NAMES[family],pct(row['accuracy']),num(row['macro_f1']),pct(row['top3'])])
            probabilistic.append([pct(f,0),NAMES[family],num(row['ce']),num(row['ece']),num(row['brier'])])
    for family in NAMES:
        for r in metrics[family]['rows']:
            seeds.append([NAMES[family],r['seed'],*[pct(r['fractions'][str(f)]['accuracy']) for f in FRACTIONS]])
    for r in diagnostics.get('early_stable',[]):
        stable_rows.append([NAMES[r['family']],*[pct(r['stable_by_fraction'][str(f)]) for f in FRACTIONS],pct(r['never_stably_correct'])])
    for r in diagnostics.get('paired_bootstrap',[]):
        paired.append([pct(r['fraction'],0),integer(r['n_drawings']),num(r['delta_accuracy']*100,2),'['+num(r['ci95'][0]*100,2)+'; '+num(r['ci95'][1]*100,2)+']'])
    scenario_names={'original':'Original','uniform_clock':'Relógio + fecho','reordered_strokes_uniform_clock':'Traços invertidos','hidden_penup':'Sem pen-up','zero_clock':'Sem relógio'}
    for r in diagnostics.get('probes',[]):
        family=r.get('family',r.get('model'))
        probes.append([NAMES[family],scenario_names.get(r['scenario'],r['scenario']),pct(r['accuracy']),pct(r['prediction_change_fraction']),num(r['mean_abs_probability_change'],4)])
    probe_lookup={(r.get('family',r.get('model')),r['scenario']):r for r in diagnostics.get('probes',[])}
    order_summary=''
    if all((family,scenario) in probe_lookup for family in NAMES for scenario in ['original','uniform_clock','reordered_strokes_uniform_clock','hidden_penup','zero_clock']):
        a=probe_lookup[('gru','reordered_strokes_uniform_clock')];b=probe_lookup[('gru','uniform_clock')];m=probe_lookup[('mlp','reordered_strokes_uniform_clock')]
        order_summary=(f'No reordenamento com relógio controlado e fechamento comum do trecho parcial, a GRU alterou o top-1 em {pct(a["prediction_change_fraction"])} dos desenhos e sua acurácia passou de {pct(b["accuracy"])} para {pct(a["accuracy"])}. A MLP alterou o top-1 em {pct(m["prediction_change_fraction"])}; sua entrada permaneceu idêntica. '
                       f'Na GRU, ocultar pen-up resultou em acurácia de {pct(probe_lookup[("gru","hidden_penup")]["accuracy"])} e ocultar o relógio em {pct(probe_lookup[("gru","zero_clock")]["accuracy"])}; o original atingiu {pct(probe_lookup[("gru","original")]["accuracy"])}. ')
    gpu=read_json(run_directory(output,'gru',selections['gru'],42)/'summary.json')['gpu']
    full_gru=metrics['gru']['mean']['1.0']['accuracy'];full_mlp=metrics['mlp']['mean']['1.0']['accuracy']
    initial_gru=metrics['gru']['mean']['0.25']['accuracy'];initial_mlp=metrics['mlp']['mean']['0.25']['accuracy']
    difference=(full_gru-full_mlp)*100
    conclusion=(f'No desenho completo, a acurácia média foi {pct(full_gru)} para a GRU e {pct(full_mlp)} para a MLP, '
                f'uma diferença de {num(difference,2)} pontos percentuais (GRU menos MLP). '
                f'Com apenas 25% dos eventos, os valores foram {pct(initial_gru)} e {pct(initial_mlp)}, respectivamente. ')
    abstract=('Investigamos o reconhecimento progressivo de 12 categorias de desenhos do Quick, Draw!. Uma GRU recebe eventos ordenados da caneta; uma MLP convencional recebe a imagem binária do mesmo prefixo, sem ordem de execução. '
              'Selecionamos hiperparâmetros por validação, treinamos do zero e repetimos as configurações escolhidas com três inicializações. '+conclusion+
              'A análise inclui aprendizado, probabilidades, confusões, exemplos e perturbações de ordem, pausas e pen-up. A comparação altera arquitetura e representação conjuntamente e não prova superioridade universal da recorrência. Um estudo secundário, explicitamente separado nas perguntas e métricas, examina a previsão de eventos futuros.')
    sections=[
        {'title':'O que fizemos?',
         'text':'Construímos um sistema que recebe um desenho ainda em andamento e tenta reconhecer qual objeto está sendo desenhado. Comparamos duas formas de observar o mesmo desenho parcial: uma GRU acompanha a sequência de movimentos da caneta, enquanto uma MLP convencional observa sua forma como imagem. As duas redes atualizam seus palpites conforme chegam novos pontos.\n\n'
                'Os dados vêm do Google Quick, Draw!, um jogo em que pessoas desenham objetos solicitados. Cada registro associa uma categoria a uma sequência de traços, com coordenadas X/Y e tempos dos pontos. Derivamos os levantamentos da caneta dos limites desses traços. Assim, podemos revelar o desenho aos poucos e estudar uma série temporal multivariada, em vez de usar apenas sua imagem final.\n\n'
                f'Nosso recorte contém {integer(sum(manifest["counts"].values()))} desenhos de 12 categorias, como peixe, gato, casa e bicicleta. São {integer(manifest["counts"]["train"])} desenhos para aprender os padrões, {integer(manifest["counts"]["validation"])} para ajustar as configurações e {integer(manifest["counts"]["test"])} para avaliar os modelos escolhidos. Retivemos apenas desenhos reconhecidos pelo próprio jogo e mantivemos cada desenho inteiro em uma única partição.\n\n'
                'Transformamos cada desenho em versões parciais. A GRU recebe movimentos, pausas e posições em ordem; a MLP recebe os pixels da imagem parcial. A categoria verdadeira serve somente para ensinar e avaliar, nunca como entrada. A saída de ambas é uma probabilidade para cada uma das 12 categorias.\n\n'
                f'Treinamos as redes do zero, buscamos larguras e taxas de aprendizagem na validação e repetimos as configurações escolhidas com {len(config["seeds"])} inicializações. Comparamos acertos, erros e probabilidades ao longo do desenho, além de testar mudanças de ordem e de canais temporais. O site executa os modelos treinados e a galeria mostra exemplos reais passo a passo. O reconhecimento é o estudo principal; a previsão de próximos pontos aparece como experimento secundário.',
         'tables':[('Resumo da comparação principal', ['Rede','O que recebe','O que devolve'],[
             ['GRU temporal','Movimentos da caneta em ordem','Probabilidades de 12 categorias'],
             ['MLP convencional','Imagem parcial em 1.024 pixels','Probabilidades de 12 categorias']])]},
        {'title':'Objetivo e ajuste dos hiperparâmetros',
         'start_on_new_page':True,
         'text':'O objetivo principal é classificar uma janela temporal crescente de movimentos da caneta: reconhecer o objeto antes de o usuário terminar. Em vez de fornecer a categoria e pedir uma continuação, como no estudo de previsão, removemos a categoria da entrada e pedimos uma distribuição sobre as 12 categorias. A pergunta experimental é como um sistema que lê a sequência se compara a uma MLP convencional que observa somente o desenho parcial como imagem. Não pressupomos que a GRU ganhe em todo prefixo, categoria ou exemplo. '
                'O procedimento de seleção aparece no início para distinguir decisões baseadas na validação das medições de teste. Cada família considera duas larguras e duas taxas, com o mesmo orçamento de 32 épocas. Todos os pesos são novos; não há transferência de redes de previsão ou de reconhecedores anteriores.',
         'tables':[('Espaço e decisões comuns', ['Item','GRU temporal','MLP de imagem'],[
             ['Larguras ocultas','128 ou 192','128 ou 256; segunda camada H/2'],
             ['Taxas iniciais','0,003 ou 0,001','0,003 ou 0,001'],
             ['Treino','AdamW; batch 256; 32 épocas','AdamW; batch 256; 32 épocas'],
             ['Regularização','Dropout 0,1 na cabeça; peso 0,0001','Dropout 0,1 após cada camada oculta; peso 0,0001'],
             ['Checkpoint','Menor CE média dos quatro prefixos de validação','Mesmo critério'],
             ['Inicializações','42/43 para seleção; 44 para replicação','Mesmas seeds e regra']])]},
        {'title':'Busca, replicação e congelamento','level':2,
         'text':'Na primeira etapa treinamos quatro candidatos por família com a seed 42. Repetimos somente os dois melhores com 43; a escolha final minimiza a média das CEs mínimas de validação dessas duas inicializações. A seed 44 treina novamente apenas a configuração vencedora e não participa da decisão. Os dois arquivos de seleção são congelados antes da rotina de teste. O scheduler cosseno reduz a taxa até 10% da inicial; a norma do gradiente é limitada a 1. Cada época apresenta uma das quatro vistas por desenho, amostrada com o mesmo gerador por seed em ambas as famílias. '
                'Um piloto de 16 épocas, restrito à validação, mostrou subajuste e foi interrompido. O protocolo principal passou a 32 épocas e às taxas 0,003/0,001 para ambas as famílias; seus pesos foram reinicializados. O piloto não compõe as tabelas de resultados principais. O teste já havia sido observado em estudos anteriores do projeto; o congelamento atual limita a escolha desta comparação, mas não transforma esse teste em confirmação externa inédita.',
         'tables':[('Triagem: uma inicialização por candidato', ['Rede','H','Taxa','Época','CE val.'],screen),
                   ('Finalistas: média das seeds 42 e 43', ['Rede','H','Taxa','CE 42','CE 43','Média'],finals)],
         'figures':fs('01_busca')},
        {'title':'Capacidade e curvas de aprendizagem','level':2,
         'text':'A MLP de imagem não utiliza Deep Sets. Ela achata a imagem e aplica camadas densas; cada pixel mantém sua posição espacial no vetor, mas a ordem em que as linhas foram desenhadas desaparece. As larguras não igualam exatamente a quantidade de parâmetros: registramos a capacidade e o custo efetivos para tornar essa diferença verificável. Tempos de treino são da configuração escolhida com a seed 42 na GPU registrada; não incluem preparação, busca inteira ou exportação e não medem a latência do navegador. '+f'A GPU do treinamento principal foi {gpu}, com float32, algoritmos determinísticos habilitados e TF32 desligado. '
                'As curvas permitem distinguir subajuste, saturação e afastamento entre treino e validação. Uma menor perda de treino por si só não determina o vencedor: usamos o checkpoint selecionado e depois comparamos as previsões no mesmo teste.',
         'tables':[('Configurações finais e custo observado', ['Rede','H','Taxa','Parâmetros','Segundos','Época'],architectures)],
         'figures':fs('02_aprendizagem','03_acuracia_validacao')},
        {'title':'Dataset e protocolo de avaliação',
         'text':'Usamos 42 mil desenhos do Google Quick, Draw!, divididos em 36 mil de treino, 3 mil de validação e 3 mil de teste. As 12 categorias são: '+', '.join(manifest['labels'])+'. Há 3 mil desenhos de treino e 250 de validação/teste por categoria. A amostra retém somente desenhos reconhecidos pelo jogo, um filtro que facilita o problema e limita a generalização a rabiscos arbitrários. IDs e geometrias são separados por desenho inteiro; não dispomos de um identificador confiável que assegure independência entre autores.\n\n'
                'Os dados preparados preservam coordenadas, fim de traço e timestamps. A amostragem espacial causal conserva limites de traços e instantes originais, com comprimentos entre 12 e 180 eventos. O eixo temporal relevante é a ordem dos eventos da caneta; os intervalos físicos entre eles são irregulares e entram explicitamente na sequência da GRU. Logo, trata-se de classificação de série temporal multivariada, e não de uma coleção de imagens sem evolução.\n\n'
                'Avaliamos quatro prefixos: 25%, 50%, 75% e 100% dos eventos de cada desenho, com arredondamento e mínimo de dois eventos. O comprimento final é usado pelo avaliador para definir esses cortes; não é entregue como atributo à rede. Em uso livre, o sistema recebe somente os eventos disponíveis, sem saber a porcentagem que falta. Nenhuma vista parcial aparece em uma partição diferente da do seu desenho de origem.',
         'tables':[('Divisão fixa por categoria', ['Partição','Desenhos por categoria','Total'],[
             ['Treino','3.000','36.000'],['Validação','250','3.000'],['Teste','250','3.000']])],
         'figures':fs('06_dados')},
        {'title':'Como apresentamos os dados e a saída',
         'text':'Para cada prefixo, calculamos sua própria caixa delimitadora. Subtraímos o centro dessa caixa e dividimos ambas as coordenadas pela maior extensão observada, com piso de uma unidade. Essa normalização é compartilhada pelas duas apresentações e usa somente o passado. Cada novo ponto pode alterar a caixa e, por isso, a demonstração recalcula o prefixo inteiro; não reutiliza um estado oculto calculado em outra normalização. Não usamos a caixa do desenho completo, a duração final ou uma categoria estimada externamente.\n\n'
                'Na GRU, um evento tem seis atributos: deslocamentos normalizados Δx e Δy, indicador pen-up, logaritmo do intervalo de tempo transformado, e posição normalizada x e y. O primeiro deslocamento e o primeiro intervalo são zero. O lote é preenchido até 180 eventos, mas a classificação lê a saída do último evento real; padding à direita não participa dessa decisão. Na MLP, renderizamos apenas os segmentos de tinta, respeitando pen-up, numa imagem binária de 32 por 32 pixels. O vetor de entrada tem 1.024 números, sem timestamp, canal de recência, índice do evento ou rótulo.\n\n'
                'Ambas devolvem 12 logits, transformados em probabilidades por softmax. Na equação, X representa a sequência ou o vetor de pixels conforme a rede. O alvo é o índice da categoria verdadeira. A função de perda é entropia cruzada multiclasse; não prevemos coordenadas ou completamos traços nesta tarefa principal.',
         'equations':[r'\widetilde{\mathbf q}_i=\frac{\mathbf q_i-\mathbf c_t}{\max(\operatorname{largura}_t,\operatorname{altura}_t,1)},\qquad i\leq t,',
                      r'\mathbf u_i=(\Delta\widetilde x_i,\Delta\widetilde y_i,p_i,\log(1+\Delta t_i/0{,}1),\widetilde x_i,\widetilde y_i),',
                      r'\Pr(y=c\mid X_t)=\frac{\exp z_c}{\sum_{j=1}^{12}\exp z_j},\qquad \mathcal L=-\log\Pr(y=y_{\rm real}).'],
         'tables':[('Exemplo numérico de dois eventos: pontos (100,50) e (108,54), separados por 0,2 s',
                    ['Atributo','Evento inicial','Segundo evento'],[
                        ['Δx normalizado','0','1'],['Δy normalizado','0','0,5'],
                        ['Pen-up','0','1'],['log(1 + Δt/0,1)','0','1,099'],
                        ['x normalizado','−0,5','0,5'],['y normalizado','−0,25','0,25']])],
         'figures':fs('04_arquiteturas','05_representacoes')},
        {'title':'O que a recorrência pode acrescentar','level':2,
         'text':'A GRU atualiza um estado oculto em cada evento. Portas de atualização e de reset regulam quanto do estado anterior é retido e como os novos movimentos entram na representação. Isso oferece uma forma aprendida de combinar o caminho dos traços, sua ordem e as pausas. A MLP recebe todos os pixels do prefixo de uma vez e pode reconhecer estrutura espacial, inclusive antes do fim; sua entrada não permite recuperar de maneira única a ordem usada para produzir uma imagem. '
                'Concatenar pontos num vetor e passá-los a uma MLP convencional seria outro controle válido, mas não eliminaria a ordem: cada posição do vetor corresponderia a um instante. Optamos pelo raster porque a pergunta atual é o contraste entre leitura temporal e observação da forma parcial. Como também mudamos a representação, não isolamos uma vantagem causal da GRU sobre toda rede densa possível. Deep Sets permanece apenas no estudo secundário de previsão, com identificação explícita.',
         'equations':[r'\mathbf r_i=\sigma(W_r\mathbf u_i+U_r\mathbf h_{i-1}+\mathbf b_r),\quad \mathbf z_i=\sigma(W_z\mathbf u_i+U_z\mathbf h_{i-1}+\mathbf b_z),',
                      r'\widetilde{\mathbf h}_i=\tanh(W_n\mathbf u_i+\mathbf b_n+\mathbf r_i\odot(U_n\mathbf h_{i-1}+\mathbf b_{hn})),',
                      r'\mathbf h_i=(1-\mathbf z_i)\odot\widetilde{\mathbf h}_i+\mathbf z_i\odot\mathbf h_{i-1}.']},
        {'title':'Métricas e interpretação',
         'text':'A acurácia mede a proporção de categorias top-1 corretas. F1 macro calcula o F1 por categoria e faz a média com pesos iguais, destacando classes difíceis mesmo numa amostra balanceada. Top-3 mede a presença do alvo entre as três hipóteses mais prováveis. A entropia cruzada e o Brier multiclasse avaliam a distribuição inteira, não somente o maior valor. O Brier aqui soma os erros quadráticos das 12 probabilidades em cada desenho. '
                'O ECE agrupa a confiança top-1 em 10 faixas e pondera a diferença absoluta entre confiança média e acurácia. É uma estimativa dependente dos bins, não uma garantia individual de probabilidade correta. Reportamos os resultados médios de três treinos e mostramos sua dispersão. Para a seed 42, um bootstrap pareado por desenho estima o intervalo da diferença de acurácia entre os dois sistemas. A reamostragem por desenho conserva o pareamento entre as redes; não incorpora a variação completa das inicializações nem a dependência entre autores desconhecidos.',
         'equations':[r'\operatorname{Acc}=\frac1N\sum_{n=1}^{N}\mathbf 1[\widehat y_n=y_n],\qquad \operatorname{Brier}=\frac1N\sum_{n=1}^{N}\sum_{c=1}^{12}(p_{nc}-\mathbf 1[y_n=c])^2,',
                      r'\operatorname{ECE}=\sum_{b=1}^{10}\frac{|B_b|}{N}\left|\operatorname{acc}(B_b)-\operatorname{conf}(B_b)\right|.']},
        {'title':'Resultados: reconhecer enquanto o desenho acontece',
         'text':conclusion+'O reconhecimento precoce deve ser medido no mesmo prefixo de cada desenho. Não basta mostrar uma animação de uma imagem final já conhecida: as probabilidades exibidas no software são recalculadas com o passado disponível. Os erros iniciais podem ser plausíveis, pois o mesmo contorno parcial é compatível com vários objetos. A hipótese temporal tem interesse justamente nessa ambiguidade, mas uma classificação pode mudar ao longo do desenho e não há garantia de confiança crescente ou acerto permanente.',
         'tables':[('Reconhecimento no teste: médias das três inicializações', ['Observado','Rede','Acurácia','F1 macro','Top-3'],recognition)],
         'figures':fs('07_reconhecimento_progressivo')},
        {'title':'Probabilidades e confiança','level':2,
         'text':'Uma rede pode acertar mais e ainda fornecer probabilidades mal calibradas. As medidas abaixo complementam o top-1: valores menores de CE, ECE e Brier são desejáveis. Não ajustamos uma temperatura usando o teste. A linha de confiabilidade ajuda a verificar se grupos de desenhos com confiança próxima de 80%, por exemplo, acertam aproximadamente nessa frequência. A confiança exibida para um único rabisco não deve ser interpretada como garantia de correção. '
                f'No desenho completo, o ECE médio foi {num(metrics["gru"]["mean"]["1.0"]["ece"])} para a GRU e {num(metrics["mlp"]["mean"]["1.0"]["ece"])} para a MLP. A rede com maior acurácia não necessariamente tem o menor desvio de calibração. '
                'A comparação por classe e os exemplos revelam situações escondidas pela média. Confusões como casa/castelo, gato/cachorro ou peixe/tubarão são compatíveis com formas compartilhadas, mas a matriz observada é que determina quais pares realmente concentraram erros neste recorte.',
         'tables':[('Qualidade probabilística: médias das três inicializações', ['Observado','Rede','CE','ECE','Brier'],probabilistic)],
         'figures':fs('08_probabilidades','09_calibracao')},
        {'title':'Variação entre treinos e diferenças pareadas','level':2,
         'text':'As três inicializações produzem trajetórias de otimização diferentes, mesmo com configuração e dados fixos. A tabela por seed evita resumir toda a comparação por um único checkpoint favorável. A análise pareada usa a seed 42 para ambas as redes e conserva os mesmos desenhos. Intervalos acima de zero favorecem a GRU; abaixo de zero favorecem a MLP; um intervalo que inclui zero não resolve a diferença sob essa análise. O teste de referência já foi visto no desenvolvimento anterior, de modo que esses intervalos são descritivos do recorte, sem reivindicação de confirmação externa inédita.',
         'tables':[('Acurácia no teste, por inicialização', ['Rede','Seed','25%','50%','75%','100%'],seeds),
                   ('Diferença pareada de acurácia: GRU menos MLP, em pontos percentuais', ['Observado','Desenhos','Diferença','IC bootstrap 95%'],paired)],
         'figures':fs('12_pareado')},
        {'title':'Categorias, cruzamentos e evidência visual',
         'text':'A matriz de confusão explicita quais categorias recebem as previsões erradas, enquanto o recall por classe mostra quanto do alvo é recuperado. Conservar somente a média global impediria perceber uma categoria em que a MLP seja competitiva ou uma em que mais contexto não resolva a ambiguidade. A classificação refere-se somente às 12 categorias treinadas: sol, maçã, relógio ou bola de basquete são exemplos conceituais de ambiguidade, mas não fazem parte deste experimento e não são usados como demonstração de desempenho. '
                'O fato de a GRU ler uma linha de cada vez não implica separar perfeitamente partes de um objeto. Em uma sequência também existem ruído, traços incompletos e padrões de execução variáveis. Detectamos cruzamentos como interseções próprias entre segmentos de tinta não adjacentes, sem contar simples encontros de extremidades. O grupo com cruzamentos pode ser mais complexo ou conter outras categorias; a análise estratificada é descritiva e não permite afirmar imunidade a sobreposição.',
         'figures':fs('10_confusao','11_categorias','15_cruzamentos')},
        {'title':'Sondas de ordem, pausas e pen-up',
         'text':'O teste mais direto da informação temporal é conservar a imagem e mudar a ordem dos traços inteiros. Invertemos a sequência desses traços sem alterar sua geometria; a união de pixels permanece exatamente igual. Como uma MLP determinística recebe o mesmo vetor, sua saída também deve ser idêntica. A GRU recebe outra sequência e pode alterar a classe ou apenas redistribuir probabilidades. Para separar uma simples mudança de relógio ou do pen-up final, a referência e a sequência reordenada usam intervalos uniformes e fecham igualmente o trecho parcial, forçando o pen-up do último evento antes da permutação. Essa referência modifica conjuntamente relógio e fechamento; não é uma ablação isolada do tempo. '
                'Aplicamos ainda duas sondas aos tokens da GRU: zerar o canal de pen-up e zerar o intervalo transformado. O raster original é preservado, portanto a entrada da MLP não muda nessas duas condições. Ocultar pen-up não apaga a geometria nem os saltos de deslocamento, que ainda podem indicar separações. Isso testa dependência dos canais sob perturbação, sem treinar novamente as redes. Sensibilidade não significa benefício: uma mudança pode corrigir ou introduzir um erro. A acurácia sob essas sondas é separada da avaliação principal e se refere a 240 desenhos de validação balanceados, com 75% dos eventos. '+order_summary,
         'tables':[('Sondas de validação: qualidade e mudança da distribuição', ['Rede','Condição','Acurácia','Top-1 mudou','Média |Δp|'],probes)],
         'figures':fs('13_ordem','14_perturbacoes')},
        {'title':'Tamanho, posição e reconhecimento precoce','level':2,
         'text':'A normalização geométrica explícita remove translação e escala positiva uniforme das duas apresentações quando a extensão permanece acima do piso de uma unidade, salvo arredondamento numérico e discretização. Prefixos degenerados abaixo desse piso são uma exceção. Isso não é uma propriedade automática da recorrência, nem decorre apenas de usar deslocamentos: multiplicar um desenho por um fator também multiplica Δx e Δy se não houver normalização adicional. Rotações, deformações e mudanças na relação entre largura e altura não foram removidas por esse procedimento. '
                'Ambas as redes podem reconhecer antes do fim, pois ambas observam um prefixo que pode já conter evidência suficiente. Uma observação isolada de acerto em 25% não prova que a classe seguirá correta até 100%; o software mostra a evolução das probabilidades e suas possíveis reversões. A tabela de acerto estável verifica somente os quatro checkpoints avaliados, e não todos os pontos intermediários: uma entrada em f exige acerto em f e em cada checkpoint seguinte. Sem sufixo correto significa falha em 100%, podendo haver acertos anteriores; não significa que a rede nunca acertou. Os exemplos de validação são selecionados por critérios registrados, incluindo casos favoráveis a cada rede e casos ambíguos. São ilustrações, sem substituir as médias do teste.',
         'tables':[('Correta em todos os checkpoints observados de f até 100%: seed 42', ['Rede','f=25%','f=50%','f=75%','f=100%','Sem sufixo correto'],stable_rows)],
         'figures':fs('16_exemplos')},
        {'title':'Software interativo e reprodutibilidade',
         'text':'O laboratório principal executa os dois classificadores no navegador, com pesos exportados da seed 42, e permite acompanhar exemplos reais ou desenhar livremente. A interface apresenta as hipóteses top-3, a imagem reduzida vista pela MLP e as probabilidades à medida que os eventos são revelados. O ritmo lento controla somente a reprodução visual; não inventa timestamps de treino. Em desenho livre não há rótulo verdadeiro disponível e, portanto, a interface não pode calcular acurácia individual. '
                'O repositório inclui configurações, partições preparadas, pesos, históricos, métricas e fontes deste relatório. O cache das duas apresentações pode ser reconstruído a partir dos dados preparados compartilhados. A busca utiliza PyTorch em CUDA, com determinismo habilitado, TF32 desligado e parâmetros de ambiente registrados; os diagnósticos e o PDF podem ser gerados em CPU. Reproduzir o protocolo em outro hardware não garante igualdade bit a bit. A exportação deve ser verificada por referências numéricas Python/JavaScript antes da publicação. '
                'A demonstração é uma ferramenta para inspecionar diferenças, não uma prova selecionada para confirmar que uma rede sempre ganha. Uma MLP sobre a sequência concatenada ou uma CNN sobre a imagem seriam controles adicionais pertinentes; não foram confundidos com o comparador efetivamente treinado aqui.'},
    ]
    gallery_section={
        'title':'Galeria progressiva: desenhos e hipóteses, passo a passo',
        'start_on_new_page':True,
        'text':'As pranchas seguintes mostram o desenho sendo revelado em cinco etapas: aproximadamente 10%, 25%, 50%, 75% e 100% dos eventos. Cada coluna contém o prefixo real, sua imagem binária, as três categorias mais prováveis de cada rede e P(alvo), a probabilidade da categoria verdadeira, mesmo quando ela não aparece no top-3. O número de eventos e o tempo observado identificam cada corte; os percentuais pedidos são arredondados para eventos inteiros. A curva inferior resume a evolução da probabilidade do alvo com mais cortes intermediários. '
               'Todos os desenhos dessas pranchas são da validação e todos os valores são recalculados com os pesos efetivos da seed 42, sem receber pontos futuros. Há exemplos das 12 categorias, incluindo contrastes favoráveis a cada rede, com ID e regra de escolha explícitos. Os casos selecionados não substituem as métricas agregadas do teste. A forma cresce por acréscimo de eventos; não ampliamos artificialmente o desenho para sugerir ganho de informação. '
               'O peixe mostrado na captura enviada durante o uso do site foi desenhado livremente pelo autor: eram 27 eventos, 6,4 s observados e um traço. Os valores exibidos, transcritos abaixo e arredondados pela interface, registram um erro concreto da MLP. Como a sequência original desse desenho não foi salva, não inventamos sua evolução nem recalculamos suas probabilidades. O peixe que abre a galeria é outro desenho, identificado no conjunto de validação, escolhido para ilustrar a mesma confusão Peixe/Cachorro; ele não é uma reconstrução da captura.',
        'tables':[('Registro da captura do desenho livre: hipóteses exibidas pelo site',
                   ['Rede','Primeira hipótese','Segunda hipótese','Terceira hipótese'],
                   [['GRU','Peixe (97,5%)','Tubarão (1,8%)','Cachorro (0,7%)'],
                    ['MLP comum','Cachorro (33,9%)','Peixe (17,4%)','Tubarão (14,5%)']])],
        'pages':gallery_pages}
    insert_at=next(i for i,s in enumerate(sections) if s['title']=='Software interativo e reprodutibilidade')
    sections.insert(insert_at,gallery_section)
    forecast=ROOT/'results/drawing_forecast_compare_v1'
    if (forecast/'models/gru/test_metrics.json').exists():
        forecast_metrics={f:read_json(forecast/'models'/f/'test_metrics.json') for f in NAMES}
        forecast_rows=[]
        for k in [5,15,30]:
            for family in NAMES:
                r=forecast_metrics[family]['horizons'][str(k)]
                forecast_rows.append([k,'GRU' if family=='gru' else 'Deep Sets',num(r['ade_pixels'],2),num(r['fde_pixels'],2),num(r['pen_f1'])])
        secondary=[]
        for stem in ['06_examples_2d_xy','06_examples','07_examples','3_examples','03_examples','g_exemplos']:
            path=forecast/'diagnostics'/(stem+'.pdf')
            if path.exists():secondary=[(path,'Exemplos do estudo secundário de previsão. As cores contrastam o futuro real e as hipóteses; sua plausibilidade visual não substitui ADE, FDE ou qualidade dos levantamentos da caneta.')];break
        sections.append({'title':'Estudo secundário: previsão dos próximos eventos',
            'text':'Este estudo responde a outra pergunta: dadas a categoria e uma janela de movimentos já observados, quais serão as posições e os estados da caneta nos próximos k eventos? Aqui o rótulo é uma condição da entrada e k vale 5, 15 ou 30 eventos; não são dias, segundos ou classes. O modelo prevê coordenadas, pen-up/fim e intervalos de tempo. Seu desempenho não pode ser apresentado como acurácia de reconhecimento. '
                   'O comparador desse estudo anterior é Deep Sets: uma MLP por evento seguida de média mascarada, ligada ao mesmo decoder multimodal da GRU. Ele não é a MLP de imagem do experimento principal e não é rebatizado como MLP convencional. A agregação é invariável a permutações dos vetores fixos, mas os atributos e o referencial causal do decoder conservam informação temporal local. O desenho de comparação anterior isola melhor o encoder sob um decoder comum; o principal muda também a apresentação para remover a ordem do raster. '
                   'O ADE mede a distância média em pixels sobre o horizonte e o FDE a distância no último ponto. Uma média de trajetórias visualmente plausíveis pode ainda falhar na estrutura do objeto. As métricas de caneta permanecem limitadas: não há promessa de completamento perfeito. A busca por validação e as três seeds desse estudo estão preservadas nos artefatos, separadas dos novos treinos de reconhecimento.',
            'tables':[('Previsão: métricas médias das três seeds, em tarefa distinta', ['k','Rede','ADE (px)','FDE (px)','F1 pen-up'],forecast_rows)],'figures':secondary})
    sections.append({'title':'Discussão e conclusão',
        'text':conclusion+'Os resultados sustentam uma comparação concreta entre dois sistemas para prefixos de desenhos, sem estabelecer que a recorrência seja necessária ou suficiente em todo cenário. A GRU dispõe de trajetória, pausas e separação temporal; a MLP de imagem depende da forma espacial já visível. As sondas mostram que a informação removida pode mudar a decisão, mas a relação entre sensibilidade e acerto deve ser medida, não presumida. '
               'Entre as limitações estão a seleção de desenhos previamente reconhecidos pelo jogo, as 12 categorias, o raster de baixa resolução, as diferenças de capacidade, o orçamento finito e a mudança conjunta de entrada e arquitetura. Os prefixos por fração não equivalem a um prazo em segundos, os autores não foram particionados de forma independente e o teste já foi observado em estudos anteriores. O conjunto não representa todo desenho livre possível. '
               'Um próximo controle seria comparar GRU e MLP sobre os mesmos eventos concatenados e, separadamente, MLP e CNN sobre a mesma imagem. Esses experimentos responderiam quais ganhos vêm da representação temporal, quais vêm da recorrência e quais vêm da estrutura espacial da rede. O trabalho atual mantém a interpretação restrita aos modelos, dados e protocolos efetivamente avaliados.'})
    references=[('Google Creative Lab — documentação e licença do Quick, Draw!', 'https://github.com/googlecreativelab/quickdraw-dataset'),
                ('Cho et al. (2014) — Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation (GRU)', 'https://arxiv.org/abs/1406.1078'),
                ('Zaheer et al. (2017) — Deep Sets (apenas comparador do estudo secundário)', 'https://arxiv.org/abs/1703.06114'),
                ('Código, configurações e resultados do projeto', 'https://github.com/martinelipedro/projeto4-irn')]
    render(output,abstract,sections,references)
    write_json(output/'report/report_manifest.json',{'task':'recognition_primary_forecast_secondary','figures':sum(len(s.get('figures',[]))+len(s.get('pages',[])) for s in sections),'recognition_figures':len(figures)+len(gallery_pages),'gallery_cases':len(gallery_pages),'gallery_snapshots':5*len(gallery_pages),'sections':len(sections),'models':['GRU','conventional image MLP'],'test_previously_observed':True})
    print('Relatório gerado:',output/'report/relatorio.pdf',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path);args=parser.parse_args();build(args.directory)
