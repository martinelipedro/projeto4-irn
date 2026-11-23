"""Relatório científico exclusivamente de previsão: GRU versus MLP DeepSets."""
import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from ..io import ROOT,read_json,write_json

NAMES={'gru':'GRU','mlp':'MLP DeepSets'}
COLORS={'gru':'#087F8C','mlp':'#D96646'}


def number(value,digits=3):return '—' if value is None else f'{value:.{digits}f}'.replace('.',',')
def integer(value):return f'{int(value):,}'.replace(',','.')
def escape(value):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','_':r'\_','#':r'\#','$':r'\$','{':r'\{','}':r'\}',
             '≥':r'$\geq$','≤':r'$\leq$','−':'-','²':r'$^2$','×':r'$\times$','±':r'$\pm$','Δ':r'$\Delta$','→':r'$\to$'}
    return ''.join(mapping.get(c,c) for c in str(value))


def render(output,sections,abstract,references):
    folder=output/'report';folder.mkdir(parents=True,exist_ok=True)
    title='Previsão temporal de traços: GRU versus MLP DeepSets'
    tex=r'''\documentclass[11pt,a4paper]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}\usepackage[brazilian]{babel}
\usepackage[margin=2cm]{geometry}\usepackage{graphicx,booktabs,longtable,amsmath,amssymb,hyperref,float,xcolor,fancyhdr}
\definecolor{ink}{HTML}{203448}\definecolor{accent}{HTML}{087F8C}
\hypersetup{colorlinks=true,urlcolor=accent,linkcolor=ink,pdftitle={Previsão temporal de traços: GRU versus MLP DeepSets},pdfauthor={Pedro Martineli}}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\setlength{\headheight}{14pt}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small Projeto 4 — Previsão de séries temporais}\fancyhead[R]{\small Quick, Draw!}\fancyfoot[C]{\thepage}
\begin{document}\begin{center}{\Large\bfseries\color{ink} '''+escape(title)+r'''}\\[5pt]
Introdução às Redes Neurais Artificiais\\Pedro Martineli (RA: 193474)
\end{center}\textbf{Resumo.} '''+escape(abstract)+'\n'
    md='# '+title+'\n\n'+abstract+'\n\n'
    for section in sections:
        level=section.get('level',1);heading=section['title'];text=section['text']
        tex+='\\'+('section' if level==1 else 'subsection')+'{'+escape(heading)+'}\n'+escape(text)+'\n'
        md+='#'*(level+1)+' '+heading+'\n\n'+text+'\n\n'
        for equation in section.get('equations',[]):tex+='\\[\n'+equation+'\n\\]\n';md+='$$\n'+equation+'\n$$\n\n'
        for caption,head,rows in section.get('tables',[]):
            md+='**'+caption+'**\n\n| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)+'\n\n'
            wide=len(head)==3 and any(len(str(cell))>40 for row in rows for cell in row)
            columns='p{.22\\linewidth}p{.34\\linewidth}p{.34\\linewidth}' if wide else 'l'+'r'*(len(head)-1)
            tex+='\\begingroup\\small\\setlength{\\tabcolsep}{4pt}\\begin{longtable}{'+columns+'}\n\\caption{'+escape(caption)+r'}\\'+'\n\\toprule\n'
            header=' & '.join(map(escape,head))+r' \\ '+'\n'
            tex+=header+'\\midrule\\endfirsthead\n\\toprule\n'+header+'\\midrule\\endhead\n\\bottomrule\\endfoot\n'
            tex+='\n'.join(' & '.join(map(escape,row))+r' \\' for row in rows)+'\n\\end{longtable}\\endgroup\n'
        for path,caption in section.get('figures',[]):
            path=Path(path)
            if not path.exists():raise FileNotFoundError(path)
            target=output/'figures'/path.name
            if target.resolve()!=path.resolve():shutil.copy2(path,target)
            relative='../figures/'+target.name
            tex+='\\begin{figure}[H]\\centering\\includegraphics[width=\\linewidth,height=.58\\textheight,keepaspectratio]{'+relative+'}\\caption{'+escape(caption)+'}\\end{figure}\n'
            png=path.with_suffix('.png')
            if png.exists():
                target_png=target.with_suffix('.png')
                if target_png.resolve()!=png.resolve():shutil.copy2(png,target_png)
                relative='../figures/'+target_png.name
            md+='!['+caption+']('+relative+')\n\n'+caption+'\n\n'
    tex+='\\section*{Referências}\\begin{enumerate}\n';md+='## Referências\n\n'
    for label,url in references:tex+='\\item \\href{'+url+'}{'+escape(label)+'}\n';md+=f'- [{label}]({url})\n'
    tex+='\\end{enumerate}\\small Dados: Google Quick, Draw!, CC BY 4.0. Código e redação elaborados com auxílio de IA; revisar antes da entrega acadêmica.\\end{document}'
    (folder/'relatorio.tex').write_text(tex);(folder/'relatorio.md').write_text(md)
    for _ in range(2):
        run=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','relatorio.tex'],cwd=folder,capture_output=True,text=True)
        if run.returncode:
            (folder/'compile_error.txt').write_text(run.stdout+run.stderr);raise RuntimeError(str(folder/'compile_error.txt'))
    (folder/'compile_error.txt').unlink(missing_ok=True)


def history(path):
    with Path(path).open() as f:rows=list(csv.DictReader(f))
    return {key:np.array([float(row[key]) for row in rows]) for key in rows[0]}


def explanatory_figures(output):
    folder=output/'figures';folder.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    def save(fig,name):fig.savefig(folder/(name+'.pdf'),bbox_inches='tight');fig.savefig(folder/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    boxes=[(.02,.42,.22,.28,'Eventos até t\n6 atributos + categoria\nw ≤64'),(.31,.65,.26,.22,'GRU\nestado atualizado em sequência'),(.31,.15,.26,.22,'MLP DeepSets\nMLP por evento + média'),(.67,.42,.30,.28,'Mesmo decoder\n64 modos por categoria\nposição + caneta + tempo')]
    for x,y,w,h,label in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.01',facecolor='#EDF4F6',edgecolor='#A8C4CD'))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=10)
    for start,end in [((.24,.56),(.31,.76)),((.24,.56),(.31,.26)),((.57,.76),(.67,.56)),((.57,.26),(.67,.56))]:
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','lw':1.8,'color':'#087F8C'})
    ax.text(.5,.02,'Comparamos o encoder; entradas, condição, alvos e decoder permanecem compartilhados.',ha='center',fontsize=10)
    save(fig,'a_arquiteturas')
    meta=read_json(output/'data/metadata.json');scale=meta['delta_scale']
    raw=np.array([[100,50,0,0],[108,54,1,.2]],float)
    token=[8/scale,4/scale,1,np.log1p(.2/.1),8/256,4/256]
    fig,axes=plt.subplots(1,2,figsize=(11,3.6),layout='constrained')
    axes[0].plot(raw[:,0],raw[:,1],'o-',color='#203448',lw=2);axes[0].invert_yaxis()
    for x,y,p,t in raw:axes[0].annotate(f'({x:g}, {y:g}) · {t:g} s',(x,y),xytext=(0,12),textcoords='offset points',ha='center')
    axes[0].set(xlim=(95,113),ylim=(60,44),title='Exemplo didático: evento2 encerra o traço',xlabel='X em pixels',ylabel='Y em pixels');axes[0].grid(alpha=.15)
    axes[1].axis('off');table=axes[1].table(cellText=[[label,number(v,4)] for label,v in zip(['dx / sΔ','dy / sΔ','pen-up','log(1 + dt / 0,1 s)','x relativo / 256','y relativo / 256'],token)],colLabels=['Atributo','Valor no evento2'],loc='center',cellLoc='left');table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,1.65)
    axes[1].set_title('Um evento é um vetor de seis valores')
    save(fig,'b_entrada_eventos')


def study_figures(output,selection):
    folder=output/'figures';paths=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    for ax,model in zip(axes,['gru','mlp']):
        family=selection['families'][model]
        rows=sorted(family['screening'],key=lambda r:r['score'])
        labels=[str(r['spec']['hidden'])+' / '+number(r['spec']['lr'],4) for r in rows];values=[r['score'] for r in rows]
        ax.barh(range(len(rows)),values,color=COLORS[model]);ax.invert_yaxis();ax.set(yticks=range(len(rows)),yticklabels=labels,title=NAMES[model]+' · triagem seed42',xlabel='J de validação · menor é melhor');ax.grid(axis='x',alpha=.15)
    for ext in ['png','pdf']:fig.savefig(folder/('c_busca.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,6.4),layout='constrained')
    for column,model in enumerate(['gru','mlp']):
        family=selection['families'][model];spec=family['best']['spec']
        for seed in [42,43,44]:
            h=history(output/'models'/model/'runs'/f"{spec['id']}_seed{seed}"/'history.csv')
            axes[0,column].plot(h['epoch'],h['train_objective'],label=f'Seed {seed}')
            line,=axes[1,column].plot(h['epoch'],h['val_rollout_objective'],label=f'Seed {seed}')
            i=int(np.argmin(h['val_rollout_objective']));axes[1,column].scatter(h['epoch'][i],h['val_rollout_objective'][i],color=line.get_color(),marker='*',s=90)
        axes[0,column].set(title=NAMES[model]+' · treino',ylabel='Perda composta');axes[1,column].set(title=NAMES[model]+' · validação',ylabel='ADE/256 + penalidade de fim')
    for ax in axes.flat:ax.set_xlabel('Época');ax.legend(fontsize=8);ax.grid(alpha=.15)
    for ext in ['png','pdf']:fig.savefig(folder/('d_aprendizagem.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)


def build(output=None):
    output=Path(output or ROOT/'results/drawing_forecast_compare_v1')
    config=read_json(output/'config.json');meta=read_json(output/'data/metadata.json')
    selections={m:read_json(output/'models'/m/'selection.json') for m in NAMES}
    metrics={m:read_json(output/'models'/m/'test_metrics.json') for m in NAMES}
    findings=read_json(output/'diagnostics/findings.json')
    explanatory_figures(output);study_figures(output,{'families':selections})
    def metric(model,k,key):return metrics[model]['horizons'][str(k)][key]
    def figure(stem,fallback):
        p=output/'diagnostics'/f'{stem}.pdf'
        return [(p,findings.get('captions',{}).get(stem,fallback))] if p.exists() else []
    search_rows=[];final_rows=[];network_rows=[]
    for model,sel in selections.items():
        for t in sel['screening']:
            search_rows.append([NAMES[model],t['spec']['hidden'],number(t['spec']['lr'],4),t['checkpoint_epoch'],number(t['pen_threshold'],2),number(t['score'])])
        for s in sel['finalists']:
            best=min([r for r in sel['finalist_scores'] if r['spec']==s],key=lambda r:r['score'])
            scores=best['seed_scores'];final_rows.append([NAMES[model],s['hidden'],number(s['lr'],4),number(best['pen_threshold'],2),number(scores[0]),number(scores[1]),number(best['score'])])
        best=sel['best'];run=read_json(output/'models'/model/best['directory']/'summary.json')
        network_rows.append([NAMES[model],best['spec']['hidden'],number(best['spec']['lr'],4),integer(run['parameters']),best['checkpoint_epoch'],number(run['seconds'],1)])
    results=[];pen=[];seeds=[]
    for k in [5,15,30]:
        for model in NAMES:
            h=metrics[model]['horizons'][str(k)]
            results.append([k,NAMES[model],integer(h['n_prefixes']),number(h['ade_pixels'],1),number(h['fde_pixels'],1),number(h['rmse_pixels'],1),number(h.get('ade_std_between_seeds'),2)])
            pen.append([k,NAMES[model],number(h['pen_f1']),number(100*h['premature_end'],1),number(h['clock_mae_seconds']),number(h['clock_median_mae_seconds'])])
    for model in NAMES:
        for r in metrics[model]['seeds']:seeds.append([NAMES[model],r['seed'],r['checkpoint_epoch'],*[number(r['horizons'][str(k)]['ade_pixels'],1) for k in [5,15,30]]])
    paired=[]
    for r in findings['paired_overall']:
        interval=r['delta_ci95'];paired.append([r['k'],integer(r['n_drawings']),number(r['delta_ade'],2),f'[{number(interval[0],2)}; {number(interval[1],2)}]',number(100*r['gru_win_fraction'],1)])
    conditions=[]
    for r in findings['paired_groups']:
        if r['k']!=15:continue
        lo,hi=r['delta_ci95'];conditions.append([{'crossings':'Cruzamentos','pen_up':'Pen-up'}.get(r['descriptor'],r['descriptor']),r['group'],integer(r['n_prefixes']),number(r['delta_ade'],2),f'[{number(lo,2)}; {number(hi,2)}]'])
    sensitivity=[]
    for r in findings['sensitivities']:
        sensitivity.append([NAMES[r['model']],{'baseline':'Original','translate_120_-80':'Translação','scale_075':'Escala 0,75','scale_125':'Escala 1,25','shuffle_interior_fixed_suffix':'Ordem permutada','mask_pen_fixed_frame':'Sem pen-up','mask_time':'Sem tempo'}.get(r['scenario'],r['scenario']),integer(r['n_prefixes']),number(r['ade'],1),number(r.get('ade_rescaled'),1),number(r.get('ratio_vs_baseline'),2)])
    windows=[]
    for r in findings.get('context_windows',[]):windows.append([NAMES[r['model']],r['window'],integer(r['n_prefixes']),number(r['ade'],1),number(r['fde'],1)])
    probes={(r['model'],r['scenario']):r for r in findings['sensitivities']}
    order_gru=probes[('gru','shuffle_interior_fixed_suffix')];order_mlp=probes[('mlp','shuffle_interior_fixed_suffix')]
    time_gru=probes[('gru','mask_time')];time_mlp=probes[('mlp','mask_time')]
    translation=findings['controls']['translation'];permutation=findings['controls']['permutation']
    context={(r['model'],r['window']):r for r in findings.get('context_windows',[])}
    context_message=''
    if all((model,w) in context for model in NAMES for w in [8,64]):
        context_message=(f'Na análise com origem64, o ADE da GRU passou de {number(context[("gru",8)]["ade"],1)} para {number(context[("gru",64)]["ade"],1)} pixels ao ampliar a janela de8para64eventos; '
                         f'o MLP passou de {number(context[("mlp",8)]["ade"],1)} para {number(context[("mlp",64)]["ade"],1)}. ')
    hypotheses=[
        ['Formas ambíguas','Prever continuações, categoria fornecida','Não mede reconhecimento de sol ou maçã'],
        ['Traços sobrepostos','Erro condicionado a cruzamentos','Não demonstra imunidade nem compara CNN'],
        ['Mais observações','Prefixos e janela com origem fixa','Alvos/coortes e distribuição podem mudar'],
        ['Tamanho e posição','Translação e fatores de escala','Δ remove translação, não magnitude'],
        ['Caneta levantada','Condicionais e retirada do canal','Retirada sem retreino é sensibilidade OOD']]
    gru30=metric('gru',30,'ade_pixels');mlp30=metric('mlp',30,'ade_pixels')
    gain=100*(1-gru30/mlp30)
    better=[str(k) for k in [5,15,30] if metric('gru',k,'ade_pixels')<metric('mlp',k,'ade_pixels')]
    abstract=(f'Este trabalho prevê os próximos eventos da caneta em desenhos do Quick, Draw!, usando uma rede recorrente GRU como modelo principal e um MLP DeepSets como comparação sem recorrência. '
        'A avaliação mantém dados, categoria fornecida, banco de trajetórias, decoder, perdas e orçamento comuns. A busca utiliza somente a validação, com treinamento do zero e três réplicas da configuração escolhida. '
        f'Em 30 eventos futuros, o ADE médio foi {number(gru30,1)} pixels para a GRU e {number(mlp30,1)} para o MLP. '
        'O relatório analisa horizontes, quantidade de contexto, cruzamentos, estado da caneta, tempo, ordem e transformações geométricas, acompanhados de exemplos visuais. As conclusões distinguem propriedades da representação de evidências específicas da recorrência. O teste é uma partição já explorada em estudos anteriores, não uma validação externa inédita.')
    sections=[
      {'title':'Problema de previsão e planejamento experimental','text':'O objetivo é receber uma janela de eventos observados até t e prever as posições da caneta, os estados de desenho e os intervalos dos próximos k eventos. O trabalho concentra-se exclusivamente em previsão de séries temporais. A categoria é uma condição fornecida, e não uma saída de reconhecimento. Usamos uma GRU para representar a evolução do estado e um MLP DeepSets para representar o conjunto dos eventos, com o mesmo decoder nos dois casos. A comparação procura identificar quando a ordem sequencial contribui para a continuação, sem pressupor superioridade universal da RNN.'},
      {'title':'Busca de hiperparâmetros e escolha pela validação','level':2,'text':'Cada família compara larguras 128/192 e taxas iniciais 0,001/0,0005. Todos os modelos foram construídos do zero: nenhuma rede é inicializada com os pesos de outra. O banco de 64 modos por categoria é ajustado apenas no treino e compartilhado. Os quatro treinos GRU da triagem, já executados com este protocolo, são reutilizados como execuções completas com hashes e proveniência; isso não constitui transferência para inicializar o MLP. A triagem usa seed 42. Dois finalistas por família são treinados novamente com seed 43; arquitetura e limiar de pen-up comuns minimizam a média de validação nas seeds 42/43. A réplica 44 é posterior à decisão. Nenhum resultado de teste escolhe uma configuração desta comparação.',
       'tables':[('Triagem seed42 por família; menor J é melhor.',['Rede','Unidades','LR inicial','Época','Limiar','J val.'],search_rows),('Decisão entre finalistas, com limiar compartilhado entre seeds.',['Rede','Unidades','LR inicial','Limiar','J42','J43','Média'],final_rows)],
       'figures':[(output/'figures/c_busca.pdf','Busca na validação. As duas famílias utilizam o mesmo critério, alvos e orçamento.')]},
      {'title':'Critério, taxas e orçamento de treinamento','level':2,'text':'O orçamento comum é de 20 épocas, lote 128, AdamW, weight decay 0,00001, clipping da norma do gradiente em 1 e taxa cossenoidal até 10% da inicial. A augmentação isotrópica de escala 0,8–1,2 ocorre somente no treino e é a mesma nas duas famílias. O checkpoint de cada execução minimiza a média de ADE/256 + 0,1 vezes a frequência de fim precoce nos horizontes 5/15/30, nos primeiros 100 desenhos de validação por classe. A seleção externa usa toda a validação, acrescenta a penalidade de F1 de pen-up e testa limiares 0,2/0,35/0,5. A perda supervisionada combina CE do modo, 8×Huber de posição, 0,4×Huber de deslocamentos, CE da caneta, 0,15×Huber do relógio e 0,2×suavidade das correções. Nenhum ponto real futuro é fornecido na inferência da validação.',
       'equations':[r'J=\frac{1}{3}\sum_{k\in\{5,15,30\}}\left[\frac{\mathrm{ADE}_k}{256}+0{,}1\,E_k+0{,}02(1-F1_{\mathrm{pen},k})\right].']},
      {'title':'Curvas de aprendizagem e capacidade','level':2,'text':'As curvas distinguem a perda composta do treino e o critério de trajetória usado na validação. Como possuem definições e unidades diferentes, seus valores não devem ser subtraídos como um gap direto. As estrelas indicam o melhor checkpoint de cada seed. O número de parâmetros é informado porque igual largura oculta não implica igual capacidade entre GRU e MLP. O tempo mostrado é o observado na execução seed42 da rede escolhida, na RTX 4050 de 6 GB em float32, com determinismo habilitado; não é um benchmark controlado de energia ou latência. A cabeça de correções começa zerada nos dois modelos; as demais camadas utilizam inicialização por seed.',
       'tables':[('Configurações finais e custo observado; tempo em segundos.',['Rede','Unidades','LR inicial','Parâmetros','Época42','Tempo42'],network_rows)],
       'figures':[(output/'figures/d_aprendizagem.pdf','Aprendizagem das três réplicas de cada família. Checkpoints são escolhidos pela validação.')]},
      {'title':'Dados, representação temporal e saídas','text':'Utilizamos 42 mil desenhos de 12 categorias: gato, peixe, flor, casa, árvore, guarda-chuva, cachorro, tubarão, castelo, cacto, carro e bicicleta. São 36 mil de treino, 3 mil de validação e 3 mil de teste, balanceados por categoria e com desenhos inteiros em uma única partição. O recorte inclui somente desenhos reconhecidos pelo jogo original. Os mesmos IDs e geometrias ficam congelados para ambas as famílias. As partições já foram usadas anteriormente; a comparação é interna sobre esse teste. Não há separação garantida por autor. Os arquivos preparados, seus hashes e os manifests permitem reconstruir o recorte sem depender de outros experimentos.'},
      {'title':'O que entra na rede','level':2,'text':f'O dado bruto contém traços com X, Y e tempos em milissegundos. Derivamos o estado de caneta a partir dos limites dos traços e convertemos o relógio para segundos relativos ao primeiro ponto. A amostragem causal mantém pontos após 14 pixels de deslocamento já observado, preservando fronteiras e tempos. Filtramos 12–180 eventos, até 4 mil pontos brutos e duração positiva até 60 s. Cada evento é um vetor de seis valores. O desvio dos deslocamentos, {number(meta["delta_scale"],2)} pixels, é calculado só no treino; posições usam escala fixa 256 e tempo 0,1 s. A categoria vira um embedding de 16 dimensões concatenado a cada evento: o encoder recebe 22 atributos por passo. O contexto contém no máximo 64 eventos; comprimentos e máscaras excluem preenchimento.',
       'equations':[r'\mathbf u_i=\left[\frac{\Delta x_i}{s_\Delta},\frac{\Delta y_i}{s_\Delta},p_i,\log\left(1+\frac{\Delta\tau_i}{0{,}1\,\mathrm{s}}\right),\frac{x_i-x_0}{256},\frac{y_i-y_0}{256}\right].'],
       'tables':[('Entrada e alvo na tarefa de previsão.',['Elemento','Formato','Conteúdo'],[['Prefixo','B × w × 6','Eventos observados, w ≤64'],['Categoria','B índices','Condição conhecida, 12 categorias'],['Entrada ao encoder','B × w × 22','Evento e embedding'],['Alvos futuros','B × k eventos','X/Y, caneta e tempo reais'],['Saída interpretável','B × k × 4','X/Y, estado e intervalo previstos']])],
       'figures':[(output/'figures/b_entrada_eventos.pdf','Exemplo didático de codificação, não uma previsão selecionada pelo acerto. O primeiro movimento e o primeiro intervalo são zero.')]},
      {'title':'O que significa prever t+k','level':2,'text':'t é um índice de evento, não um instante uniformemente espaçado em segundos. Se 32 eventos foram observados, k=5 prevê os eventos 33–37, enquanto k=30 prevê 33–62. Os intervalos entre esses eventos também são previstos. A saída interna do decoder tem seis valores por evento: duas coordenadas, três escores de estado e um valor de relógio. A saída decodificada é X, Y, estado e intervalo; o código também conserva deslocamentos e uma máscara de encerramento. Depois de encerrar, a posição é mantida, a caneta fica levantada e o relógio não avança. Encerrar enquanto ainda há futuro real é contado como erro, não como remoção da amostra.',
       'equations':[r'\widehat{\mathbf S}_{t+1:t+k}=f_\theta(\mathbf u_{t-w+1:t},c),\quad k\in\{5,15,30\},\quad\widehat\tau_{t+j}=\tau_t+\sum_{\ell=1}^j\widehat{\Delta\tau}_{t+\ell}.']},
      {'title':'Como isolamos a contribuição da recorrência','text':'A GRU atualiza um estado h a cada evento. O DeepSets aplica um MLP Tanh 22→H→H a cada evento, calcula uma média mascarada, concatena a contagem observada w/64 e usa uma camada de fusão para produzir um estado do mesmo tamanho. Esse encoder não possui GRU, LSTM ou outra recorrência. A média é invariável a permutações dos vetores, mas os próprios vetores já contêm movimento e tempo locais. Além disso, o decoder comum usa o último ponto e a direção dos últimos movimentos observados. Portanto, não se deve chamar o MLP inteiro de totalmente cego à ordem ou invariável a qualquer permutação da entrada. O ramo visual CNN de estudos anteriores não integra esta comparação principal.',
       'equations':[r'h_i^{\mathrm{GRU}}=\operatorname{GRU}([\mathbf u_i,e(c)],h_{i-1}),',r'h^{\mathrm{MLP}}=\tanh\!\left(W_\rho\left[\frac{\sum_i m_i\phi([\mathbf u_i,e(c)])}{\sum_i m_i},\frac{\sum_i m_i}{64}\right]+b_\rho\right).'],
       'figures':[(output/'figures/a_arquiteturas.pdf','Comparação dos encoders, com condição e decoder compartilhados. A RNN é o modelo principal; o MLP é um controle arquitetural concreto, não qualquer rede possível.')]},
      {'title':'Decoder compartilhado e hipóteses de trajetória','level':2,'text':'O banco contém 64 padrões de 30 eventos por categoria, aprendido com três janelas de cada desenho de treino. A seleção supervisionada do modo usa o futuro somente como alvo da perda durante o treino. Na inferência, o modo vem apenas do passado e da categoria fornecida. A rede prevê correções limitadas por tanh e estados de caneta e relógio. O referencial usa a última posição e a direção dos últimos três segmentos válidos de tinta, sem coordenadas futuras. Não há busca por desenhos parecidos na inferência. Compartilhar esse mecanismo controla uma parte importante da representação, mas também limita a diversidade das duas famílias. A inspiração em modos vem de MultiPath; a implementação não reproduz o artigo nem o Sketch-RNN.'},
      {'title':'Hipóteses sobre o que a RNN pode ajudar','text':'Ideias intuitivas sobre memória precisam ser transformadas em perguntas mensuráveis. As redes usadas aqui são GRUs, não LSTMs. Frases sobre reconhecer maçã ou sol, separar perfeitamente partes de objetos ou ser imune a rabiscos não são resultados deste estudo. As categorias citadas nesses exemplos nem fazem parte do recorte de 12 classes. O que medimos é a continuação condicionada a uma categoria fornecida. As hipóteses abaixo orientam gráficos e controles, sem converter vantagens possíveis em garantias.',
       'tables':[('Da intuição ao experimento de previsão.',['Ideia','Operação no estudo','Limite da conclusão'],hypotheses)]},
      {'title':'Resultados de teste por horizonte','text':'Cada previsão recebe apenas o prefixo real, e toda a trajetória futura é emitida sem receber novos pontos reais. Usam-se cortes de 16/32 eventos no resultado principal; o recorte inicial exclui prefixos com cinco ou menos eventos restantes e cada k exige ao menos k eventos futuros. As populações de horizontes diferentes não são idênticas. ADE é a distância euclidiana média ao longo dos k eventos, e FDE é a distância no último evento. O RMSE é calculado por janela nas k×2 coordenadas e depois promediado. A tabela usa médias de três seeds, sem ensemble. Todas as famílias e referências simples usam os mesmos alvos em cada comparação.',
       'equations':[r'\mathrm{ADE}_k=\frac1k\sum_{j=1}^k\|\widehat{\mathbf p}_{t+j}-\mathbf p_{t+j}\|_2,\qquad\mathrm{FDE}_k=\|\widehat{\mathbf p}_{t+k}-\mathbf p_{t+k}\|_2.'],
       'tables':[('Médias de teste entre seeds42/43/44; erros em pixels.',['k','Rede','Prefixos','ADE','FDE','RMSE','DP ADE'],results)],
       'figures':figure('01_horizons_comparison','Comparação das redes e referências simples por horizonte.')},
      {'title':'Réplicas, diferenças pareadas e caudas de erro','level':2,'text':f'A GRU obteve ADE menor que o MLP nos horizontes {", ".join(better) if better else "nenhum dos três"} eventos. Em 30 eventos, a redução relativa de ADE diante do MLP foi {number(gain,1)}%; valores negativos indicam vantagem do comparador. A tabela de réplicas mostra a sensibilidade ao treinamento. Os diagnósticos de caudas e condicionais usam a seed42, definida antes de consultar esses resultados. Os intervalos pareados reamostram desenhos inteiros, preservando juntos seus prefixos. Diferença negativa favorece a GRU. São intervalos condicionais aos pesos, e não intervalos de previsão para um novo traço.',
       'tables':[('Resultados individuais; checkpoint por validação.',['Rede','Seed','Época','ADE5','ADE15','ADE30'],seeds),('Diferença ADE(GRU) − ADE(MLP), seed42; intervalo por desenho.',['k','Desenhos','Diferença','IC95','GRU vence (%)'],paired)],
       'figures':figure('05_paired_errors','Distribuição dos erros e comparação pareada dos mesmos desenhos.')},
      {'title':'Quantidade de contexto e previsão progressiva','text':'A evolução com prefixos de 8/16/32/64 eventos é uma análise de previsão progressiva, não de reconhecimento da categoria. Ao mudar o instante de origem, também se mudam os eventos futuros a prever; a figura informa as coortes disponíveis. Por isso, uma curva melhor não prova isoladamente que mais passado foi a causa. A análise complementar fixa origem 64 e futuro 15, variando somente a janela disponível. Nessa análise, janelas internas diferem dos prefixos de início vistos no treino e são tratadas como sensibilidade fora da distribuição de treinamento, sem retreino.',
       'tables':[('Sensibilidade à janela com origem 64 e os mesmos alvos15.',['Rede','Janela','Prefixos','ADE','FDE'],windows)] if windows else [],
       'figures':figure('02_prefix_cohorts','Efeito do prefixo e da janela disponível, com contagens e alvos explicitados.')},
      {'title':'Cruzamentos, sobreposição e estados de caneta','text':'Cruzamentos do prefixo e frequência de pen-up são descritores calculados antes do futuro, comuns às duas redes. A comparação condicionada ajuda a verificar se certos padrões são mais difíceis ou favorecem um encoder. Ela não identifica causalidade: complexidade, tamanho e categoria também variam entre grupos. O comparador é um MLP de eventos, não uma CNN de imagem; não se avaliou diretamente a alegação de que a RNN seria imune ao borrão de uma imagem raster. O estado de caneta fornece separações observadas, mas não garante que a rede saiba qual segmento é olho, rosto ou contorno.',
       'tables':[('Diferenças condicionadas em k15, seed42; valores negativos favorecem a GRU.',['Descritor','Grupo','Prefixos','Diferença','IC95'],conditions)],
       'figures':figure('03_geometry_groups','Erros condicionados à geometria e aos levantamentos da caneta.')},
      {'title':'Ordem, posição, escala e pausas: controles de sensibilidade','text':'A translação das coordenadas desaparece na codificação por deslocamentos e posições relativas; esse controle deve beneficiar ambos os modelos, e não ser atribuído à recorrência. Já Δx/Δy preservam magnitude: uma casa maior gera deslocamentos maiores. Portanto, não há independência automática de escala. Avaliamos fatores 0,75/1,25 e ADE dividido pelo fator para comparar a escala do erro. A permutação mantém o primeiro evento e um sufixo suficiente para preservar os últimos três segmentos válidos e o referencial do decoder. Permutar vetores já codificados pode produzir uma sequência fisicamente inconsistente; o teste mede sensibilidade do encoder, não uma intervenção causal válida no desenho. Retirar canais de caneta ou relógio sem retreino também é uma perturbação fora da distribuição. No ensaio de pen-up, mantemos explicitamente o quadro original do decoder para não misturar mudança de representação geométrica com a retirada do canal.',
       'tables':[('Sensibilidades em k15 na subamostra determinística; não são novos modelos retreinados.',['Rede','Cenário','n','ADE','ADE/escala','Razão/base'],sensitivity)],
       'figures':figure('04_sensitivities','Controles geométricos e perturbações de ordem e canais.')+figure('07_translation_control','Controle de translação; propriedade da representação compartilhada.')},
      {'title':'O que os controles efetivamente mostraram','level':2,'text':f'Na subamostra de {integer(order_gru["n_prefixes"])} prefixos, mudar a ordem interna aumentou o ADE da GRU em {number(100*(order_gru["ratio_vs_baseline"]-1),1)}%, enquanto o MLP permaneceu numericamente igual (razão {number(order_mlp["ratio_vs_baseline"],4)}). O referencial do decoder não mudou (diferença máxima {number(permutation["max_frame_delta"],6)}). A GRU trocou o modo escolhido em {number(100*permutation["gru_mode_changed_fraction"],1)}% dos prefixos. Isso confirma sensibilidade da representação recorrente à ordenação, mas não torna a permutação uma trajetória humana válida. Na translação, a maior alteração de previsão foi {number(max(translation["gru_max_prediction_delta_pixels"],translation["mlp_max_prediction_delta_pixels"]),6)} pixel; a estabilidade resulta do pré-processamento comum. Zerar o relógio aumentou o ADE em {number(100*(time_gru["ratio_vs_baseline"]-1),1)}% na GRU e {number(100*(time_mlp["ratio_vs_baseline"]-1),1)}% no MLP: o canal temporal ajuda as duas arquiteturas. Os fatores de escala não produziram invariância exata. '+context_message+'Os ensaios com janelas internas são fora da distribuição de início de desenho; são sensibilidades, não garantias de robustez.'},
      {'title':'Tempo e separação de traços','text':'F1 de pen-up complementa os erros espaciais porque a maioria dos eventos mantém a caneta no papel. Uma linha próxima em pixels pode ligar regiões que deveriam estar separadas. O modelo estima intervalos positivos enquanto ativo e zero após encerramento. Comparamos o MAE do intervalo à referência que repete a mediana positiva do prefixo. A regra absorvente de encerramento mantém a posição para que terminar antes da hora continue contando como erro. Melhorar ADE não implica melhorar relógio ou F1; esses aspectos são reportados separadamente. Os valores baixos de F1 mostram que as redes ainda erram separações de traços, contrariando a hipótese de compreensão perfeita dos componentes de um objeto.',
       'tables':[('Caneta e relógio; médias de três seeds. MAE em segundos por intervalo.',['k','Rede','F1 pen-up','Fim precoce (%)','MAE dt','Mediana dt'],pen)],
       'figures':figure('09_time_pen_quality','Qualidade espacial, do estado de caneta e do relógio são dimensões distintas.')},
      {'title':'Casos visuais e diferenças entre categorias','text':'Os exemplos da demonstração são escolhidos na validação por regras documentadas: categorias, disponibilidade de futuro e casos em que cada rede apresenta menor erro. Esse critério produz exemplos explicativos, e não uma amostra aleatória para estimar vantagem média. O relatório inclui vitórias do MLP e falhas da GRU. O futuro real serve à visualização e às métricas, sem entrar na inferência. As curvas X/Y e o desenho 2D representam a mesma previsão; no painel temporal mostramos parte do passado para legibilidade, embora o modelo receba todo o prefixo indicado.',
       'figures':figure('06_examples_2d_xy','Casos de validação com passado, futuro observado e previsões dos dois modelos.')+figure('08_category_comparison','Heterogeneidade do erro por categoria; escala e complexidade também influenciam pixels.')},
      {'title':'Software de demonstração e reprodução','text':'O site permite escolher uma categoria fornecida, revelar o prefixo de um exemplo real, selecionar horizonte, comparar a GRU com o MLP, avançar pontos lentamente e exibir o futuro real para conferência. A categoria é informada explicitamente: não há reconhecedor embutido no experimento. O usuário também pode desenhar e solicitar previsões locais, sem enviar o desenho a um servidor. O ritmo da reprodução serve à apresentação e não altera timestamps. Controles de ordem, escala e posição devem ser lidos como experimentos de sensibilidade, não como garantias de robustez. Os pesos executados no navegador são confrontados numericamente com PyTorch. O repositório projeto4-irn contém código, dados preparados com proveniência, configs, históricos, métricas, figuras e este único relatório. A publicação usa GitHub Pages e não precisa de backend de treinamento.'},
      {'title':'Limitações e conclusão','text':'A comparação testa uma GRU concreta contra um MLP DeepSets concreto com decoder comum. Não demonstra vantagem sobre qualquer rede, nem mede reconhecimento de objetos. O banco de modos e a categoria fornecida restringem as continuações; o mesmo passado pode admitir vários futuros humanos válidos. A busca tem orçamento finito, duas seeds na decisão e três na avaliação; não assegura arquitetura ótima. O teste previamente explorado e a ausência de separação por autor limitam validade externa. Os controles diferenciam benefícios da representação, da recorrência e do decoder. A conclusão deve considerar em conjunto erro espacial, caneta, tempo, grupos de desenhos e exemplos em que cada rede falha, sem transformar memória recorrente em uma promessa de perfeição.'}
    ]
    references=[('Google Quick, Draw! Dataset — vetores, timestamps e licença.','https://github.com/googlecreativelab/quickdraw-dataset'),
      ('Cho et al. (2014) — unidades recorrentes com portas.','https://arxiv.org/abs/1406.1078'),
      ('Zaheer et al. (2017) — Deep Sets e agregação invariante a permutações.','https://arxiv.org/abs/1703.06114'),
      ('Chai et al. (2020) — MultiPath, inspiração para modos de trajetória.','https://proceedings.mlr.press/v100/chai20a.html'),
      ('Ha e Eck (2017) — A Neural Representation of Sketch Drawings.','https://arxiv.org/abs/1704.03477')]
    render(output,sections,abstract,references)
    write_json(output/'report_scope.json',{'task':'forecasting_only','models':['gru','mlp'],'test_previously_observed':True,
        'conditional_category':True,'hypotheses_are_evaluated_not_assumed':True,'narrative':'Não mede reconhecimento; GRU e DeepSets compartilham representação e decoder.'})
    print('Relatório de previsão:',output/'report/relatorio.pdf',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'results/drawing_forecast_compare_v1')
    build(parser.parse_args().directory)
