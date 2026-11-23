"""Pranchas de desenhos parciais, com inferência real dos dois classificadores."""
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..io import ROOT, read_json, write_json
from .data import represent
from .export import load_models, predict


FRACTIONS = [.10, .25, .50, .75, 1.]
COLORS = {'gru': '#087F8C', 'mlp': '#D96646'}
INK = '#203448'
ADDED = '#D96646'


def percentage(value):
    return f'{100*value:.1f}%'.replace('.', ',')


def plot_prefix(ax, points, cut, previous):
    """Enquadramento fixo da figura; nenhum ponto futuro é enviado à inferência."""
    p = np.asarray(points)
    low, high = p[:, :2].min(0), p[:, :2].max(0)
    center = (low + high) / 2
    radius = max(float((high-low).max()), 1.) * .58
    for i in range(cut):
        color = ADDED if i >= previous else INK
        if i and p[i-1, 2] < .5:
            ax.plot(p[i-1:i+1, 0], p[i-1:i+1, 1], color=color,
                    lw=1.8, solid_capstyle='round')
        else:
            ax.scatter(p[i, 0], p[i, 1], color=color, s=6, zorder=3)
    ax.scatter(p[cut-1, 0], p[cut-1, 1], s=14, color=ADDED, zorder=4)
    ax.set(xlim=(center[0]-radius, center[0]+radius),
           ylim=(center[1]+radius, center[1]-radius), aspect='equal')
    ax.axis('off')


def plot_probabilities(ax, probabilities, family, label, labels):
    p = np.asarray(probabilities)
    best = np.argsort(-p, kind='stable')[:3]
    color = COLORS[family]
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    correct = int(best[0]) == label
    name = 'GRU' if family == 'gru' else 'MLP'
    ax.text(.02, .98, name, color=color, fontweight='bold', fontsize=9, va='top')
    ax.text(.98, .98, 'acerta' if correct else 'erra', ha='right', va='top', fontsize=8,
            color='#267257' if correct else '#A04835')
    for rank, index in enumerate(best):
        y = .76 - rank*.23
        ax.text(.02, y, labels[index], fontsize=7.6, va='center',
                fontweight='bold' if index == label else 'normal', color=INK)
        ax.text(.98, y-.10, percentage(p[index]), fontsize=7.6, va='center', ha='right', color=INK)
        ax.plot([.02, .63], [y-.10, y-.10], color='#E3E7E8', lw=3.5, solid_capstyle='round')
        ax.plot([.02, .02+.61*p[index]], [y-.10, y-.10], color=color, lw=3.5,
                solid_capstyle='round')
    ax.text(.02, .015, 'P(alvo): '+percentage(p[label]), fontsize=7.5, color=color, va='bottom')


def fish_contrast_case(output, models):
    """Reconstrói a seleção ilustrativa usando somente a validação, se necessário."""
    target = output/'report/fish_case_candidate.json'
    if target.exists():
        return read_json(target)
    source_name = read_json(output/'config.json')['source_experiment']
    source = ROOT/'results'/source_name/'data'
    records = read_json(source/'records.json')
    cases = []
    with np.load(source/'sequences.npz') as arrays:
        for index, record in enumerate(records):
            if record['label']==1 and record['split']=='validation':
                cases.append({**record, 'points':arrays[f'points_{index}']})
    probabilities = predict(models, points=[case['points'] for case in cases])
    eligible = np.flatnonzero((probabilities['gru'].argmax(1)==1)&
                             (probabilities['mlp'].argmax(1)==6))
    if not len(eligible):
        return None
    index = int(max(eligible, key=lambda i:float(probabilities['mlp'][i,6]-probabilities['mlp'][i,1])))
    chosen = cases[index]
    chosen['points'] = chosen['points'].tolist()
    chosen['selection_rule'] = ('Validação: desenhos Peixe com GRU correta e MLP prevendo Cachorro ao final; '
                                'maior diferença P_MLP(Cachorro)-P_MLP(Peixe). '
                                'Caso extremo ilustrativo; não é o desenho da captura do usuário.')
    write_json(target, chosen)
    return chosen


def make_gallery(output, examples, labels):
    torch.set_num_threads(4)
    models = load_models(output)
    by_label = {e['label']: e for e in examples[:12]}
    # Cobrir as12 categorias e incluir contrastes já documentados na validação.
    for phrase in ['favorece GRU', 'favorece MLP']:
        candidates = [e for e in examples if phrase in e.get('selection_rule', '')]
        for e in (candidates[:1] if phrase == 'favorece GRU' else candidates):
            by_label[e['label']] = e
    ordered = [by_label[1]] + [by_label[i] for i in range(12) if i != 1]
    fish = fish_contrast_case(output, models)
    if fish is not None:
        fish.update({'source_split': 'validation', 'label_name': labels[fish['label']]})
        ordered.insert(0, fish)
    pages, saved = [], []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    for case_index, example in enumerate(ordered, start=1):
        points = np.asarray(example['points'], np.float32)
        n = len(points); label = int(example['label'])
        cuts = [max(2, int(np.floor(n*f+.5))) for f in FRACTIONS]
        curve_cuts = sorted(set(cuts + [max(2, int(np.floor(n*f+.5))) for f in np.arange(.1, 1.01, .1)]))
        probabilities = predict(models, points=[points[:cut] for cut in curve_cuts])
        lookup = {cut: j for j, cut in enumerate(curve_cuts)}
        fig = plt.figure(figsize=(7.1, 8.05), facecolor='white')
        grid = fig.add_gridspec(6, 5, left=.035, right=.985, top=.97, bottom=.055,
                               height_ratios=[.27, 1.4, .55, 1.14, 1.14, 1.23],
                               hspace=.29, wspace=.23)
        previous = 0; snapshots = []
        for column, (fraction, cut) in enumerate(zip(FRACTIONS, cuts)):
            elapsed = float(points[cut-1, 3] - points[0, 3])
            ax = fig.add_subplot(grid[0, column]); ax.axis('off')
            ax.text(.5, 1, f'{int(fraction*100)}% observado', ha='center', va='top',
                    fontsize=8.4, fontweight='bold', color=INK)
            ax.text(.5, .27, f'{cut}/{n} eventos · {elapsed:.1f} s'.replace('.', ','),
                    ha='center', va='top', fontsize=7.4, color='#596C77')
            plot_prefix(fig.add_subplot(grid[1, column]), points, cut, previous)
            previous = cut
            _, image = represent(points[:cut])
            ax = fig.add_subplot(grid[2, column]); ax.imshow(image, cmap='gray_r', vmin=0, vmax=1,
                                                           interpolation='nearest'); ax.axis('off')
            ax.set_title('Pixels da MLP', fontsize=7.2, pad=2)
            j = lookup[cut]
            for row, family in [(3, 'gru'), (4, 'mlp')]:
                plot_probabilities(fig.add_subplot(grid[row, column]), probabilities[family][j],
                                   family, label, labels)
            snapshots.append({'fraction_requested': fraction, 'cut': cut,
                              'fraction_observed': cut/n, 'elapsed_seconds': elapsed,
                              'gru': probabilities['gru'][j].tolist(), 'mlp': probabilities['mlp'][j].tolist()})
        ax = fig.add_subplot(grid[5, :])
        box = ax.get_position()
        ax.set_position([.095, box.y0, .89, box.height])
        for family in ['gru', 'mlp']:
            ax.plot(100*np.array(curve_cuts)/n, probabilities[family][:, label], '-o',
                    color=COLORS[family], ms=3, lw=1.5, label='GRU' if family=='gru' else 'MLP comum')
        ax.set(xlim=(0, 101), ylim=(0, 1.04), xticks=[10,25,50,75,100],
               xlabel='Eventos observados (%)', ylabel='P(categoria real)',
               yticks=[0,.25,.5,.75,1], yticklabels=['0%','25%','50%','75%','100%'])
        ax.tick_params(labelsize=8); ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
        ax.grid(alpha=.18); ax.legend(loc='lower right', fontsize=8)
        stem = f'gallery_{case_index:02d}_{example["id"]}'
        path = output/'figures'/f'{stem}.pdf'
        for ext in ['pdf', 'png']:
            fig.savefig(path.with_suffix('.'+ext), dpi=200)
        plt.close(fig)
        rule = example.get('selection_rule', '')
        case_name = labels[label]
        if case_index == 1 and fish is not None:case_name += ' — confusão com cachorro'
        elif 'favorece GRU' in rule:case_name += ' — contraste favorável à GRU'
        elif 'favorece MLP' in rule:case_name += ' — contraste favorável à MLP'
        caption = (f'Categoria real: {labels[label]}. Cada coluna usa somente o prefixo indicado. '
                   'Top-3 e P(alvo) são probabilidades da seed 42; laranja no desenho destaca '
                   'os eventos acrescentados desde a coluna anterior. O enquadramento é fixo apenas na ilustração; '
                   'a entrada de cada rede é normalizada usando somente seu próprio prefixo.')
        pages.append({'title': f'Caso {case_index} — '+case_name,
                      'text': f'Validação · ID {example["id"]} · {n} eventos no desenho completo. '+rule,
                      'figure': path, 'caption': caption})
        saved.append({'id': example['id'], 'label': label, 'label_name': labels[label],
                      'source_split': 'validation', 'selection_rule': rule,
                      'points': points.tolist(), 'snapshots': snapshots,
                      'curve': [{'cut': cut, 'gru': probabilities['gru'][j].tolist(),
                                 'mlp': probabilities['mlp'][j].tolist()} for j,cut in enumerate(curve_cuts)]})
    write_json(output/'report/gallery_cases.json', saved)
    assert {e['label'] for e in saved} == set(range(12))
    assert all(e['source_split']=='validation' for e in saved)
    assert all(s['cut'] <= len(e['points']) and len(s['gru'])==len(s['mlp'])==12
               for e in saved for s in e['snapshots'])
    checkpoint_hashes = {}
    for family in ['gru', 'mlp']:
        spec = read_json(output/family/'frozen_selection.json')['best']['spec']
        path = output/family/f'h{spec["hidden"]}_lr{spec["lr"]}_seed42'/'best.pt'
        checkpoint_hashes[family] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(output/'report/gallery_audit.json', {'passed': True, 'cases': len(saved),
               'snapshots': len(saved)*5, 'categories': 12, 'split': 'validation',
               'seed': 42, 'trained_new_weights': False, 'checkpoint_sha256': checkpoint_hashes,
               'all_12_probabilities_saved': True, 'display_frame_not_model_normalization': True,
               'user_screenshot_reconstructed': False})
    return pages
