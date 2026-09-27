"""Plot the two fixed start panels from one stabilize-avoid evaluation JSON/NPZ."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# This file lives at examples/quad2d_stabilize_avoid/plot.py in the repository.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'custom'))
from ssm.tasks.quad2d_stabilize.env import get_layout

INDICES = (0, 28, 57, 85, 114, 142, 171, 199)
COLORS = {'left': '#087f8c', 'right': '#4167b2', 'other': '#8c6876'}


def read_panels(path):
    data = json.loads(path.read_text())
    if data['task'] != 'quad2d_stabilize' or data['protocol']['episodes'] != 200 or data['protocol']['horizon'] != 400:
        raise ValueError('Expected complete 200-episode, 400-step stabilize-avoid panels.')
    arrays_path = path.with_suffix('.npz')
    if hashlib.sha256(arrays_path.read_bytes()).hexdigest() != data['identity']['arrays_sha256']:
        raise ValueError('Adjacent NPZ does not match the evaluation JSON.')
    panels = {}
    with np.load(arrays_path, allow_pickle=False) as arrays:
        for kind in ('vel_small', 'exact'):
            episodes = data['panels'][kind]['episodes']
            states = arrays[f'{kind}/states']
            lengths = arrays[f'{kind}/steps_taken']
            if states.shape != (200, 401, 6) or [e['episode'] for e in episodes] != list(range(200)):
                raise ValueError(f'{kind}: incomplete or reordered episodes.')
            if not np.array_equal(lengths, [e['length'] for e in episodes]) or np.any((lengths < 1) | (lengths > 400)):
                raise ValueError(f'{kind}: inconsistent scored lengths.')
            costs = np.asarray([e['cost'] for e in episodes], float)
            if not np.isfinite(states).all() or not np.isfinite(costs).all():
                raise ValueError(f'{kind}: non-finite saved states or costs.')
            success = np.asarray([e['success'] for e in episodes], bool)
            collision = np.asarray([e['collision'] for e in episodes], bool)
            left = sum(e['route'].startswith('left') for e in episodes)
            right = sum(e['route'].startswith('right') for e in episodes)
            panels[kind] = dict(states=states, lengths=lengths, episodes=episodes,
                                success=int(success.sum()), collision=int(collision.sum()),
                                cost=float(costs.mean()), margin=int((costs > 0).sum()),
                                zero_cost_success=int((success & (costs == 0)).sum()),
                                routes=(left, right, 200 - left - right))
    if not np.all(panels['exact']['states'][:, 0] == panels['exact']['states'][0, 0]):
        raise ValueError('The exact panel does not have one identical initial state.')
    return data, panels


def draw_panel(ax, layout, panel, linewidth):
    xmin, xmax = layout.xlim
    zmin, zmax = layout.zlim
    m, b = layout.obstacle_margin, layout.boundary_margin
    ax.add_patch(Rectangle((xmin, zmin), xmax-xmin, zmax-zmin,
                          facecolor='#fcfdfe', edgecolor='#adb7c2', linewidth=.7))
    ax.add_patch(Rectangle((xmin+b, zmin+b), xmax-xmin-2*b, zmax-zmin-2*b,
                          fill=False, edgecolor='#c8ab83', linewidth=.65, linestyle=(0, (3, 2.5))))
    for obstacle in layout.obstacles:
        ax.add_patch(Rectangle((obstacle.xmin-m, obstacle.zmin-m), obstacle.size[0]+2*m, obstacle.size[1]+2*m,
                              facecolor='#f1e5d2', edgecolor='#bb9460', alpha=.72,
                              linewidth=.8, linestyle=(0, (3, 2)), zorder=2))
        ax.add_patch(Rectangle((obstacle.xmin, obstacle.zmin), *obstacle.size,
                              facecolor='#5c6775', edgecolor='#475363', linewidth=.65, zorder=3))
    for index in INDICES:
        n = int(panel['lengths'][index])
        xy = panel['states'][index, :n+1, :][:, [0, 2]]
        route = panel['episodes'][index]['route']
        color = COLORS['left' if route.startswith('left') else 'right' if route.startswith('right') else 'other']
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=linewidth, alpha=.70, solid_capstyle='round', zorder=5)
        if n >= 3:
            k = int(.23*n)
            ax.annotate('', xy=xy[k+3], xytext=xy[k],
                        arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': linewidth,
                                    'mutation_scale': 8, 'alpha': .78}, zorder=6)
    start = panel['states'][INDICES[0], 0, [0, 2]]
    ax.scatter(*start, s=34, facecolor='#253343', edgecolor='white', lw=.8, zorder=8)
    ax.scatter(*layout.goal, marker='*', s=120, facecolor='#b88026', edgecolor='white', lw=.7, zorder=8)
    ax.set(xlim=(xmin-.04, xmax+.04), ylim=(zmin-.04, zmax+.04), xlabel='$x$ (m)', ylabel='$z$ (m)',
           xticks=[-2, -1, 0, 1, 2], yticks=[-1, 0, 1, 2, 3])
    ax.set_aspect('equal', adjustable='box')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=4)


def make_figure(data, panels, layout):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11.5,
                         'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': .65,
                         'text.color': '#253343', 'axes.labelcolor': '#253343',
                         'xtick.color': '#657181', 'ytick.color': '#657181', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(10.5, 5.25), facecolor='white')
    main = fig.add_axes([.066, .18, .50, .73])
    control = fig.add_axes([.657, .275, .305, .54])
    draw_panel(main, layout, panels['vel_small'], 1.65)
    draw_panel(control, layout, panels['exact'], 1.20)
    main.set_title('a  Small initial-velocity perturbations', loc='left', pad=29, weight='semibold')
    spec = data['protocol']['panels']['vel_small']
    main.text(0, 1.03, f"$v_x^0\\sim U{spec['vx']}$,  $v_z^0\\sim U{spec['vz']}$ m/s",
              transform=main.transAxes, fontsize=9.3, color='#657181', va='bottom')
    control.set_title('b  Identical initial state', loc='left', pad=25, weight='semibold', fontsize=10.8)
    vx, vz = panels['exact']['states'][0, 0, [1, 3]]
    control.text(0, 1.025, f'$v_x^0={vx:g}$, $v_z^0={vz:g}$ m/s',
                 transform=control.transAxes, fontsize=9.3, color='#657181', va='bottom')
    main.annotate('Start', panels['vel_small']['states'][0, 0, [0, 2]], xytext=(12, -10),
                  textcoords='offset points', fontsize=9, color='#475363')
    main.annotate('Goal', layout.goal, xytext=(12, 1), textcoords='offset points', fontsize=9, color='#8b631f')
    handles = [Line2D([0], [0], color=COLORS['left'], lw=2, label='Left route'),
               Line2D([0], [0], color=COLORS['right'], lw=2, label='Right route'),
               Line2D([0], [0], color=COLORS['other'], lw=2, label='Center / no crossing'),
               Rectangle((0, 0), 1, 1, fc='#5c6775', ec='#475363', label='Nominal obstacle'),
               Rectangle((0, 0), 1, 1, fc='#f1e5d2', ec='#bb9460', ls=(0, (3, 2)), label='Safety buffer')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .005), ncol=3,
               frameon=False, fontsize=9.2, handlelength=1.9, columnspacing=1.8)
    return fig


def caption(data, panels, layout):
    identity = data['identity']
    model = Path(identity['checkpoint']).name
    lines = [f"Checkpoint `{model}` (SHA-256 `{identity['checkpoint_sha256'][:12]}`).",
             'Eight fixed episodes per panel: ' + ', '.join(map(str, INDICES)) + '.',
             'Raw trajectories retain failures and end at their scored lengths; no smoothing or selection.',
             f'Gray regions are nominal obstacles. The obstacle buffer is {layout.obstacle_margin:g} m; the workspace inset is {layout.boundary_margin:g} m.', '',
             '| Panel | Success / 200 | Collisions / 200 | Mean margin cost | Any margin violation / 200 | Success with zero cost / 200 | Left / right / other |',
             '|---|---:|---:|---:|---:|---:|---|']
    for kind in ('vel_small', 'exact'):
        row = panels[kind]
        routes = ' / '.join(map(str, row['routes']))
        lines.append(f"| {kind} | {row['success']} | {row['collision']} | {row['cost']:.3f} | {row['margin']} | {row['zero_cost_success']} | {routes} |")
    lines += ['', 'Statistics use all 200 episodes, not only the eight displayed. Collision-free success does not imply zero margin cost. A route split after changing initial velocity does not establish multimodality at one identical full state.',
              'Only the two start panels are drawn; any lower-region panel remains in the evaluation JSON and complete report.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--out-prefix', type=Path, required=True)
    args = parser.parse_args()
    outputs = {kind: Path(str(args.out_prefix) + suffix) for kind, suffix in
               (('png', '.png'), ('pdf', '.pdf'), ('caption', '.md'))}
    for output in outputs.values():
        if output.exists():
            parser.error(f'Refusing to overwrite: {output}')
    data, panels = read_panels(args.result)
    layout = get_layout(data['checkpoint_config']['layout_name'])
    fig = make_figure(data, panels, layout)
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for kind in ('png', 'pdf'):
        metadata = {'Creator': 'Matplotlib; stabilize-avoid plot.py'}
        if kind == 'pdf':
            metadata.update(CreationDate=None, ModDate=None)
        with outputs[kind].open('xb') as stream:
            fig.savefig(stream, format=kind, dpi=300, bbox_inches='tight', pad_inches=.08, metadata=metadata)
    plt.close(fig)
    with outputs['caption'].open('x') as stream:
        stream.write(caption(data, panels, layout))
    print('Wrote ' + ', '.join(str(path) for path in outputs.values()))


if __name__ == '__main__':
    main()
