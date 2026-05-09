#!/usr/bin/env python3
"""
AutoTherm WebUI — Animated TIM Optimization Visualization
Runs at http://localhost:7860
"""

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Polygon, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import numpy as np
import io
import time
from PIL import Image

# ── Palette ───────────────────────────────────────────────────────────────────
BG     = '#08111E'
PANEL  = '#0C1A2E'
BORDER = '#1A3A6A'
CYAN   = '#00D4FF'
GREEN  = '#00FF99'
ORANGE = '#FF8C00'
WHITE  = '#E8F4FD'
DIM    = '#4A6A8A'
GOLD   = '#D4A020'

THERMAL = LinearSegmentedColormap.from_list('th',
    ['#0033FF','#0099FF','#00FFCC','#FFCC00','#FF4400','#FF0000'])

# ── Isometric projection ───────────────────────────────────────────────────────
ANG = np.radians(30)
COS30, SIN30 = np.cos(ANG), np.sin(ANG)

def proj_pts(pts):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.column_stack([(x - y) * COS30, (x + y) * SIN30 + z * 0.90])

# ── Optimization data from the 50-iteration run ───────────────────────────────
# (iteration, R_th, visual_order_level)
KEY_POINTS = [
    (0,  2.28e-5, 0.00),
    (1,  7.42e-6, 0.12),
    (30, 5.49e-6, 0.65),
    (49, 4.55e-6, 1.00),
]
BASELINE_R = 2.28e-5
BEST_ITERS = {1, 30, 49}
CRASH_ITERS = {8}
# All sampled R_th values across iterations (for the sparkline)
SAMPLED = {
    0: 2.28e-5, 1: 7.42e-6, 8: 2.10e-5, 15: 8.91e-6,
    22: 6.83e-6, 30: 5.49e-6, 38: 5.71e-6, 44: 5.12e-6, 49: 4.55e-6,
}

def get_params(iteration):
    """Smooth-interpolate R_th and order_level for any iteration 0-49."""
    pts = KEY_POINTS
    for i in range(len(pts) - 1):
        i0, r0, o0 = pts[i]
        i1, r1, o1 = pts[i + 1]
        if i0 <= iteration <= i1:
            t = (iteration - i0) / (i1 - i0) if i1 > i0 else 0.0
            t = t * t * (3 - 2 * t)          # smoothstep
            return r0 + t * (r1 - r0), o0 + t * (o1 - o0)
    return pts[-1][1], pts[-1][2]

# ── Glass cube ─────────────────────────────────────────────────────────────────
CORNERS = np.array([
    [0,0,0],[1,0,0],[1,1,0],[0,1,0],
    [0,0,1],[1,0,1],[1,1,1],[0,1,1],
], dtype=float) - 0.5
FACES = [
    ([3,2,6,7], '#0A2040', 0.20),
    ([0,3,7,4], '#0C2848', 0.18),
    ([0,1,2,3], '#080E1C', 0.15),
    ([0,1,5,4], '#0D2040', 0.22),
    ([1,2,6,5], '#0F2848', 0.20),
    ([4,5,6,7], '#101E38', 0.25),
]
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def draw_cube(ax, cx, cy, S, edge_col=CYAN):
    c = CORNERS * S
    p = proj_pts(c)
    px, py = p[:, 0] + cx, p[:, 1] + cy
    depth = [np.mean(c[idxs, 1] - c[idxs, 2]) for idxs, _, _ in FACES]
    for i in np.argsort(depth)[::-1]:
        idxs, fc, a = FACES[i]
        ax.fill([px[j] for j in idxs], [py[j] for j in idxs], color=fc, alpha=a, zorder=2)
    for i, j in EDGES:
        ax.plot([px[i], px[j]], [py[i], py[j]], color=edge_col, alpha=0.55, lw=1.1, zorder=20)

def glow(ax, xs, ys, color, lw=9, n=8, zorder=6, a=1.0):
    for i in range(n, 0, -1):
        f = i / n
        ax.plot(xs, ys, color=color, lw=lw*f, alpha=0.06*f*a,
                solid_capstyle='round', zorder=zorder)
    ax.plot(xs, ys, color='white', lw=lw*0.12, alpha=0.65*a,
            solid_capstyle='round', zorder=zorder + 1)

# ── Per-iteration structure renderer ──────────────────────────────────────────
def draw_structure(ax, cx, cy, order, S=3.2):
    """Blend between chaotic baseline (order=0) and hex-layered optimal (order=1)."""

    # Cube edge colour brightens from dim blue → cyan with order
    r = 0.13 - order * 0.12
    g = 0.40 + order * 0.42
    b = 0.53 + order * 0.47
    edge_col = '#{:02x}{:02x}{:02x}'.format(
        max(0, min(255, int(r*255))),
        max(0, min(255, int(g*255))),
        max(0, min(255, int(b*255))),
    )
    draw_cube(ax, cx, cy, S, edge_col)

    # ── Random metal droplets (baseline style, fade out) ──────────────────────
    n_rand_m = int((1 - order) * 48)
    if n_rand_m > 0:
        rng = np.random.RandomState(11)
        m3 = (rng.rand(48, 3) - 0.5) * S * 0.82
        dep = m3[:, 1] - m3[:, 2]
        for i in np.argsort(dep)[:n_rand_m]:
            d = (dep[i] + S*1.2) / (S*2.4)
            pp = proj_pts(m3[i:i+1])[0]
            r_ = 0.050 + 0.030*d
            a_ = (0.55 + 0.35*d) * (1 - order)
            ax.add_patch(Circle((pp[0]+cx, pp[1]+cy), r_,
                                color='#C8A020', alpha=a_, zorder=8+d))
            ax.add_patch(Circle((pp[0]+cx-r_*.3, pp[1]+cy+r_*.3), r_*.3,
                                color='#FFE860', alpha=a_*.6, zorder=8+d+.1))

    # ── Ordered metal clusters between layers (fade in) ────────────────────────
    n_lay_m = int(order * 50)
    if n_lay_m > 0:
        n_lay = max(2, int(order * 7) + 1)
        layer_zs = np.linspace(-0.44, 0.44, n_lay) * S
        rng2 = np.random.RandomState(55)
        for _ in range(n_lay_m):
            li = rng2.randint(0, len(layer_zs) - 1)
            z_m = (layer_zs[li] + layer_zs[min(li+1, len(layer_zs)-1)]) / 2 \
                  + rng2.uniform(-0.02, 0.02)*S
            x_m = rng2.uniform(-0.38, 0.38)*S
            y_m = rng2.uniform(-0.38, 0.38)*S
            pp = proj_pts(np.array([[x_m, y_m, z_m]]))[0]
            d = 0.4 + 0.4*(y_m + S*.5)/S
            r_ = 0.040 + 0.018*d
            a_ = (0.65 + 0.25*d) * order
            ax.add_patch(Circle((pp[0]+cx, pp[1]+cy), r_, color=GOLD, alpha=a_, zorder=8+d))
            ax.add_patch(Circle((pp[0]+cx-r_*.3, pp[1]+cy+r_*.3), r_*.28,
                                color='#FFE870', alpha=a_*.55, zorder=8+d+.1))

    # ── Random filler ellipses (baseline, fade out) ────────────────────────────
    n_rand_f = int((1 - order) * 22)
    if n_rand_f > 0:
        rng3 = np.random.RandomState(22)
        f3 = (rng3.rand(22, 3) - 0.5)*S*0.78
        angles = rng3.uniform(0, 180, 22)
        dep2 = f3[:, 1] - f3[:, 2]
        for i in np.argsort(dep2)[:n_rand_f]:
            d = (dep2[i]+S*1.2)/(S*2.4)
            a_ = (0.50+0.35*d)*(1-order)
            pp = proj_pts(f3[i:i+1])[0]
            ax.add_patch(Ellipse((pp[0]+cx, pp[1]+cy),
                                 0.24+.06*d, 0.07+.02*d, angle=angles[i],
                                 color='#2870C0', alpha=a_, zorder=7+d))

    # ── Hex layers (fade in layer by layer) ───────────────────────────────────
    n_complete = min(7, int(order * 8))
    partial = (order * 8) - n_complete
    total_draw = n_complete + (1 if partial > 0.15 else 0)

    if total_draw > 0:
        layer_zs = np.linspace(-0.44, 0.44, 7)*S
        for li in range(min(total_draw, 7)):
            lz = layer_zs[li]
            heat = li / 6.0
            la = partial if li == n_complete else 1.0

            NX, NY = 4, 3
            for hx in range(NX):
                for hy in range(NY):
                    x_c = ((hx/(NX-1)) - 0.5)*S*0.76
                    y_c = ((hy/(NY-1)) - 0.5)*S*0.62 + (0.09 if hx%2 else 0)
                    ang6 = np.linspace(0, 2*np.pi, 6, endpoint=False)
                    r6 = 0.112*S
                    hex3 = np.array([[x_c+r6*np.cos(a), y_c+r6*np.sin(a)*0.58, lz]
                                     for a in ang6])
                    hp = proj_pts(hex3); hp[:, 0] += cx; hp[:, 1] += cy
                    dn = 0.4 + 0.4*(hy/(NY-1))
                    ax.add_patch(Polygon(list(zip(hp[:, 0], hp[:, 1])), closed=True,
                                         color='#1E5898', alpha=(0.55+0.2*dn)*la, zorder=5+dn))
                    ax.plot(list(hp[:, 0])+[hp[0, 0]], list(hp[:, 1])+[hp[0, 1]],
                            color='#60C0FF', alpha=(0.5+0.3*dn)*la, lw=0.7, zorder=5+dn+.1)

            # Ordered heat channels
            ch_a = min(1.0, max(0, (order - 0.25)/0.5)) * la
            for hi in range(5):
                y_h = (-0.35 + hi*0.18)*S
                x3  = np.linspace(-0.46, 0.46, 22)*S
                pp = proj_pts(np.column_stack([x3, np.full(22,y_h), np.full(22,lz+0.015*S)]))
                pp[:, 0] += cx; pp[:, 1] += cy
                glow(ax, pp[:, 0], pp[:, 1], THERMAL(heat), lw=9, n=8, zorder=6, a=ch_a)

    # ── Chaotic heat paths (baseline, fade out) ────────────────────────────────
    n_chaotic = max(0, int((1 - min(1, order*1.3)) * 10))
    if n_chaotic > 0:
        rng4 = np.random.RandomState(33)
        for _ in range(n_chaotic):
            t = np.linspace(0, 1, 16)
            f1, f2 = rng4.uniform(3,9), rng4.uniform(2,7)
            p1, p2 = rng4.uniform(0,6), rng4.uniform(0,6)
            x3 = (np.linspace(-.45,.45,16) + .25*np.sin(f1*np.pi*t+p1))*S
            y3 = (rng4.uniform(-.3,.3) + .25*np.cos(f2*np.pi*t+p2))*S
            z3 = np.linspace(-.45,.45,16)*S
            pp = proj_pts(np.column_stack([x3, y3, z3]))
            pp[:, 0] += cx; pp[:, 1] += cy
            heat = (z3/S + 0.5)
            a_fade = max(0, 1 - order*1.4)
            for j in range(15):
                glow(ax, [pp[j,0],pp[j+1,0]], [pp[j,1],pp[j+1,1]],
                     THERMAL((heat[j]+heat[j+1])/2), lw=6, n=5, zorder=6, a=a_fade)

# ── Full frame renderer ────────────────────────────────────────────────────────
def make_frame(iteration):
    r_th, order = get_params(iteration)
    improvement = max(0.0, (BASELINE_R - r_th) / BASELINE_R * 100)
    is_best   = iteration in BEST_ITERS
    is_crash  = iteration in CRASH_ITERS

    fig = plt.figure(figsize=(18, 11), facecolor=BG, dpi=110)
    ax  = fig.add_axes([0, 0, 1, 1], facecolor=BG)
    ax.set_xlim(0, 18); ax.set_ylim(0, 11)
    ax.set_aspect('equal'); ax.axis('off')

    # Outer border (flashes green on new best)
    border_col = '#76B900' if is_best else ('#FF4444' if is_crash else '#1A3560')
    ax.add_patch(FancyBboxPatch((0.12, 0.12), 17.76, 10.76,
                 boxstyle="round,pad=0.15", linewidth=2.5 if is_best else 1.5,
                 edgecolor=border_col, facecolor='none', zorder=0))

    # ── Title ─────────────────────────────────────────────────────────────────
    stage_lbl = "BASELINE" if iteration == 0 else f"ITERATION  {iteration} / 49"
    if is_best:
        status_txt, status_col = "★  NEW BEST  ★", GREEN
    elif is_crash:
        status_txt, status_col = "⚠  SIMULATION RESTART", '#FF4444'
    else:
        status_txt, status_col = "OPTIMIZING …", DIM

    ax.text(9.0, 10.7, f'AutoTherm — AI-Driven TIM Molecular Optimization',
            color=WHITE, fontsize=15, fontweight='bold', ha='center', va='top',
            fontfamily='monospace')
    ax.text(9.0, 10.25, f'{stage_lbl}   |   {status_txt}',
            color=status_col, fontsize=9, ha='center', va='top', fontfamily='monospace')
    ax.text(9.0, 9.85, 'GROMACS 2025.1 MD  ·  16,000 atoms  ·  3× NVIDIA Blackwell GB10  ·  Qwen3.5-122B-A10B-AWQ',
            color=DIM, fontsize=6.5, ha='center', va='top', fontfamily='monospace')

    # NVIDIA badge
    ax.add_patch(FancyBboxPatch((0.25, 9.9), 2.8, 0.72, boxstyle="round,pad=0.07",
                 facecolor='#76B900', edgecolor='none', zorder=1))
    ax.text(1.65, 10.26, 'NVIDIA  RESEARCH', color='white', fontsize=7.5,
            fontweight='bold', ha='center', va='center', fontfamily='monospace')

    # ── BIG R_th display (top-right) ──────────────────────────────────────────
    if r_th < 5e-6:   rth_col = GREEN
    elif r_th < 8e-6: rth_col = CYAN
    elif r_th < 1.5e-5: rth_col = ORANGE
    else:             rth_col = '#FF4444'

    ax.add_patch(FancyBboxPatch((12.5, 9.1), 5.3, 1.6, boxstyle="round,pad=0.12",
                 facecolor='#060D18', edgecolor=rth_col, linewidth=1.8, zorder=1))
    ax.text(15.15, 10.55, 'THERMAL RESISTANCE', color=DIM, fontsize=7.5,
            ha='center', va='top', fontfamily='monospace')
    ax.text(15.15, 10.12, f'{r_th:.3e}  m²K/W', color=rth_col, fontsize=15,
            ha='center', va='top', fontfamily='monospace', fontweight='bold')
    ax.text(15.15, 9.28, f'▼  {improvement:.1f}%  improvement  vs  baseline',
            color=GREEN, fontsize=8, ha='center', va='bottom', fontfamily='monospace')

    # ── LEFT PANEL: composition + order meter ─────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.2, 0.7), 4.0, 8.8, boxstyle="round,pad=0.1",
                 facecolor=PANEL, edgecolor=BORDER, linewidth=1.0, zorder=1))
    ax.text(0.42, 9.3, 'FORMULATION', color=CYAN, fontsize=9, fontweight='bold',
            fontfamily='monospace')

    # Compositions (interpolated)
    metal_f  = 0.600 + order * 0.088
    filler_f = max(0.05, 0.250 - order * 0.139)
    binder_f = 1.0 - metal_f - filler_f
    comps = [
        ('InGa Metal',   metal_f,  GOLD,      '●'),
        ('Graphene/BN',  filler_f, CYAN,       '[H]'),
        ('Polysiloxane', binder_f, '#AA99FF',  '◆'),
    ]
    yc = 8.9
    for name, val, col, sym in comps:
        ax.text(0.42, yc, f'{sym}  {name}', color=col, fontsize=8, fontfamily='monospace')
        bw = max(0.05, val * 3.6)
        ax.add_patch(FancyBboxPatch((0.42, yc-0.44), bw, 0.28,
                     boxstyle="round,pad=0.02", facecolor=col, alpha=0.55, zorder=2))
        ax.text(0.42+bw+0.08, yc-0.28, f'{val*100:.1f} wt%',
                color=col, fontsize=7.5, va='center', fontfamily='monospace')
        yc -= 1.05

    # Order meter
    ax.text(0.42, 5.7, 'STRUCTURAL ORDER', color=CYAN, fontsize=8,
            fontweight='bold', fontfamily='monospace')
    ax.add_patch(FancyBboxPatch((0.42, 5.15), 3.5, 0.38,
                 boxstyle="round,pad=0.02", facecolor='#080F1A', edgecolor=DIM, linewidth=0.5, zorder=2))
    ax.add_patch(FancyBboxPatch((0.42, 5.15), max(0.06, order*3.5), 0.38,
                 boxstyle="round,pad=0.02", facecolor=CYAN, alpha=0.75, zorder=3))
    ax.text(0.42, 4.95, f'{order:.0%}  disorder → hexagonal layers',
            color=DIM, fontsize=6.5, fontfamily='monospace')

    # LJ cross-interactions
    ax.text(0.42, 4.45, 'CROSS-INTERACTIONS  ε  (kJ/mol)', color=CYAN, fontsize=7.5,
            fontweight='bold', fontfamily='monospace')
    e_mf = 3.50 + order * 3.50
    e_mb = 2.20 + order * 3.80
    e_fb = 1.90 + order * 3.10
    for label, val, yp in [('ε metal-filler ', e_mf, 4.05),
                             ('ε metal-binder ', e_mb, 3.65),
                             ('ε filler-binder', e_fb, 3.25)]:
        ax.text(0.42, yp, f'{label}: {val:.2f}', color=WHITE, fontsize=7.5,
                fontfamily='monospace')

    # Temperature / time
    ax.text(0.42, 2.7, 'SIM CONDITIONS', color=CYAN, fontsize=7.5,
            fontweight='bold', fontfamily='monospace')
    sim_t = 100 + order * 100
    ax.text(0.42, 2.38, f'T = 350 K   |   {sim_t:.0f} ps', color=WHITE, fontsize=7.5,
            fontfamily='monospace')
    ax.text(0.42, 2.05, f'N = 16,000 atoms  |  NPT ensemble', color=WHITE, fontsize=7.5,
            fontfamily='monospace')

    # Compute
    ax.text(0.42, 1.55, 'COMPUTE', color=CYAN, fontsize=7.5,
            fontweight='bold', fontfamily='monospace')
    ax.text(0.42, 1.22, '⚡ 3 × DGX Spark  (GB10 Blackwell)', color=WHITE, fontsize=7.5,
            fontfamily='monospace')
    ax.text(0.42, 0.88, '  Qwen3.5-122B on node0 (vLLM)', color=DIM, fontsize=7,
            fontfamily='monospace')

    # ── CENTER: 3D molecular structure ────────────────────────────────────────
    draw_structure(ax, 9.0, 5.5, order, S=3.4)

    # ── RIGHT PANEL: R_th chart ───────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((13.8, 0.7), 4.0, 8.8, boxstyle="round,pad=0.1",
                 facecolor=PANEL, edgecolor=BORDER, linewidth=1.0, zorder=1))
    ax.text(14.0, 9.3, 'R_th  HISTORY', color=CYAN, fontsize=9,
            fontweight='bold', fontfamily='monospace')

    cx0, cy0, cw, ch = 14.1, 1.3, 3.5, 7.5
    r_min, r_max = 4e-6, 2.4e-5

    # Grid
    for gv in [0.2, 0.4, 0.6, 0.8, 1.0]:
        gy = cy0 + gv*ch
        ax.plot([cx0, cx0+cw], [gy, gy], color='#152535', lw=0.5, zorder=3)
        rv = r_max - gv*(r_max-r_min)
        ax.text(cx0-0.07, gy, f'{rv:.0e}', color=DIM, fontsize=5.8,
                ha='right', va='center', fontfamily='monospace')

    def s_r(r): return cy0 + ch*(1-(r-r_min)/(r_max-r_min))
    def s_i(i): return cx0 + cw*i/49

    # Draw known sampled points up to this iteration
    visible = sorted((i, r) for i, r in SAMPLED.items() if i <= iteration)
    if len(visible) > 1:
        xs = [s_i(i) for i, r in visible]
        ys = [s_r(r) for i, r in visible]
        ax.plot(xs, ys, color=CYAN, lw=1.2, alpha=0.5, zorder=4)

    for i, r in visible:
        col = GREEN if i in BEST_ITERS else ('#FF4444' if i in CRASH_ITERS else DIM)
        ax.add_patch(Circle((s_i(i), s_r(r)), 0.06, color=col, zorder=5))

    # Current position (interpolated)
    ax.add_patch(Circle((s_i(iteration), s_r(r_th)), 0.10,
                        color=CYAN, zorder=6))

    # Improvement annotations
    if iteration >= 1:
        ax.text(s_i(1)+0.05, s_r(7.42e-6)-0.15, f'★ iter 1\n7.42×10⁻⁶',
                color=GREEN, fontsize=5.5, fontfamily='monospace', zorder=7)
    if iteration >= 30:
        ax.text(s_i(30)+0.05, s_r(5.49e-6)-0.25, f'★ iter 30\n5.49×10⁻⁶',
                color=GREEN, fontsize=5.5, fontfamily='monospace', zorder=7)
    if iteration >= 49:
        ax.text(s_i(49)-0.45, s_r(4.55e-6)+0.2, f'★ BEST\n4.55×10⁻⁶',
                color=GREEN, fontsize=5.5, fontfamily='monospace', zorder=7)

    # ── Progress bar (bottom) ─────────────────────────────────────────────────
    ax.add_patch(FancyBboxPatch((0.2, 0.12), 17.6, 0.52,
                 boxstyle="round,pad=0.04", facecolor='#060D18',
                 edgecolor=BORDER, linewidth=0.7, zorder=1))
    prog_w = max(0.05, 16.8 * iteration / 49)
    ax.add_patch(FancyBboxPatch((0.68, 0.17), prog_w, 0.38,
                 boxstyle="round,pad=0.02",
                 facecolor=GREEN if is_best else CYAN, alpha=0.35, zorder=2))
    ax.text(0.42, 0.38, 'PROGRESS:', color=DIM, fontsize=6.5,
            va='center', fontfamily='monospace')
    ax.text(17.7, 0.38, f'{iteration}/49', color=WHITE, fontsize=7.5,
            va='center', ha='right', fontfamily='monospace', fontweight='bold')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf)

# ── Frame cache ────────────────────────────────────────────────────────────────
_cache: dict = {}

def get_frame(iteration: int) -> Image.Image:
    if iteration not in _cache:
        _cache[iteration] = make_frame(iteration)
    return _cache[iteration]

# ── Animation generator ────────────────────────────────────────────────────────
def run_animation(pause_s: float):
    """Yield (image, status_html) for each iteration 0→49."""
    for it in range(50):
        r_th, order = get_params(it)
        improvement = max(0, (BASELINE_R - r_th) / BASELINE_R * 100)
        img = get_frame(it)

        label = "BASELINE" if it == 0 else f"Iteration {it}"
        badge = ""
        if it in BEST_ITERS:
            badge = "<span style='color:#00FF99;font-weight:bold'> ★ NEW BEST</span>"
        elif it in CRASH_ITERS:
            badge = "<span style='color:#FF4444'> ⚠ RESTART</span>"

        status_html = (
            f"<div style='font-family:monospace;font-size:14px;color:#E8F4FD'>"
            f"<b>{label}</b>{badge}&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"R_th = <b style='color:#00D4FF'>{r_th:.3e}</b> m²K/W&nbsp;&nbsp;|&nbsp;&nbsp;"
            f"Improvement: <b style='color:#00FF99'>{improvement:.1f}%</b>"
            f"</div>"
        )
        yield img, status_html
        if it < 49:
            time.sleep(pause_s)

def jump_to(iteration: int):
    r_th, _ = get_params(iteration)
    improvement = max(0, (BASELINE_R - r_th) / BASELINE_R * 100)
    status_html = (
        f"<div style='font-family:monospace;font-size:14px;color:#E8F4FD'>"
        f"<b>Iteration {iteration}</b>&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"R_th = <b style='color:#00D4FF'>{r_th:.3e}</b> m²K/W&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Improvement: <b style='color:#00FF99'>{improvement:.1f}%</b>"
        f"</div>"
    )
    return get_frame(iteration), status_html

# ── Gradio UI ──────────────────────────────────────────────────────────────────
CSS = """
body, .gradio-container { background: #08111E !important; }
.dark { background: #08111E !important; }
h1, h2, h3 { color: #00D4FF !important; font-family: monospace; }
.gr-button-primary { background: #00D4FF !important; color: #08111E !important;
                     font-weight: bold; font-family: monospace; }
.gr-button { background: #0C1A2E !important; color: #E8F4FD !important;
             border: 1px solid #1A3A6A !important; font-family: monospace; }
footer { display: none !important; }
"""

with gr.Blocks(title="AutoTherm TIM Optimizer") as demo:

    gr.Markdown("""
# AutoTherm — AI-Driven TIM Optimization
### 50 Iterations · GROMACS 2025.1 MD · Qwen3.5-122B · 3× NVIDIA Blackwell GB10
""")

    frame_display = gr.Image(
        label="", show_label=False,
        height=660,
    )

    status_box = gr.HTML(
        value="<div style='font-family:monospace;color:#4A6A8A'>Press ▶ Start Animation to begin</div>"
    )

    with gr.Row():
        start_btn  = gr.Button("▶  Start Animation (all 50 iterations)", variant="primary", scale=3)
        pause_sl   = gr.Slider(1, 10, value=5, step=0.5, label="Pause per frame (s)", scale=2)

    with gr.Row():
        iter_slider = gr.Slider(0, 49, value=0, step=1, label="Jump to iteration", scale=4)
        jump_btn    = gr.Button("Go", scale=1)

    gr.Markdown("""
---
**Key results from the 50-iteration loop:**
| Iteration | R_th (m²K/W) | Change |
|-----------|-------------|--------|
| 0 — Baseline | 2.28 × 10⁻⁵ | — |
| 1 | 7.42 × 10⁻⁶ | ★ −67.4% |
| 30 | 5.49 × 10⁻⁶ | ★ −75.9% |
| **49 — Optimal** | **4.55 × 10⁻⁶** | **★ −80.1%** |

Best formulation: **68.8 wt% InGa · 11.1 wt% graphene/BN · 20.1 wt% polysiloxane**
""")

    # Wire up events
    start_btn.click(
        fn=run_animation,
        inputs=[pause_sl],
        outputs=[frame_display, status_box],
    )
    jump_btn.click(
        fn=jump_to,
        inputs=[iter_slider],
        outputs=[frame_display, status_box],
    )
    iter_slider.release(
        fn=jump_to,
        inputs=[iter_slider],
        outputs=[frame_display, status_box],
    )

if __name__ == "__main__":
    print("Pre-rendering baseline frame …")
    get_frame(0)
    print("Ready — launching at http://localhost:7860")
    demo.launch(server_port=7860, server_name="0.0.0.0", share=False,
                theme=gr.themes.Base(), css=CSS)
