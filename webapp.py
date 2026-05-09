#!/usr/bin/env python3
"""
AutoTherm WebUI — Full-screen animated TIM dashboard.
v3: proper 3D nanotube cylinders, readable panels, reference-matching layout.
"""

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
import numpy as np
import io, time, threading, os, json
from PIL import Image
from http.server import HTTPServer, SimpleHTTPRequestHandler
import logging
import paramiko

# ── Serve viewer.html via mini HTTP server on port 7863 ───────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Cluster config (node0 runs loop.sh + vLLM) ────────────────────────────────
CLUSTER_HOST = os.getenv("CLUSTER_HOST", "10.137.203.228")
CLUSTER_USER = "nvidia"
CLUSTER_PASS = "nvidia"
CLUSTER_REPO = "/home/nvidia/autotherm"
BASELINE_RTH  = 2.28e-5

_state_cache = {"data": None, "ts": 0.0}
_state_lock  = threading.Lock()

def _fetch_results_tsv():
    """Read results.tsv from cluster node0 via paramiko; fall back to local copy."""
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(CLUSTER_HOST, username=CLUSTER_USER, password=CLUSTER_PASS, timeout=5)
        _, out, _ = c.exec_command(f"cat {CLUSTER_REPO}/results.tsv")
        txt = out.read().decode("utf-8", errors="replace")
        c.close()
        return txt, True
    except Exception:
        local = os.path.join(_HERE, "results.tsv")
        if os.path.exists(local):
            with open(local) as f:
                return f.read(), False
        return "", False

def _build_state():
    content, live = _fetch_results_tsv()
    rows = []
    for line in content.splitlines():
        if not line.strip() or line.startswith("commit"):
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        rth_str, status = parts[1], parts[3]
        desc = parts[4].strip() if len(parts) > 4 else ""
        try:    rth = float(rth_str)
        except: rth = None
        rows.append({"rth": rth, "status": status, "desc": desc})

    hist = [{"iter": 0, "rth": BASELINE_RTH, "status": "keep",
              "desc": "BASELINE — Disordered CNT network (human guess)"}]
    running_best = BASELINE_RTH
    for i, r in enumerate(rows):
        it = i + 1
        if r["status"] == "keep" and r["rth"] is not None:
            running_best = min(running_best, r["rth"])
        hist.append({"iter": it, "rth": r["rth"], "status": r["status"], "desc": r["desc"]})

    alignment = max(0.0, min(1.0, (BASELINE_RTH - running_best) / BASELINE_RTH))
    n = len(rows)
    best_iter = max((i+1 for i,r in enumerate(rows) if r["status"]=="keep" and r["rth"] is not None),
                   default=0)
    return {
        "from_cluster":    live,
        "running":         live and n < 50,
        "current_iter":    n,
        "total_iters":     max(50, n + 1),
        "best_iter":       best_iter,
        "baseline_rth":    BASELINE_RTH,
        "best_rth":        running_best,
        "alignment_order": round(alignment, 4),
        "rth_history":     hist,
    }

def get_state():
    """Return cached cluster state, refreshing every 5 s."""
    with _state_lock:
        now = time.time()
        if _state_cache["data"] is None or now - _state_cache["ts"] > 5:
            try:
                _state_cache["data"] = _build_state()
            except Exception as e:
                if _state_cache["data"] is None:
                    _state_cache["data"] = {"error": str(e), "current_iter": 0, "rth_history": []}
            _state_cache["ts"] = now
        return _state_cache["data"]

class _SilentHandler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=_HERE, **kw)
    def log_message(self, *_): pass
    def do_GET(self):
        if self.path.startswith("/state.json"):
            data = json.dumps(get_state()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        else:
            super().do_GET()

def _start_viewer_server():
    try:
        srv = HTTPServer(('localhost', 7863), _SilentHandler)
        srv.serve_forever()
    except OSError:
        pass  # already bound

threading.Thread(target=_start_viewer_server, daemon=True).start()

# ── Palette ────────────────────────────────────────────────────────────────────
BG    = '#05101C'
PANEL = '#070E18'
CARD  = '#0B1928'
BORD  = '#1E4060'
CYAN  = '#00D8FF'
GREEN = '#00FF88'
GOLD  = '#D4A020'
WHITE = '#D8EEFF'
MUTED = '#6090B8'
DIM   = '#2A4060'
RED   = '#FF3A3A'
ORNG  = '#FF8C00'
LBLUE = '#4090C8'   # nanotube base colour

THERMAL = LinearSegmentedColormap.from_list('th',
    ['#0022EE','#0088FF','#00FFCC','#FFCC00','#FF4400','#FF0000'])

ANG = np.radians(30)
COS30, SIN30 = np.cos(ANG), np.sin(ANG)

def proj(pts):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return np.column_stack([(x-y)*COS30, (x+y)*SIN30 + z*0.90])

# ── Simulation data ────────────────────────────────────────────────────────────
KEY = [(0, 2.28e-5, 0.0), (1, 7.42e-6, 0.12),
       (30, 5.49e-6, 0.65), (49, 4.55e-6, 1.0)]
BASELINE_R = 2.28e-5
BEST  = {1, 30, 49}
CRASH = {8}
SAMP  = {0:2.28e-5, 1:7.42e-6, 8:2.10e-5, 15:8.91e-6,
         22:6.83e-6, 30:5.49e-6, 38:5.71e-6, 44:5.12e-6, 49:4.55e-6}

LOG_ENTRIES = [
    (0,  "ITER #0 — BASELINE",
         "metal=60%, filler=25%, binder=15%  ε_mf=3.50",
         "Disordered CNT network. High phonon scattering at random interfaces."),
    (1,  "ITER #1  ★ NEW BEST",
         "ε_mf → 6.0   filler → 30%",
         "Strengthening cross-interactions. Percolation network begins forming."),
    (8,  "ITER #8  ⚠ CRASH",
         "T=300 K experiment",
         "grompp(eq) failed — topology diverged. Reverting temperature to 350 K."),
    (15, "ITER #15",
         "σ(metal-filler)=0.310   ε_mf → 6.5",
         "Atomic size matching improved. Recovery from crash confirmed."),
    (22, "ITER #22",
         "ε_mf=7.0  ε_mb=5.5  ε_fb=4.0",
         "Maximum cross-coupling test. Regime nearing saturation."),
    (30, "ITER #30  ★ NEW BEST",
         "metal=53%  filler=32%  binder=15%",
         "Hexagonal percolation network confirmed. 26% improvement on previous best."),
    (38, "ITER #38",
         "SIM_TIME → 200 ps  (same composition)",
         "Extended simulation for better phonon statistics. Minor regression in noise."),
    (44, "ITER #44",
         "ε_mf=7.0  ε_mb=6.0  ε_fb=5.0",
         "Maximum cross-coupling confirmed. Structure converging toward optimum."),
    (49, "ITER #49  ★ OPTIMAL",
         "metal=53%  filler=32%  ε_mf=7.0  ε_mb=6.0",
         "Self-assembled hexagonal CNT layers. 5× improvement over baseline."),
]

def get_params(it):
    for i in range(len(KEY)-1):
        i0,r0,o0 = KEY[i]; i1,r1,o1 = KEY[i+1]
        if i0 <= it <= i1:
            t = (it-i0)/(i1-i0) if i1>i0 else 0.0
            t = t*t*(3-2*t)
            return r0+t*(r1-r0), o0+t*(o1-o0)
    return KEY[-1][1], KEY[-1][2]

def visible_log(iteration):
    return [e for e in LOG_ENTRIES if e[0] <= iteration][-5:]

# ── Glass cube ──────────────────────────────────────────────────────────────────
CORNERS = (np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                      [0,0,1],[1,0,1],[1,1,1],[0,1,1]], float) - 0.5)
CUBE_FACES = [
    ([3,2,6,7], '#081830', 0.35),
    ([0,3,7,4], '#0A1E38', 0.30),
    ([0,1,2,3], '#060A14', 0.25),
    ([0,1,5,4], '#0B1C34', 0.38),
    ([1,2,6,5], '#0D2038', 0.32),
    ([4,5,6,7], '#0E1A30', 0.42),
]
CUBE_EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def draw_cube(ax, cx, cy, S, ec, ea=0.75):
    c = CORNERS*S; p = proj(c)
    px, py = p[:,0]+cx, p[:,1]+cy
    depth = [np.mean(c[idx,1]-c[idx,2]) for idx,_,_ in CUBE_FACES]
    for i in np.argsort(depth)[::-1]:
        idx,fc,a = CUBE_FACES[i]
        ax.fill([px[j] for j in idx],[py[j] for j in idx],color=fc,alpha=a,zorder=2)
    for i,j in CUBE_EDGES:
        ax.plot([px[i],px[j]],[py[i],py[j]],color=ec,alpha=ea,lw=1.8,zorder=25)

def glow(ax, xs, ys, col, lw=10, n=9, z=6, a=1.0):
    for i in range(n, 0, -1):
        f = i/n
        ax.plot(xs,ys,color=col,lw=lw*f,alpha=0.10*f*a,
                solid_capstyle='round',zorder=z)
    ax.plot(xs,ys,color='white',lw=lw*0.14,alpha=0.75*a,
            solid_capstyle='round',zorder=z+1)

# ── 3-D nanotube cylinder ───────────────────────────────────────────────────────
def draw_tube(ax, cx, cy, start3, end3, R, col_hex, alpha_base, S):
    """
    Render a 3D-looking nanotube cylinder using ring-polygon silhouette.
    start3, end3: 3D endpoints (scaled to S).
    R: tube radius (in same units).
    """
    p1, p2 = np.array(start3, float), np.array(end3, float)
    d = p2 - p1
    L = np.linalg.norm(d)
    if L < 1e-8:
        return
    dh = d / L

    # Two perpendicular axes for the tube cross-section
    up = np.array([0,0,1]) if abs(dh[2]) < 0.88 else np.array([0,1,0])
    r1 = np.cross(dh, up);  r1 /= np.linalg.norm(r1)
    r2 = np.cross(dh, r1)

    N = 20
    angs = np.linspace(0, 2*np.pi, N, endpoint=False)

    def ring_2d(center):
        pts3 = np.array([center + R*(np.cos(a)*r1 + np.sin(a)*r2) for a in angs])
        p2d = proj(pts3); p2d[:,0] += cx; p2d[:,1] += cy
        return p2d

    rng1 = ring_2d(p1)
    rng2 = ring_2d(p2)

    # Z-order by depth of tube midpoint
    mid = (p1+p2)/2
    z = 7.0 + (mid[1]-mid[2]) / (2*L+6*R)

    # Silhouette edges: top and bottom extremes of each ring in 2D-y
    i1t = int(np.argmax(rng1[:,1]));  i1b = int(np.argmin(rng1[:,1]))
    i2t = int(np.argmax(rng2[:,1]));  i2b = int(np.argmin(rng2[:,1]))

    # ── Main body (trapezoid / parallelogram silhouette) ──────────────────────
    rgb = np.array(mcolors.to_rgb(col_hex))
    body4 = np.array([rng1[i1t], rng2[i2t], rng2[i2b], rng1[i1b]])
    ax.add_patch(Polygon(body4, color=rgb*0.78, alpha=alpha_base*0.92, zorder=z))

    # Bright highlight strip (top 20% of body height)
    top_strip = np.array([
        rng1[i1t],
        rng2[i2t],
        rng2[(i2t+2)%N],
        rng1[(i1t+2)%N],
    ])
    ax.add_patch(Polygon(top_strip, color=np.minimum(rgb*1.5+0.2, 1.0),
                          alpha=alpha_base*0.45, zorder=z+0.15))

    # Shadow strip (bottom 20%)
    bot_strip = np.array([
        rng1[i1b],
        rng2[i2b],
        rng2[(i2b-2)%N],
        rng1[(i1b-2)%N],
    ])
    ax.add_patch(Polygon(bot_strip, color=np.maximum(rgb*0.3, 0),
                          alpha=alpha_base*0.55, zorder=z+0.15))

    # Top edge line (sharp specular highlight)
    ax.plot([rng1[i1t,0],rng2[i2t,0]], [rng1[i1t,1],rng2[i2t,1]],
            color='white', lw=1.6, alpha=alpha_base*0.60,
            solid_capstyle='round', zorder=z+0.6)
    # Bottom edge
    ax.plot([rng1[i1b,0],rng2[i2b,0]], [rng1[i1b,1],rng2[i2b,1]],
            color=tuple(rgb*0.4), lw=0.9, alpha=alpha_base*0.55, zorder=z+0.1)

    # ── End caps ──────────────────────────────────────────────────────────────
    d1_depth = p1[1]-p1[2];  d2_depth = p2[1]-p2[2]
    # Draw back cap first, front cap on top
    for ring2d, _ in sorted([(rng1,d1_depth),(rng2,d2_depth)], key=lambda x: x[1]):
        cap_ctr = ring2d.mean(axis=0)
        # Outer ring fill
        ax.add_patch(Polygon(ring2d, color=rgb*0.85,
                              alpha=alpha_base*0.82, zorder=z+0.3))
        # Inner darker hollow
        inner = cap_ctr + (ring2d - cap_ctr)*0.42
        ax.add_patch(Polygon(inner, color=rgb*0.22,
                              alpha=alpha_base*0.95, zorder=z+0.4))
        # Thin rim
        ax.plot(list(ring2d[:,0])+[ring2d[0,0]],
                list(ring2d[:,1])+[ring2d[0,1]],
                color=tuple(np.minimum(rgb+0.15, 1.0)),
                lw=0.8, alpha=alpha_base*0.70, zorder=z+0.5)


# ── Baseline nanotube layout (random / hazardous) ──────────────────────────────
# Each entry: (center_frac[3], direction[3], length_frac, radius_frac)
BASELINE_TUBES = [
    ([+0.18, +0.10, +0.08], [+0.80, +0.30, +0.50], 0.72, 0.058),
    ([-0.12, -0.22, +0.02], [+0.25, +0.75, +0.35], 0.68, 0.055),
    ([+0.05, +0.28, -0.25], [+0.60, -0.15, +0.78], 0.70, 0.060),
    ([-0.28, +0.08, +0.22], [-0.40, +0.65, +0.65], 0.66, 0.056),
    ([+0.32, -0.18, -0.10], [+0.70, +0.55, -0.45], 0.74, 0.062),
    ([+0.00, +0.02, +0.28], [-0.20, +0.92, +0.33], 0.65, 0.054),
    ([-0.30, -0.08, -0.20], [+0.55, -0.52, +0.65], 0.69, 0.057),
    ([+0.22, +0.30, -0.12], [-0.78, +0.10, +0.62], 0.71, 0.059),
    ([-0.08, -0.28, +0.18], [+0.42, +0.80, -0.42], 0.67, 0.055),
    ([+0.28, +0.02, +0.32], [+0.88, -0.28, +0.38], 0.64, 0.056),
    ([-0.14, +0.24, -0.28], [-0.58, +0.40, +0.72], 0.68, 0.058),
    ([+0.10, -0.12, -0.18], [+0.18, +0.60, +0.78], 0.70, 0.060),
    ([-0.22, +0.20, +0.05], [+0.65, -0.70, +0.30], 0.63, 0.054),
    ([+0.15, -0.30, +0.20], [-0.35, +0.85, -0.40], 0.66, 0.057),
]

def draw_structure(ax, cx, cy, order, S=5.2):
    # ── Cube shell ─────────────────────────────────────────────────────────────
    ec = '#{:02x}{:02x}{:02x}'.format(
        max(0, int((0.10-order*0.09)*255)),
        min(255, int((0.38+order*0.62)*255)),
        min(255, int((0.52+order*0.48)*255)),
    )
    draw_cube(ax, cx, cy, S, ec, ea=0.65+order*0.28)

    # ── BASELINE elements — fade out as order increases ────────────────────────
    fade_out = max(0.0, 1.0 - order*1.25)

    if fade_out > 0.02:
        # --- Scattered InGa droplets ---
        rng = np.random.RandomState(11)
        pts = np.clip((rng.rand(28,3)-0.5)*S*0.74, -0.40*S, 0.40*S)
        dep = pts[:,1]-pts[:,2]
        for i in np.argsort(dep):
            d  = (dep[i]+S*1.2)/(S*2.4)
            pp = proj(pts[i:i+1])[0]
            r  = 0.062+0.032*d
            a  = (0.65+0.28*d)*fade_out
            ax.add_patch(Circle((pp[0]+cx,pp[1]+cy), r, color='#C8A020', alpha=a, zorder=8+d))
            ax.add_patch(Circle((pp[0]+cx-r*.28,pp[1]+cy+r*.28), r*.30,
                                color='#FFE860', alpha=a*.55, zorder=8+d+.05))

        # --- Random nanotubes ---
        for cfrac, dfrac, Lfrac, Rfrac in BASELINE_TUBES:
            ctr   = np.array(cfrac)*S
            # clip center so tube never sticks out
            ctr   = np.clip(ctr, -0.38*S, 0.38*S)
            d3    = np.array(dfrac, float); d3 /= np.linalg.norm(d3)
            L     = Lfrac*S
            R     = Rfrac*S
            # Clip endpoints to cube interior
            half  = min(L/2, 0.41*S - R)
            p1    = np.clip(ctr - d3*half, -0.43*S, 0.43*S)
            p2    = np.clip(ctr + d3*half, -0.43*S, 0.43*S)
            draw_tube(ax, cx, cy, p1, p2, R, LBLUE, fade_out*0.92, S)

        # --- Chaotic heat paths (phonon scattering) ---
        rng2 = np.random.RandomState(33)
        for _ in range(9):
            t  = np.linspace(0,1,20)
            f1 = rng2.uniform(3,8); f2 = rng2.uniform(2,7)
            p1 = rng2.uniform(0,6); p2 = rng2.uniform(0,6)
            x3 = np.clip((np.linspace(-.43,.43,20)+.18*np.sin(f1*np.pi*t+p1))*S,-0.43*S,0.43*S)
            y3 = np.clip((rng2.uniform(-.30,.30)+.18*np.cos(f2*np.pi*t+p2))*S,-0.43*S,0.43*S)
            z3 = np.linspace(-.43,.43,20)*S
            pp = proj(np.column_stack([x3,y3,z3])); pp[:,0]+=cx; pp[:,1]+=cy
            heat = (z3/S+0.5)
            for j in range(19):
                glow(ax,[pp[j,0],pp[j+1,0]],[pp[j,1],pp[j+1,1]],
                     THERMAL((heat[j]+heat[j+1])/2),lw=7,n=6,z=6,a=fade_out*0.95)

    # ── OPTIMISED elements — fade in as order increases ────────────────────────
    fade_in = min(1.0, max(0.0, order))

    if fade_in > 0.05:
        # Aligned nanotubes in horizontal layers
        n_layers = min(6, max(1, int(order*7)))
        layer_zs = np.linspace(-0.41, 0.41, 6)*S

        for li in range(n_layers):
            lz   = layer_zs[li]
            heat = li/5.0
            la   = min(1.0, fade_in*1.3) if li < n_layers-1 else min(1.0, (order*7-li))
            la   = max(0.0, la)

            tube_col_rgb = np.array(THERMAL(heat*0.85+0.08)[:3])
            tube_col_hex = '#{:02x}{:02x}{:02x}'.format(
                int(tube_col_rgb[0]*255), int(tube_col_rgb[1]*255), int(tube_col_rgb[2]*255))

            # 3 parallel tubes in each layer (varying Y positions)
            for ti, ty_frac in enumerate([-0.28, 0.0, +0.28]):
                ty  = ty_frac*S
                R   = 0.072*S
                p1  = np.array([-0.42*S, ty, lz])
                p2  = np.array([+0.42*S, ty, lz])
                draw_tube(ax, cx, cy, p1, p2, R, tube_col_hex, la*0.94, S)

            # Heat channels flowing through layer
            ch_a = min(1.0, max(0, (order-0.18)/0.55))*la
            if ch_a > 0.02:
                for hi in range(4):
                    yh = (-0.32+hi*0.22)*S
                    x3 = np.linspace(-0.41,0.41,24)*S
                    pp = proj(np.column_stack([x3,np.full(24,yh),np.full(24,lz+0.005*S)]))
                    pp[:,0]+=cx; pp[:,1]+=cy
                    glow(ax,pp[:,0],pp[:,1],THERMAL(heat),lw=12,n=10,z=9,a=ch_a)

        # InGa droplets between layers
        rng3 = np.random.RandomState(55)
        n_drops = int(fade_in*40)
        nl = max(2, n_layers)
        lzm = np.linspace(-0.41,0.41,nl)*S
        for _ in range(n_drops):
            li  = rng3.randint(0,max(1,nl-1))
            zm  = (lzm[li]+lzm[min(li+1,nl-1)])/2 + rng3.uniform(-0.012,0.012)*S
            xm  = rng3.uniform(-0.37,0.37)*S
            ym  = rng3.uniform(-0.37,0.37)*S
            pp  = proj(np.array([[xm,ym,zm]]))[0]
            d   = 0.40+0.40*(ym+S*.5)/S
            r   = 0.050+0.022*d
            a   = (0.72+0.22*d)*fade_in
            ax.add_patch(Circle((pp[0]+cx,pp[1]+cy),r,color=GOLD,alpha=a,zorder=8+d))
            ax.add_patch(Circle((pp[0]+cx-r*.27,pp[1]+cy+r*.27),r*.30,
                                color='#FFE870',alpha=a*.50,zorder=8+d+.05))

# ── Thermal colour-scale bar ────────────────────────────────────────────────────
def draw_colorbar(ax, x, y, h, w=0.28):
    N = 80
    for i in range(N):
        ax.add_patch(plt.Rectangle((x, y+i*h/N), w, h/N,
                                    color=THERMAL(1-i/N), zorder=3))
    ax.text(x+w/2, y+h+0.22, 'High', color=WHITE, fontsize=8.5,
            ha='center', fontfamily='monospace', va='bottom', fontweight='bold')
    ax.text(x+w/2, y-0.22, 'Low',  color=WHITE, fontsize=8.5,
            ha='center', fontfamily='monospace', va='top',    fontweight='bold')

# ── Fake telemetry ──────────────────────────────────────────────────────────────
_rtel = np.random.RandomState(99)
_GPU  = np.clip(84 + _rtel.randn(5500)*4.0, 35, 100)
_CPU  = np.clip(60 + _rtel.randn(5500)*7.0, 15, 100)
for _s in [350, 720, 1440, 2160, 2880, 3600]:
    _GPU[_s:_s+70] = np.clip(_GPU[_s:_s+70]*0.30, 8, 100)
    _CPU[_s:_s+70] = np.clip(_CPU[_s:_s+70]*1.35, 0, 100)

def tel_slice(it):
    o = it*110
    return _GPU[o:o+5000:10], _CPU[o:o+5000:10]

def draw_telemetry(ax, x0, y0, w, h, data, col, label):
    ax.add_patch(FancyBboxPatch((x0,y0),w,h,
                 boxstyle='round,pad=0.06',
                 facecolor='#050D18',edgecolor=BORD,lw=0.8,zorder=1))
    ax.text(x0+0.14,y0+h-0.06,label,
            color=col,fontsize=7.0,fontweight='bold',
            fontfamily='monospace',va='top')
    # Plot
    px0,py0,pw,ph = x0+0.36,y0+0.18,w-0.50,h-0.35
    xs = np.linspace(px0,px0+pw,len(data))
    ys = py0+(data/100)*ph
    ax.fill_between(xs,py0,ys,color=col,alpha=0.22,zorder=2)
    ax.plot(xs,ys,color=col,lw=0.9,alpha=0.90,zorder=3)
    ax.plot([px0,px0+pw],[py0+ph*0.50,py0+ph*0.50],
            color=BORD,lw=0.5,linestyle='--',zorder=2)
    ax.text(px0-0.06,py0+ph,'100%',color=MUTED,fontsize=6.0,
            ha='right',va='top',fontfamily='monospace')
    ax.text(px0-0.06,py0+ph*.5,'50%',color=MUTED,fontsize=6.0,
            ha='right',va='center',fontfamily='monospace')
    ax.text(px0-0.06,py0,'0%',color=MUTED,fontsize=6.0,
            ha='right',va='bottom',fontfamily='monospace')

# ── Text wrap ───────────────────────────────────────────────────────────────────
def wrap(txt, w):
    words = txt.split(); lines, cur = [], ''
    for word in words:
        if len(cur)+len(word)+1 <= w: cur = (cur+' '+word).strip()
        else:
            if cur: lines.append(cur)
            cur = word
    if cur: lines.append(cur)
    return lines[:3]

# ── LEFT PANEL ──────────────────────────────────────────────────────────────────
def draw_left(ax, iteration, r_th):
    X0,Y0,W,H = 0.20, 0.20, 5.55, 12.60

    ax.add_patch(FancyBboxPatch((X0,Y0),W,H,
                 boxstyle='round,pad=0.12',
                 facecolor=PANEL,edgecolor=BORD,lw=1.4,zorder=1))

    # Section header
    ax.text(X0+0.20,Y0+H-0.14,
            'LLM (QWEN3.5-122B) REASONING LOG',
            color=CYAN,fontsize=8.5,fontweight='bold',
            fontfamily='monospace',va='top')
    ax.plot([X0+0.20,X0+W-0.20],[Y0+H-0.52,Y0+H-0.52],
            color=BORD,lw=0.8)

    # Log cards
    entries   = visible_log(iteration)
    card_h    = 2.10
    card_gap  = 0.10
    card_y0   = Y0+H-0.62

    for k, (it, head, sub1, sub2) in enumerate(reversed(entries[-5:])):
        ey  = card_y0 - k*(card_h+card_gap)
        cur = (it == entries[-1][0])
        best= it in BEST
        crash= it in CRASH

        if best:   card_col, head_col = '#003018', GREEN
        elif crash: card_col, head_col = '#1A0808', RED
        elif cur:  card_col, head_col = '#0A1A2C', CYAN
        else:      card_col, head_col = '#080F1C', MUTED

        # Card background
        ax.add_patch(FancyBboxPatch((X0+0.16,ey-card_h),W-0.32,card_h,
                     boxstyle='round,pad=0.08',
                     facecolor=card_col,edgecolor=head_col if (best or cur) else BORD,
                     lw=1.0 if (best or cur) else 0.6,zorder=2))

        # Header line
        ax.text(X0+0.30,ey-0.16,head,
                color=head_col,fontsize=8.5,fontweight='bold',
                fontfamily='monospace',va='top',zorder=3)
        # Sub-line 1
        for li,line in enumerate(wrap(sub1,50)):
            ax.text(X0+0.30,ey-0.56-li*0.32,line,
                    color=WHITE,fontsize=7.8,
                    fontfamily='monospace',va='top',zorder=3)
        # Sub-line 2
        for li,line in enumerate(wrap(sub2,52)):
            ax.text(X0+0.30,ey-1.10-li*0.32,line,
                    color=MUTED,fontsize=7.2,
                    fontfamily='monospace',va='top',zorder=3)

    # ── R_th sparkline ─────────────────────────────────────────────────────────
    ch  = 3.40
    cy0 = Y0+0.20
    cw  = W-0.32

    ax.add_patch(FancyBboxPatch((X0+0.16,cy0),cw,ch,
                 boxstyle='round,pad=0.08',
                 facecolor='#050C18',edgecolor=BORD,lw=0.8,zorder=2))
    ax.text(X0+0.28,cy0+ch-0.10,
            'BEST  THERMAL  RESISTANCE  (m²K/W)',
            color=CYAN,fontsize=7.5,fontweight='bold',
            fontfamily='monospace',va='top')

    r_min,r_max = 4e-6, 2.4e-5
    def sr(r): return cy0+ch*(1-(r-r_min)/(r_max-r_min))
    def si(i): return X0+0.32+(cw-0.24)*(i/49)

    for gf in [0.25,.50,.75,1.0]:
        gy = cy0+gf*ch
        ax.plot([X0+0.24,X0+0.24+cw-0.14],[gy,gy],color='#0E2030',lw=0.5,zorder=3)
        rv = r_max-gf*(r_max-r_min)
        ax.text(X0+0.22,gy,f'{rv:.0e}',color=MUTED,fontsize=6.5,
                ha='right',va='center',fontfamily='monospace')

    vis = sorted((i,r) for i,r in SAMP.items() if i<=iteration)
    if len(vis)>1:
        xs=[si(i) for i,r in vis]; ys=[sr(r) for i,r in vis]
        ax.plot(xs,ys,color=CYAN,lw=1.4,alpha=0.55,zorder=4)
        ax.fill_between(xs,[cy0]*len(xs),ys,color=CYAN,alpha=0.12,zorder=3)

    for i,r in vis:
        col = GREEN if i in BEST else (RED if i in CRASH else MUTED)
        ax.add_patch(Circle((si(i),sr(r)),0.060,color=col,zorder=5))
        if i in BEST:
            ax.text(si(i),sr(r)+0.20,f'{r:.1e}',color=GREEN,fontsize=6.0,
                    ha='center',fontfamily='monospace',zorder=6)

    # Current cursor
    rc,_ = get_params(iteration)
    ax.add_patch(Circle((si(iteration),sr(rc)),0.11,color=WHITE,zorder=7))
    ax.add_patch(Circle((si(iteration),sr(rc)),0.07,color=CYAN, zorder=8))

# ── RIGHT PANEL ─────────────────────────────────────────────────────────────────
def draw_right(ax, iteration, r_th):
    X0,Y0,W,H = 18.25, 0.20, 5.55, 12.60

    ax.add_patch(FancyBboxPatch((X0,Y0),W,H,
                 boxstyle='round,pad=0.12',
                 facecolor=PANEL,edgecolor=BORD,lw=1.4,zorder=1))

    ax.text(X0+0.20,Y0+H-0.14,
            'LLM  LIVE  METRICS',
            color=CYAN,fontsize=8.5,fontweight='bold',
            fontfamily='monospace',va='top')
    ax.plot([X0+0.20,X0+W-0.20],[Y0+H-0.52,Y0+H-0.52],
            color=BORD,lw=0.8)

    # Metric cards
    rc,_ = get_params(iteration)
    imp  = max(0,(BASELINE_R-rc)/BASELINE_R*100)
    speed= 42.0+iteration*0.28
    toks = 8192+iteration*380
    metrics = [
        ('TOKENS GENERATED', f'{toks:,}',     CYAN),
        ('GEN SPEED',         f'{speed:.1f} tok/s', GREEN),
        ('R_th IMPROVEMENT',  f'{imp:.1f}%',  GREEN if imp>50 else (CYAN if imp>20 else MUTED)),
    ]
    my = Y0+H-0.62
    for label, val, col in metrics:
        ax.add_patch(FancyBboxPatch((X0+0.16,my-1.65),W-0.32,1.52,
                     boxstyle='round,pad=0.08',
                     facecolor=CARD,edgecolor=BORD,lw=0.7,zorder=2))
        ax.text(X0+0.30,my-0.22,label,
                color=MUTED,fontsize=7.8,fontfamily='monospace',va='top')
        ax.text(X0+0.30,my-0.76,val,
                color=col,fontsize=13.5,fontweight='bold',
                fontfamily='monospace',va='top')
        my -= 1.75

    # Telemetry
    ty = my - 0.22
    ax.text(X0+0.20,ty,'CLUSTER  TELEMETRY',
            color=CYAN,fontsize=8.5,fontweight='bold',
            fontfamily='monospace',va='top')
    ax.plot([X0+0.20,X0+W-0.20],[ty-0.38,ty-0.38],color=BORD,lw=0.8)

    gpu_d, cpu_d = tel_slice(iteration)
    draw_telemetry(ax, X0+0.16, ty-4.60, W-0.32, 3.90,
                   gpu_d, GREEN,  'NODE 1-4  BLACKWELL GPU  (NCCL ACTIVE)')
    draw_telemetry(ax, X0+0.16, ty-9.20, W-0.32, 3.90,
                   cpu_d, '#30AAFF', 'NODE 1-4  GRACE CPU  (RERUN ACTIVE)')

# ── Full frame ──────────────────────────────────────────────────────────────────
def make_frame(iteration):
    r_th, order = get_params(iteration)
    improvement = max(0, (BASELINE_R-r_th)/BASELINE_R*100)
    is_best  = iteration in BEST
    is_crash = iteration in CRASH

    fig = plt.figure(figsize=(24,13.5), facecolor=BG, dpi=100)
    ax  = fig.add_axes([0,0,1,1], facecolor=BG)
    ax.set_xlim(0,24); ax.set_ylim(0,13.5)
    ax.set_aspect('equal'); ax.axis('off')

    # Outer border
    bc = GREEN if is_best else (RED if is_crash else '#162840')
    ax.add_patch(FancyBboxPatch((0.08,0.08),23.84,13.34,
                 boxstyle='round,pad=0.15',
                 linewidth=3.0 if is_best else 1.6,
                 edgecolor=bc,facecolor='none',zorder=0))

    # Header bar
    ax.add_patch(plt.Rectangle((0.10,12.62),23.80,0.80,color='#07101E',zorder=1))
    ax.plot([0.10,23.90],[12.62,12.62],color=BORD,lw=0.9,zorder=2)

    # NVIDIA badge
    ax.add_patch(FancyBboxPatch((0.22,12.68),3.00,0.68,
                 boxstyle='round,pad=0.07',facecolor='#76B900',edgecolor='none',zorder=3))
    ax.text(1.72,13.02,'NVIDIA  RESEARCH',color='white',fontsize=8.5,
            fontweight='bold',ha='center',va='center',fontfamily='monospace')

    # Title
    ax.text(12.0,13.26,
            'THE "AHA!" MOMENT: AI-DRIVEN TIM DISCOVERY',
            color=WHITE,fontsize=16.5,fontweight='bold',
            ha='center',va='top',fontfamily='monospace')

    # Date
    ax.text(23.75,13.22,'May 9, 2026',color=MUTED,fontsize=8.5,
            ha='right',va='top',fontfamily='monospace')
    ax.text(23.75,12.85,'DGX Spark Cluster',color=DIM,fontsize=7.5,
            ha='right',va='top',fontfamily='monospace')

    # Panels
    draw_left(ax, iteration, r_th)
    draw_right(ax, iteration, r_th)

    # ── Center: MOLECULAR VIEWER ───────────────────────────────────────────────
    CX = 12.0
    ax.text(CX,12.35,'MOLECULAR  VIEWER',
            color=WHITE,fontsize=10.0,fontweight='bold',
            ha='center',va='top',fontfamily='monospace')

    draw_structure(ax, CX, 7.0, order, S=5.2)
    draw_colorbar(ax, 15.65, 4.30, 5.30, w=0.28)

    # Bottom label
    if r_th < 5.5e-6:   rc2 = GREEN
    elif r_th < 9e-6:   rc2 = CYAN
    elif r_th < 1.6e-5: rc2 = ORNG
    else:               rc2 = RED

    ax.add_patch(FancyBboxPatch((6.20,0.28),11.60,1.58,
                 boxstyle='round,pad=0.12',
                 facecolor='#06101E',edgecolor=rc2,lw=2.0,zorder=3))

    if iteration == 0:
        lbl  = 'BASELINE DESIGN  (HUMAN GUESS)'
        desc = (f'R_th = {r_th:.3e} m²K/W   |   '
                f'Disordered CNT network — high phonon scattering at random interfaces')
    elif is_best:
        lbl  = f'AI-OPTIMIZED STRUCTURE  (ITERATION #{iteration})'
        desc = (f'R_th = {r_th:.3e} m²K/W   |   '
                f'{improvement:.1f}% improvement — aligned CNT layers, ordered heat pathways  ★')
    else:
        lbl  = f'ITERATION  #{iteration}  /  49'
        desc = (f'R_th = {r_th:.3e} m²K/W   |   '
                f'{improvement:.1f}% improvement vs baseline')

    ax.text(12.0,1.75,lbl,color=rc2,fontsize=12.5,fontweight='bold',
            ha='center',va='top',fontfamily='monospace')
    ax.text(12.0,1.18,desc,color=MUTED,fontsize=8.5,
            ha='center',va='top',fontfamily='monospace')

    # Progress bar
    ax.add_patch(FancyBboxPatch((6.20,0.12),11.60,0.14,
                 boxstyle='round,pad=0.02',facecolor=DIM,alpha=0.30,zorder=2))
    pw = 11.60*(iteration/49)
    ax.add_patch(FancyBboxPatch((6.20,0.12),max(0.05,pw),0.14,
                 boxstyle='round,pad=0.02',
                 facecolor=GREEN if is_best else CYAN,alpha=0.60,zorder=3))

    buf = io.BytesIO()
    fig.savefig(buf,format='png',bbox_inches='tight',dpi=100)
    buf.seek(0); plt.close(fig)
    return Image.open(buf)

# ── Cache + helpers ─────────────────────────────────────────────────────────────
_cache: dict = {}

def get_frame(it):
    if it not in _cache:
        _cache[it] = make_frame(it)
    return _cache[it]

def animate(pause_s):
    for it in range(50):
        r,_ = get_params(it)
        imp = max(0,(BASELINE_R-r)/BASELINE_R*100)
        badge = ' ★ NEW BEST' if it in BEST else (' ⚠ CRASH' if it in CRASH else '')
        st = (f"<span style='font-family:monospace;font-size:15px;color:#D8EEFF'>"
              f"<b>{'Baseline' if it==0 else f'Iteration {it}'}</b>{badge}"
              f" &nbsp;|&nbsp; R_th = <b style='color:#00D8FF'>{r:.3e}</b> m²K/W"
              f" &nbsp;|&nbsp; <b style='color:#00FF88'>{imp:.1f}%</b> improvement</span>")
        yield get_frame(it), st
        if it < 49: time.sleep(pause_s)

def jump_to(it):
    r,_ = get_params(it)
    imp = max(0,(BASELINE_R-r)/BASELINE_R*100)
    st = (f"<span style='font-family:monospace;font-size:15px;color:#D8EEFF'>"
          f"<b>{'Baseline' if it==0 else f'Iteration {it}'}</b>"
          f" &nbsp;|&nbsp; R_th = <b style='color:#00D8FF'>{r:.3e}</b> m²K/W"
          f" &nbsp;|&nbsp; <b style='color:#00FF88'>{imp:.1f}%</b> improvement</span>")
    return get_frame(it), st

# ── Gradio full-screen CSS ───────────────────────────────────────────────────────
CSS = """
*, body { box-sizing:border-box; margin:0; padding:0; }
html { height:100%; }
body {
    background:#05101C !important;
    min-height:100vh; overflow:hidden;
}
.gradio-container, .gradio-container > * {
    background:#05101C !important;
    max-width:100% !important;
    width:100% !important;
    min-width:100% !important;
    margin:0 !important;
    padding:0 !important;
    overflow:hidden;
}
/* Kill Gradio's default centering / max-width cap */
.contain { max-width:100% !important; }
.app { max-width:100% !important; padding:0 !important; }
/* Image fills full width, respects viewport height */
#img_out { width:100% !important; }
#img_out img {
    width:100% !important;
    max-height:calc(100vh - 108px) !important;
    object-fit:contain !important;
    display:block;
    background:#05101C;
}
/* Control bar */
#ctrl { gap:10px !important; padding:4px 0 2px !important; align-items:center; }
#ctrl button {
    background:#00D8FF !important; color:#05101C !important;
    font-weight:bold !important; font-family:monospace !important;
    border-radius:6px !important; font-size:14px !important;
    min-height:40px !important;
}
#ctrl .secondary { background:#0B1928 !important; color:#D8EEFF !important;
    border:1px solid #1E4060 !important; }
label span, .label-wrap { color:#6090B8 !important; font-family:monospace !important;
    font-size:12px !important; }
input[type=range] { accent-color:#00D8FF; }
footer, .built-with { display:none !important; }
"""

VIEWER_CSS = CSS + """
/* Tab styling */
.tabs { background:#04090F !important; }
.tab-nav button {
    background:#070E18 !important; color:#5090B8 !important;
    font-family:monospace !important; font-size:13px !important;
    border-bottom:2px solid transparent !important;
}
.tab-nav button.selected {
    color:#00D8FF !important;
    border-bottom:2px solid #00D8FF !important;
}
/* 3D viewer iframe — fills viewport minus the tab bar (~44px) */
#viewer3d { width:100% !important; }
#viewer3d iframe {
    width:100% !important;
    height:calc(100vh - 44px) !important;
    border:none !important;
    display:block;
}
"""

VIEWER_HTML = """
<iframe
  src="http://localhost:7863/viewer.html"
  style="width:100%;height:calc(100vh - 52px);border:none;display:block;background:#04090F;"
  allowfullscreen>
</iframe>
"""

with gr.Blocks(title="AutoTherm TIM Optimizer") as demo:
    with gr.Tabs():
        # ── Tab 1: Movie-quality 3D viewer ────────────────────────────────────
        with gr.Tab("3D Viewer  (WebGL)"):
            gr.HTML(VIEWER_HTML, elem_id="viewer3d")

        # ── Tab 2: Analytics dashboard (matplotlib) ───────────────────────────
        with gr.Tab("Analytics Dashboard"):
            img_out = gr.Image(label="", show_label=False, height=None, elem_id="img_out")
            status  = gr.HTML(
                "<span style='font-family:monospace;color:#2A4060'>"
                "Press ▶ Start to animate all 50 iterations, or drag the slider to jump</span>")
            with gr.Row(elem_id="ctrl"):
                start_btn = gr.Button("▶  Start Animation  (50 iterations)", variant="primary", scale=3)
                pause_sl  = gr.Slider(1,10,value=5,step=0.5,label="Pause per frame (s)",scale=2)
                iter_sl   = gr.Slider(0,49,value=0,step=1,label="Jump to iteration",scale=3)
                jump_btn  = gr.Button("Go", elem_classes=["secondary"], scale=1)

            start_btn.click(animate,  inputs=[pause_sl], outputs=[img_out,status])
            jump_btn.click( jump_to,  inputs=[iter_sl],  outputs=[img_out,status])
            iter_sl.release(jump_to,  inputs=[iter_sl],  outputs=[img_out,status])

if __name__ == "__main__":
    print("3D viewer  -> http://localhost:7863/viewer.html")
    print("Gradio app -> http://localhost:7862")
    demo.launch(server_port=7862, server_name="0.0.0.0", share=False,
                theme=gr.themes.Base(), css=VIEWER_CSS)
