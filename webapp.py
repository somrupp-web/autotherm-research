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
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import logging
import paramiko

# ── Serve viewer.html via mini HTTP server on port 7863 ───────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

# ── Cluster scaling image (base64 embedded) ───────────────────────────────────
import base64 as _b64
_CLUSTER_SCALING_IMG = ""
_img_path = os.path.join(_HERE, "cluster_scaling.jpeg")
if os.path.exists(_img_path):
    with open(_img_path, "rb") as _f:
        _CLUSTER_SCALING_IMG = _b64.b64encode(_f.read()).decode()

# ── Cluster config (node0 runs loop.sh + vLLM) ────────────────────────────────
CLUSTER_HOST = os.getenv("CLUSTER_HOST", "10.137.203.228")
CLUSTER_USER = "nvidia"
CLUSTER_PASS = "nvidia"
CLUSTER_REPO = "/home/nvidia/autotherm"       # default; overridden by _ui_cluster_repo
BASELINE_RTH  = 2.28e-5

_state_cache = {"data": None, "ts": 0.0}
_state_lock  = threading.Lock()

# ── UI-controlled data source ─────────────────────────────────────────────────
_ui_lock        = threading.Lock()
_ui_mode        = "live"          # "live" | "saved"
_ui_cluster_ip  = CLUSTER_HOST    # editable via UI
_ui_saved_file  = None            # absolute path to selected state_*.json

_UNSET = object()  # sentinel so None can mean "clear"

_ui_cluster_repo = CLUSTER_REPO   # which experiment dir to read on the cluster

def _set_ui_mode(mode, ip=None, saved_path=_UNSET, repo=None):
    global _ui_mode, _ui_cluster_ip, _ui_saved_file, _ui_cluster_repo
    with _ui_lock:
        _ui_mode = mode
        if ip:                       _ui_cluster_ip   = ip.strip()
        if repo:                     _ui_cluster_repo = repo.strip()
        if saved_path is not _UNSET: _ui_saved_file   = saved_path

def _list_saved_runs():
    """Return list of (label, path) for all state_*.json files in _HERE."""
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(_HERE, "state_*.json")))
    results = []
    for p in files:
        label = os.path.basename(p).replace("state_", "").replace(".json", "").replace("_", " ")
        results.append((label, p))
    return results

# ── Real-time GPU telemetry from compute nodes ────────────────────────────────
COMPUTE_NODES_GPU = [
    ('spark-0c01', '10.137.203.184'),
    ('spark-1aa0', '10.137.203.174'),
    ('spark-1b93', '10.137.203.177'),
]
_gpu_history     = []   # rolling list of avg GPU util % floats
_gpu_lock        = threading.Lock()
GPU_HISTORY_MAX  = 360  # 1 hour at 10 s intervals

def _query_gpu_util(ip):
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(ip, username=CLUSTER_USER, password=CLUSTER_PASS, timeout=5)
        _, out, _ = c.exec_command(
            'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null',
            timeout=8)
        r = out.read().decode('utf-8', errors='replace').strip()
        c.close()
        vals = [int(x.strip()) for x in r.splitlines() if x.strip().isdigit()]
        return sum(vals) / len(vals) if vals else None
    except Exception:
        return None

def _gpu_poll_loop():
    while True:
        results, lock2 = [], threading.Lock()
        def _query(ip):
            v = _query_gpu_util(ip)
            if v is not None:
                with lock2: results.append(v)
        threads = [threading.Thread(target=_query, args=(ip,), daemon=True)
                   for _, ip in COMPUTE_NODES_GPU]
        for t in threads: t.start()
        for t in threads: t.join(timeout=10)
        if results:
            avg = round(sum(results) / len(results), 1)
            with _gpu_lock:
                _gpu_history.append(avg)
                if len(_gpu_history) > GPU_HISTORY_MAX:
                    del _gpu_history[0]
        time.sleep(10)

threading.Thread(target=_gpu_poll_loop, daemon=True).start()

def _ssh_read(client, path):
    """Read a remote file via an open paramiko client. Returns '' on missing."""
    _, out, _ = client.exec_command(f"cat {path} 2>/dev/null")
    return out.read().decode("utf-8", errors="replace").strip()

def _fetch_cluster_data():
    """Fetch results.tsv + nanotube geometry JSONs from cluster node0."""
    with _ui_lock:
        host = _ui_cluster_ip
        repo = _ui_cluster_repo
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(host, username=CLUSTER_USER, password=CLUSTER_PASS, timeout=5)
        results   = _ssh_read(c, f"{repo}/results.tsv")
        geom_best = _ssh_read(c, f"{repo}/nanotube_geometry_best.json")
        geom_base = _ssh_read(c, f"{repo}/nanotube_geometry_baseline.json")
        c.close()
        return results, geom_best, geom_base, True
    except Exception:
        results = ""
        local = os.path.join(_HERE, "results.tsv")
        if os.path.exists(local):
            with open(local) as f:
                results = f.read()
        geom_best = ""
        geom_base = ""
        for fname, var in [("nanotube_geometry_best.json", "best"),
                           ("nanotube_geometry_baseline.json", "base")]:
            p = os.path.join(_HERE, fname)
            if os.path.exists(p):
                with open(p) as f:
                    if var == "best":  geom_best = f.read().strip()
                    else:              geom_base = f.read().strip()
        return results, geom_best, geom_base, False

def _build_state():
    content, geom_best_raw, geom_base_raw, live = _fetch_cluster_data()
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
        # Extract explicit iter number from description (e.g. "iter-44:")
        import re as _re
        m = _re.search(r'iter-(\d+)', desc)
        iter_num = int(m.group(1)) if m else None
        rows.append({"rth": rth, "status": status, "desc": desc, "iter_num": iter_num})

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
    # Use the highest explicit iter number from descriptions (more reliable than row count,
    # since crashes sometimes skip writing a row — n can be < 100 even when loop finished)
    max_iter_seen = max((r["iter_num"] for r in rows if r["iter_num"] is not None), default=n)
    best_iter = max((i+1 for i,r in enumerate(rows) if r["status"]=="keep" and r["rth"] is not None),
                   default=0)

    # Parse nanotube geometry JSONs (None if not yet available)
    def _parse_geom(raw):
        try:
            return json.loads(raw) if raw else None
        except Exception:
            return None

    with _gpu_lock:
        gpu_tel = list(_gpu_history)

    return {
        "from_cluster":          live,
        "running":               live and max_iter_seen < 100,
        "current_iter":          n,
        "total_iters":           max(100, max_iter_seen + 1),
        "best_iter":             best_iter,
        "baseline_rth":          BASELINE_RTH,
        "best_rth":              running_best,
        "alignment_order":       round(alignment, 4),
        "rth_history":           hist,
        "nanotube_geometry_best":     _parse_geom(geom_best_raw),
        "nanotube_geometry_baseline": _parse_geom(geom_base_raw),
        "gpu_telemetry":         gpu_tel,
    }

def get_state():
    """Return cached state — checks UI mode first, then falls back to live cluster cache."""
    with _ui_lock:
        mode, saved = _ui_mode, _ui_saved_file
    if mode == "saved" and saved and os.path.exists(saved):
        with open(saved, encoding="utf-8") as f:
            return json.load(f)
    with _state_lock:
        return _state_cache["data"] or {"current_iter": 0, "rth_history": [], "running": False}

def _state_refresh_loop():
    """Background thread: refresh cluster state every 5 s without blocking HTTP."""
    while True:
        try:
            data = _build_state()
            with _state_lock:
                _state_cache["data"] = data
                _state_cache["ts"] = time.time()
        except Exception:
            pass
        time.sleep(5)

threading.Thread(target=_state_refresh_loop, daemon=True).start()

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
        elif self.path.startswith("/debug.json"):
            with _ui_lock:
                info = {"ui_mode": _ui_mode, "ui_cluster_ip": _ui_cluster_ip,
                        "ui_saved_file": _ui_saved_file}
            with _state_lock:
                info["cache_is_none"] = _state_cache["data"] is None
                if _state_cache["data"]:
                    info["cache_from_cluster"] = _state_cache["data"].get("from_cluster")
                    info["cache_current_iter"] = _state_cache["data"].get("current_iter")
            data = json.dumps(info, indent=2).encode()
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
        srv = ThreadingHTTPServer(('localhost', 7863), _SilentHandler)
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

# ── Live simulation data (derived from cluster state) ─────────────────────────
BASELINE_R = BASELINE_RTH  # alias used throughout drawing code

def _live_data():
    """Derive samp/best_set/crash_set/log_entries from cached cluster state."""
    state = get_state()
    hist  = state.get('rth_history', [])

    samp            = {}
    best_set        = set()
    crash_set       = set()
    log_entries     = []
    running_best_at = {}
    running_best    = BASELINE_RTH

    for entry in hist:
        it     = entry['iter']
        rth    = entry.get('rth')
        status = entry.get('status', '')
        desc   = entry.get('desc', '').strip()

        if rth is not None:
            samp[it] = rth

        if status == 'keep' and rth is not None and rth < running_best:
            running_best = rth
            best_set.add(it)

        running_best_at[it] = running_best

        if status in ('crash', 'error', 'fail'):
            crash_set.add(it)

        if it == 0:
            head = 'ITER #0 — BASELINE'
        elif it in best_set:
            head = f'ITER #{it}  ★ NEW BEST'
        elif it in crash_set:
            head = f'ITER #{it}  ⚠ CRASH'
        else:
            head = f'ITER #{it}'

        sub1 = desc[:80] if desc else '—'
        log_entries.append((it, head, sub1, ''))

    return {
        'samp':            samp,
        'best_set':        best_set,
        'crash_set':       crash_set,
        'log_entries':     log_entries,
        'running_best_at': running_best_at,
    }


def get_params(it):
    """Return (r_th, alignment_order) at iteration it from live data."""
    data = _live_data()
    rba  = data['running_best_at']
    if not rba:
        return BASELINE_RTH, 0.0
    valid = [k for k in rba if k <= it]
    r_th  = rba[max(valid)] if valid else BASELINE_RTH
    order = max(0.0, min(1.0, (BASELINE_RTH - r_th) / BASELINE_RTH))
    return r_th, order


def visible_log(iteration):
    entries = _live_data()['log_entries']
    return [e for e in entries if e[0] <= iteration][-5:]

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
def draw_left(ax, iteration, r_th, max_iter=99):
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

    ld = _live_data()
    for k, (it, head, sub1, sub2) in enumerate(reversed(entries[-5:])):
        ey  = card_y0 - k*(card_h+card_gap)
        cur = (it == entries[-1][0])
        best  = it in ld['best_set']
        crash = it in ld['crash_set']

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
    def si(i): return X0+0.32+(cw-0.24)*(i/max_iter)

    for gf in [0.25,.50,.75,1.0]:
        gy = cy0+gf*ch
        ax.plot([X0+0.24,X0+0.24+cw-0.14],[gy,gy],color='#0E2030',lw=0.5,zorder=3)
        rv = r_max-gf*(r_max-r_min)
        ax.text(X0+0.22,gy,f'{rv:.0e}',color=MUTED,fontsize=6.5,
                ha='right',va='center',fontfamily='monospace')

    vis = sorted((i,r) for i,r in ld['samp'].items() if i<=iteration)
    if len(vis)>1:
        xs=[si(i) for i,r in vis]; ys=[sr(r) for i,r in vis]
        ax.plot(xs,ys,color=CYAN,lw=1.4,alpha=0.55,zorder=4)
        ax.fill_between(xs,[cy0]*len(xs),ys,color=CYAN,alpha=0.12,zorder=3)

    for i,r in vis:
        col = GREEN if i in ld['best_set'] else (RED if i in ld['crash_set'] else MUTED)
        ax.add_patch(Circle((si(i),sr(r)),0.060,color=col,zorder=5))
        if i in ld['best_set']:
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
    state    = get_state()
    max_iter = max(1, state.get('total_iters', 100) - 1)
    r_th, order = get_params(iteration)
    improvement = max(0, (BASELINE_R-r_th)/BASELINE_R*100)
    ld = _live_data()
    is_best  = iteration in ld['best_set']
    is_crash = iteration in ld['crash_set']

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
    draw_left(ax, iteration, r_th, max_iter=max_iter)
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
        lbl  = f'ITERATION  #{iteration}  /  {max_iter}'
        desc = (f'R_th = {r_th:.3e} m²K/W   |   '
                f'{improvement:.1f}% improvement vs baseline')

    ax.text(12.0,1.75,lbl,color=rc2,fontsize=12.5,fontweight='bold',
            ha='center',va='top',fontfamily='monospace')
    ax.text(12.0,1.18,desc,color=MUTED,fontsize=8.5,
            ha='center',va='top',fontfamily='monospace')

    # Progress bar
    ax.add_patch(FancyBboxPatch((6.20,0.12),11.60,0.14,
                 boxstyle='round,pad=0.02',facecolor=DIM,alpha=0.30,zorder=2))
    pw = 11.60*(iteration/max_iter)
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
    _cache.clear()  # always regenerate so max_iter from live state is current
    _cache[it] = make_frame(it)
    return _cache[it]

def animate(pause_s):
    total = get_state().get('total_iters', 100)
    for it in range(total):
        r,_ = get_params(it)
        imp = max(0,(BASELINE_R-r)/BASELINE_R*100)
        ld    = _live_data()
        badge = ' ★ NEW BEST' if it in ld['best_set'] else (' ⚠ CRASH' if it in ld['crash_set'] else '')
        st = (f"<span style='font-family:monospace;font-size:15px;color:#D8EEFF'>"
              f"<b>{'Baseline' if it==0 else f'Iteration {it}'}</b>{badge}"
              f" &nbsp;|&nbsp; R_th = <b style='color:#00D8FF'>{r:.3e}</b> m²K/W"
              f" &nbsp;|&nbsp; <b style='color:#00FF88'>{imp:.1f}%</b> improvement</span>")
        yield get_frame(it), st
        if it < total-1: time.sleep(pause_s)

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
.tab-nav, .tab-nav * { overflow:visible !important; }
.tabs > div:first-child { overflow-x:auto !important; }
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
/* Connect screen — bigger fonts for radio, textbox, buttons */
#connect_screen .wrap { gap:16px !important; }
#connect_screen input[type=text], #connect_screen input[type=email] {
    font-size:18px !important; font-weight:600 !important;
    height:52px !important; padding:0 14px !important;
}
#connect_screen button {
    font-size:18px !important; font-weight:700 !important;
    min-height:52px !important;
}
#connect_screen .gradio-radio label span {
    font-size:20px !important; font-weight:700 !important;
    color:#D8EEFF !important;
}
#connect_screen .gradio-radio input[type=radio] {
    width:20px !important; height:20px !important;
}
#connect_screen select, #connect_screen .gradio-dropdown {
    font-size:18px !important; font-weight:600 !important;
}
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
/* Cluster scaling overlay screen */
#cluster_scaling_screen { background:#05101C !important; }
#cluster_close_row { padding:6px 12px !important; align-items:center !important; }
#cluster_close_row button {
    background:#1A0808 !important; color:#FF6060 !important;
    border:1px solid #FF3A3A !important; font-family:monospace !important;
    font-weight:bold !important; min-height:36px !important;
}
/* Cluster Scaling button in topbar */
#cluster_scale_btn button {
    background:#0B2040 !important; color:#00D8FF !important;
    border:1px solid #1E4060 !important; font-family:monospace !important;
    font-weight:bold !important;
}
"""

VIEWER_HTML = """
<iframe
  src="http://localhost:7863/viewer.html"
  style="width:100%;height:calc(100vh - 52px);border:none;display:block;background:#04090F;"
  allowfullscreen>
</iframe>
"""

def _saved_run_labels():
    return [label for label, _ in _list_saved_runs()] or ["(no saved runs found)"]

_REPO_MAP = {
    "1-node run  (currently running)": "/home/nvidia/autotherm_1node",
    "4-node run  (finished, 89 iters)": "/home/nvidia/autotherm",
}

def _connect_live(ip, repo_choice):
    ip = (ip or "").strip()
    if not ip:
        return "<span style='color:#FF3A3A;font-family:monospace'>⚠ Enter a cluster IP first</span>"
    repo = _REPO_MAP.get(repo_choice, "/home/nvidia/autotherm_1node")
    try:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(ip, username=CLUSTER_USER, password=CLUSTER_PASS, timeout=6)
        _, out, _ = c.exec_command(f"hostname && ls {repo}/results.tsv 2>/dev/null && echo OK")
        info = out.read().decode("utf-8", errors="replace").strip()
        c.close()
        _set_ui_mode("live", ip=ip, repo=repo)
        short = repo.split("/")[-1]
        return (f"<span style='color:#00FF88;font-family:monospace'>"
                f"✔ Connected to {ip} ({short})"
                + (f" — {info.splitlines()[0]}" if info else "") +
                f"</span>")
    except Exception as e:
        return f"<span style='color:#FF3A3A;font-family:monospace'>✘ {ip}: {e}</span>"

def _preview_saved(label):
    """Show info about a saved run WITHOUT setting the mode — just for the dropdown preview."""
    runs = {lbl: path for lbl, path in _list_saved_runs()}
    path = runs.get(label)
    if not path:
        return "<span style='color:#FF3A3A;font-family:monospace'>⚠ Run not found</span>"
    try:
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        note = s.get("_note", "")
        best = s.get("best_rth", "?")
        niters = s.get("current_iter", "?")
        return (f"<span style='color:#00D8FF;font-family:monospace'>"
                f"{niters} iterations · best R_th = {float(best):.3e} m²K/W"
                + (f"<br>{note}" if note else "") +
                f"</span>")
    except Exception as e:
        return f"<span style='color:#FF3A3A;font-family:monospace'>⚠ {e}</span>"

def _load_saved(label):
    """Actually commit to saved mode — only called when Load button is clicked."""
    runs = {lbl: path for lbl, path in _list_saved_runs()}
    path = runs.get(label)
    if not path:
        return "<span style='color:#FF3A3A;font-family:monospace'>⚠ Run not found</span>"
    _set_ui_mode("saved", saved_path=path)
    return _preview_saved(label).replace("font-family:monospace'>",
                                         "font-family:monospace'>✔ Loaded: " + label + "<br>")

CONNECT_HTML = """
<div style="text-align:center; padding:56px 0 36px; font-family:monospace;">
  <div style="color:#00D8FF; font-size:42px; font-weight:900; margin-bottom:10px; letter-spacing:3px;">
    AutoTherm TIM Optimizer
  </div>
  <div style="color:#6090B8; font-size:18px; font-weight:600;">
    AI-driven Thermal Interface Material research on DGX Spark
  </div>
</div>
"""

with gr.Blocks(title="AutoTherm TIM Optimizer") as demo:

    # ── Screen 1: Connection / source selection ────────────────────────────────
    with gr.Column(visible=True, elem_id="connect_screen") as connect_screen:
        gr.HTML(CONNECT_HTML)

        with gr.Column(elem_id="connect_card"):
            gr.HTML("<div style='color:#D8EEFF;font-size:22px;font-weight:800;"
                    "margin-bottom:20px;letter-spacing:1px;'>Select Data Source</div>")

            mode_radio = gr.Radio(
                choices=["Live Cluster", "Saved Run"],
                value="Live Cluster",
                label="",
                interactive=True,
                elem_id="mode_radio",
            )

            # Live cluster inputs
            with gr.Column(visible=True, elem_id="live_inputs") as live_inputs:
                gr.HTML("<div style='color:#6090B8;font-size:16px;font-weight:600;margin:12px 0 6px;'>"
                        "Cluster node IP:</div>")
                with gr.Row():
                    ip_input = gr.Textbox(
                        value=CLUSTER_HOST,
                        label="",
                        placeholder="e.g. 10.137.203.228",
                        scale=5,
                        container=False,
                    )
                    connect_btn = gr.Button("Connect", variant="primary", scale=2)
                gr.HTML("<div style='color:#6090B8;font-size:16px;font-weight:600;margin:12px 0 6px;'>"
                        "Experiment to monitor:</div>")
                repo_radio = gr.Radio(
                    choices=["1-node run  (currently running)",
                             "4-node run  (finished, 89 iters)"],
                    value="1-node run  (currently running)",
                    label="",
                    interactive=True,
                )
                conn_status = gr.HTML(
                    "<span style='color:#2A4060;font-size:15px;font-weight:600;'>Enter IP and click Connect</span>")

            # Saved run inputs
            with gr.Column(visible=False, elem_id="saved_inputs") as saved_inputs:
                gr.HTML("<div style='color:#6090B8;font-size:16px;font-weight:600;margin:12px 0 6px;'>"
                        "Choose a saved run snapshot:</div>")
                saved_dd = gr.Dropdown(
                    choices=_saved_run_labels(),
                    label="",
                    interactive=True,
                    container=False,
                )
                saved_status = gr.HTML(
                    "<span style='color:#2A4060;font-size:15px;font-weight:600;'>Select a run above</span>")
                load_btn = gr.Button("Load Saved Run", variant="primary")

    # ── Screen 2: Main dashboard ───────────────────────────────────────────────
    with gr.Column(visible=False) as dashboard_screen:
        # Back button strip
        with gr.Row(elem_id="topbar"):
            src_label = gr.HTML(
                "<span style='font-family:monospace;color:#6090B8;font-size:14px;font-weight:700;"
                "padding:6px 12px;'>Source: —</span>")
            cluster_scale_btn = gr.Button("Cluster Scaling", scale=0, size="sm",
                                          elem_id="cluster_scale_btn")
            back_btn = gr.Button("Change Source", scale=0, size="sm",
                                 elem_classes=["secondary"])

        with gr.Tabs():
            with gr.Tab("3D Viewer  (WebGL)"):
                gr.HTML(VIEWER_HTML, elem_id="viewer3d")

            with gr.Tab("Analytics Dashboard"):
                img_out = gr.Image(label="", show_label=False, height=None, elem_id="img_out")
                status  = gr.HTML(
                    "<span style='font-family:monospace;color:#2A4060'>"
                    "Press Start to animate all iterations, or drag the slider to jump</span>")
                with gr.Row(elem_id="ctrl"):
                    start_btn = gr.Button("Start Animation", variant="primary", scale=3)
                    pause_sl  = gr.Slider(1,10,value=5,step=0.5,label="Pause per frame (s)",scale=2)
                    iter_sl   = gr.Slider(0,99,value=0,step=1,label="Jump to iteration",scale=3)
                    jump_btn  = gr.Button("Go", elem_classes=["secondary"], scale=1)

                start_btn.click(animate,  inputs=[pause_sl], outputs=[img_out,status])
                jump_btn.click( jump_to,  inputs=[iter_sl],  outputs=[img_out,status])
                iter_sl.release(jump_to,  inputs=[iter_sl],  outputs=[img_out,status])

    # ── Screen 3: Cluster Scaling image overlay ───────────────────────────────
    with gr.Column(visible=False, elem_id="cluster_scaling_screen") as cluster_screen:
        with gr.Row(elem_id="cluster_close_row"):
            gr.HTML("<span style='font-family:monospace;color:#00D8FF;font-size:16px;"
                    "font-weight:bold;padding:6px 0;'>"
                    "4-Node DGX Spark (GB10) Cluster: Enabling Production-Scale GROMACS for TIMs"
                    "</span>")
            cluster_close_btn = gr.Button("✕  Cancel", scale=0, size="sm",
                                          elem_id="cluster_close_row")
        gr.HTML(f"""
        <div style="width:100%;text-align:center;padding:4px 16px 16px;background:#05101C;">
          <img src="data:image/jpeg;base64,{_CLUSTER_SCALING_IMG}"
               style="max-width:99%;max-height:calc(100vh - 80px);
                      object-fit:contain;border:2px solid #1E4060;
                      border-radius:8px;display:inline-block;">
        </div>
        """)

    # ── Callbacks ──────────────────────────────────────────────────────────────
    def _switch_mode(choice):
        if choice == "Live Cluster":
            # Clear saved file so get_state() can't accidentally return saved data
            _set_ui_mode("live", saved_path=None)
            return gr.update(visible=True), gr.update(visible=False)
        else:
            return gr.update(visible=False), gr.update(visible=True)

    mode_radio.change(_switch_mode, inputs=[mode_radio],
                      outputs=[live_inputs, saved_inputs])

    def _do_connect(ip, repo_choice):
        msg = _connect_live(ip, repo_choice)
        if "Connected" in msg:
            max_it = max(1, get_state().get('total_iters', 100) - 1)
            short = _REPO_MAP.get(repo_choice, repo_choice).split("/")[-1]
            label = (
                "<span style='font-family:monospace;font-size:14px;font-weight:700;"
                "padding:4px 12px;border-radius:4px;"
                "background:#003A18;color:#00FF88;border:1px solid #00CC66;'>"
                f"&#9679; LIVE &nbsp;—&nbsp; {ip.strip()} / {short}"
                "</span>"
            )
            return (msg,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(value=label),
                    gr.update(maximum=max_it))
        return msg, gr.update(), gr.update(), gr.update(), gr.update()

    connect_btn.click(_do_connect, inputs=[ip_input, repo_radio],
                      outputs=[conn_status, connect_screen, dashboard_screen, src_label, iter_sl])

    def _do_load(label):
        return _preview_saved(label)

    saved_dd.change(_do_load, inputs=[saved_dd], outputs=[saved_status])

    def _launch_saved(label):
        msg = _load_saved(label)
        if "Loaded" in msg:
            max_it = max(1, get_state().get('total_iters', 100) - 1)
            disp = label if label else "Saved Run"
            lbl = (
                "<span style='font-family:monospace;font-size:14px;font-weight:700;"
                "padding:4px 12px;border-radius:4px;"
                "background:#2A1500;color:#FFA040;border:1px solid #CC6600;'>"
                f"&#128190; SAVED &nbsp;—&nbsp; {disp}"
                "</span>"
            )
            return (msg,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(value=lbl),
                    gr.update(maximum=max_it))
        return msg, gr.update(), gr.update(), gr.update(), gr.update()

    load_btn.click(_launch_saved, inputs=[saved_dd],
                   outputs=[saved_status, connect_screen, dashboard_screen, src_label, iter_sl])

    back_btn.click(lambda: (gr.update(visible=True), gr.update(visible=False)),
                   outputs=[connect_screen, dashboard_screen])

    cluster_scale_btn.click(
        lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[dashboard_screen, cluster_screen])

    cluster_close_btn.click(
        lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[dashboard_screen, cluster_screen])

    def _init_slider():
        max_it = max(1, get_state().get('total_iters', 100) - 1)
        return gr.update(maximum=max_it)

    demo.load(_init_slider, outputs=[iter_sl])

if __name__ == "__main__":
    print("3D viewer  -> http://localhost:7863/viewer.html")
    print("Gradio app -> http://localhost:7862")
    demo.launch(server_port=7862, server_name="0.0.0.0", share=False,
                theme=gr.themes.Base(), css=VIEWER_CSS)
