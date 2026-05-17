# AutoTherm Research — Local Setup Guide

A WebUI that visualizes an LLM-driven Thermal Interface Material (TIM)
optimization run on a 4-node DGX Spark cluster, with live 3D charts for
the optimization curve, cluster scaling speedup, max-atoms capacity, and
GPU utilization across all nodes.

## Quick start

### 1. Prerequisites
- Python 3.10+
- A modern browser (Chrome 89+, Firefox 108+, Edge 89+)

### 2. Install dependencies

```bash
pip install gradio paramiko numpy matplotlib python-docx
```

### 3. Run the webapp

```bash
cd autotherm-research
python webapp.py
```

You should see:
```
3D viewer  -> http://localhost:7863/viewer.html
Gradio app -> http://localhost:7862
```

### 4. Open the dashboard

Open **http://localhost:7862** in your browser.

You'll see a "Connect" screen with two options:

- **Live Cluster** — connect to a running DGX Spark cluster via SSH (`paramiko`).
  Requires reachable IPs and the cluster's `autotherm-research` repo path.
- **Saved Run** — pick one of the bundled `state_*.json` files in `data/`
  to replay a previously-recorded optimization session.

For a quick demo with no cluster access, choose **Saved Run** → `4node run1`.

## Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│  webapp.py  ── Gradio app on :7862                                │
│              ── Static HTTP server on :7863 (serves HTML panels)  │
│              ── /state.json   live or saved iteration history     │
│              ── /gpu.json     live or saved per-node GPU samples  │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │ iframe http://localhost:7863/dashboard_grid.html
                                  ▼
   ┌──────────────────────────┬──────────────────────────────────┐
   │                          │  scaling_3d.html  (4-node speedup) │
   │      viewer.html         ├──────────────────────────────────┤
   │  (rotating cube + tubes) │  max_atoms_3d.html  (capacity)    │
   │                          ├──────────────────────────────────┤
   ├──────────────────────────┤  gpu_live_3d.html  (live GPUs)   │
   │   rth_graph.html         │                                  │
   └──────────────────────────┴──────────────────────────────────┘
```

### Files

- `webapp.py` — Gradio frontend + paramiko SSH backend + static file server
- `viewer.html` — Three.js cube + LLM reasoning panels (loaded with
  `?minimal=1` inside the dashboard to hide side panels)
- `dashboard_grid.html` — 5-panel CSS grid wrapper
- `scaling_3d.html` — Twin 3D ribbons, GROMACS scaling (4-node vs 1-node)
- `rth_graph.html` — 3D ribbon of R_th vs iteration (best curve)
- `max_atoms_3d.html` — 3D bar chart of max atoms 1-node vs 4-node
- `gpu_live_3d.html` — 4 parallel 3D ribbons, live GPU utilization
- `data/` — Saved state JSON files, scaling benchmark JSON, geometry
  files, raw `results.tsv` for offline preview
- `benchmark_scaling.py` — Run the GROMACS scaling benchmark on the
  cluster (1-node vs 3-node, N=1M to 10M+, PME GPU offload, UCX RDMA)
- `test_10M_1node.sh` — Standalone test for the 10M-atom single-node
  PME crash diagnosis (debugging tool, runs on cluster)

## Connecting to a live cluster

In the connect screen, choose **Live Cluster** and supply:

- **Head node IP** (e.g. `10.137.203.228`)
- **Autotherm repo path** (e.g. `/home/nvidia/autotherm`)

The webapp will:
1. SSH to the head node via `paramiko` (password authentication —
   credentials are pulled from environment / hardcoded fallback)
2. Read `results.tsv`, `nanotube_geometry_*.json` for the iteration history
3. Poll all 4 cluster nodes every 5 s for GPU utilization
4. Bundle GPU history into the in-memory state cache so a future
   "Save State" includes the multi-GPU data

## Dashboard interactions

- **Mouse over a panel** — auto-rotation pauses (cinematic camera stops
  so you can read the chart). Move out to resume.
- **Drag / scroll inside a panel** — manual orbit / zoom (Three.js
  OrbitControls). Camera resets to auto-rotate on mouseleave.
- **Double-click any of the 4 graphs** — opens an expanded fullscreen
  modal of that panel. Close with the X button, the dark backdrop, or
  the **ESC** key. (The cube panel intentionally has no expand action
  because it already contains its own controls.)

## Saving a run

When in live cluster mode, the in-memory state can be saved as
`data/state_<runname>.json`. It will include `gpu_per_node_history`
(the 5 s-cadence per-GPU samples from the polling loop). Future
"Saved Run" loads of that file replay the real GPU data instead of
falling back to mock data.

## Mock data fallback

If you run the webapp without a cluster connection AND haven't loaded
a saved state, the GPU panel falls back to a deterministic mock
generator (see `gpu_live_3d.html` → `generateMockSample`). The live
badge changes to **● DEMO** to make this clear.
