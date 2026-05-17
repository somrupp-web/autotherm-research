#!/usr/bin/env python3
"""
benchmark_scaling.py — GROMACS wall-clock scaling: 1-node vs 3-node

Multi-node config: 3 MPI ranks across nodes 1-3 (same as autotherm loop)
Single-node config: 1 MPI rank on node 1 only (same hardware, single GPU)

For each N in N_VALUES:
  1. Build system (gro/top/ndx) locally on node0
  2. Run em + short eq on 3-node (not benchmarked, prep only)
  3. Time mdrun(prod) 10ps on 3-node
  4. Time mdrun(prod) 10ps on 1-node  (same .tpr, fair comparison)
  5. Save (N, t_1node, t_3node) to scaling_results.json

Run from node0: python3 benchmark_scaling.py
"""

import os, sys, time, json, math, shutil, subprocess, tempfile

GMX    = "/usr/local/gromacs/bin/gmx_mpi"
MPIRUN = "mpirun"
# With GPU-resident execution (-update gpu -pme gpu -bonded gpu) only a few CPU
# OpenMP threads are needed; using all 20 thrashes the GPU offload path.
NTOMP  = "4"

HOSTFILE_3NODE = "/tmp/bench_hostfile_3node"
HOSTFILE_1NODE = "/tmp/bench_hostfile_1node"

# Worker nodes (nodes 1-3) — node0 excluded to avoid local-loop MPI issues
REMOTE_NODES = ["10.137.203.184", "10.137.203.174", "10.137.203.177"]
RANK0_NODE   = REMOTE_NODES[0]   # rank0 always lands here; pull outputs from here

N_VALUES  = [2000000, 3000000, 4000000]   # smoke test at N=1M passed; expanding
BENCH_PS  = 200.0     # matches production loop SIM_TIME_PS
DT        = 0.002
OUTPUT_JSON = "/home/nvidia/autotherm/scaling_results_v3_2to4M.json"

COMPOSITION = {'metal': 0.50, 'filler': 0.35, 'binder': 0.15}
# Charges chosen to make PME meaningful AND keep the system net-neutral:
#   0.35*(+1.2) + 0.15*(-2.80) = 0   (metal stays neutral at 0)
# PME activation is what generates the FFT halo-exchange traffic that
# benefits from RDMA over CX7.
LJ_PARAMS = {
    'metal':  {'sigma': 0.280, 'epsilon': 6.00, 'mass': 97.0, 'charge':  0.00},
    'filler': {'sigma': 0.340, 'epsilon': 5.00, 'mass': 12.0, 'charge':  1.20},
    'binder': {'sigma': 0.410, 'epsilon': 1.60, 'mass': 44.0, 'charge': -2.80},
}
CROSS = {
    ('metal', 'filler'): {'sigma': 0.310, 'epsilon': 8.00},
    ('metal', 'binder'): {'sigma': 0.345, 'epsilon': 8.00},
    ('filler','binder'): {'sigma': 0.375, 'epsilon': 8.00},
}
TEMPERATURE = 300.0

_COMPONENTS = list(COMPOSITION.keys())
_RESNAMES   = {'metal': 'MET', 'filler': 'FIL', 'binder': 'BND'}
_SHORT      = {'metal': 'MT',  'filler': 'FL',  'binder': 'BD'}

def log(msg):
    print(f"[bench] {msg}", flush=True)

def _run(cmd, workdir, timeout, label):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=workdir, capture_output=True, timeout=timeout)
    elapsed = time.time() - t0
    ok = "OK" if r.returncode == 0 else "FAIL"
    log(f"{label} {ok} ({elapsed:.1f}s)")
    if r.returncode != 0:
        log(f"  stderr tail: {r.stderr.decode(errors='replace')[-400:]}")
    return r.returncode, elapsed

def _write_hostfiles():
    with open(HOSTFILE_3NODE, "w") as f:
        for node in REMOTE_NODES:
            f.write(f"{node} slots=1\n")
    with open(HOSTFILE_1NODE, "w") as f:
        f.write(f"{REMOTE_NODES[0]} slots=1\n")
    log(f"Hostfiles written — 3-node: {REMOTE_NODES}, 1-node: [{REMOTE_NODES[0]}]")

def _sync_to_nodes(workdir, *files):
    for node in REMOTE_NODES:
        subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no",
                        f"nvidia@{node}", f"mkdir -p {workdir}"],
                       capture_output=True, timeout=10)
        for f in files:
            subprocess.run(["scp", "-q", f, f"nvidia@{node}:{workdir}/"],
                           capture_output=True, timeout=600)

def _pull_from_rank0(workdir, *files):
    # Force fsync on the remote so partial writes don't get pulled
    subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no",
                    f"nvidia@{RANK0_NODE}", "sync"],
                   capture_output=True, timeout=60)
    for fname in files:
        # Large gro files at N=4M can reach ~700MB → use generous timeout
        r = subprocess.run(["scp", "-q", "-o", "StrictHostKeyChecking=no",
                            f"nvidia@{RANK0_NODE}:{workdir}/{fname}",
                            os.path.join(workdir, fname)],
                           capture_output=True, timeout=600)
        local_size = os.path.getsize(os.path.join(workdir, fname)) if os.path.exists(os.path.join(workdir, fname)) else 0
        # Check remote size for sanity
        rs = subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no",
                             f"nvidia@{RANK0_NODE}",
                             f"stat -c %s {workdir}/{fname} 2>/dev/null || echo 0"],
                            capture_output=True, timeout=30)
        remote_size = int(rs.stdout.decode().strip() or 0)
        log(f"  pull {fname}: remote={remote_size:,}B  local={local_size:,}B  scp_rc={r.returncode}")

def _cleanup_remote(workdir):
    for node in REMOTE_NODES:
        subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no",
                        f"nvidia@{node}", f"rm -rf {workdir}"],
                       capture_output=True, timeout=10)

def _box_size(n_total, density=0.85):
    avg_sig = sum(LJ_PARAMS[c]['sigma'] * COMPOSITION[c] for c in _COMPONENTS)
    vol = n_total * (avg_sig ** 3) / density
    return vol ** (1/3)

def _write_top(path, n_total):
    lines = [
        "[ defaults ]",
        "; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ",
        "1         2          yes        1.0      1.0", "",
        "[ atomtypes ]",
        "; name  at.num  mass  charge  ptype  sigma  epsilon",
    ]
    for c in _COMPONENTS:
        p = LJ_PARAMS[c]
        lines.append(f"  {c:<8} 1  {p['mass']:.2f}  {p['charge']:.3f}  A  {p['sigma']:.4f}  {p['epsilon']:.4f}")
    lines += ["", "[ nonbond_params ]"]
    for (c1, c2), p in CROSS.items():
        lines.append(f"  {c1}  {c2}  1  {p['sigma']:.4f}  {p['epsilon']:.4f}")
    for c in _COMPONENTS:
        n = max(1, int(round(n_total * COMPOSITION[c])))
        lines += [
            "", "[ moleculetype ]",
            "; name  nrexcl", f"{_RESNAMES[c]}  1", "",
            "[ atoms ]",
            "; nr  type  resnr  residue  atom  cgnr  charge  mass",
            f"  1   {c}   1     {_RESNAMES[c]}   {_SHORT[c]}   1    {LJ_PARAMS[c]['charge']:.3f}    {LJ_PARAMS[c]['mass']:.2f}",
        ]
    lines += ["", "[ system ]", "TIM", "", "[ molecules ]"]
    for c in _COMPONENTS:
        n = max(1, int(round(n_total * COMPOSITION[c])))
        lines.append(f"{_RESNAMES[c]}  {n}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

def _write_gro(path, n_total):
    box_l = _box_size(n_total)
    atoms = []
    for c in _COMPONENTS:
        n = max(1, int(round(n_total * COMPOSITION[c])))
        atoms.extend([c] * n)
    n_actual = len(atoms)
    n_side = math.ceil(n_actual ** (1/3))
    spacing = box_l / n_side
    with open(path, "w") as f:
        f.write(f"TIM System\n{n_actual}\n")
        for idx, c in enumerate(atoms):
            ix = idx % n_side
            iy = (idx // n_side) % n_side
            iz = idx // (n_side * n_side)
            x = (ix + 0.5) * spacing
            y = (iy + 0.5) * spacing
            z = (iz + 0.5) * spacing
            aid = (idx + 1) % 100000
            f.write(f"{aid:5d}{_RESNAMES[c]:<5}{_SHORT[c]:<5}{aid:5d}"
                    f"{x:8.3f}{y:8.3f}{z:8.3f}\n")
        f.write(f"   {box_l:.5f}   {box_l:.5f}   {box_l:.5f}\n")
    return n_actual

def _write_ndx(path, n_total):
    groups = {c: [] for c in _COMPONENTS}
    offset = 1
    for c in _COMPONENTS:
        n = max(1, int(round(n_total * COMPOSITION[c])))
        groups[c] = list(range(offset, offset + n))
        offset += n
    all_atoms = list(range(1, offset))
    with open(path, "w") as f:
        for name, indices in groups.items():
            f.write(f"[ {name} ]\n")
            for i, idx in enumerate(indices):
                f.write(f"{idx:9d}")
                if (i + 1) % 15 == 0:
                    f.write("\n")
            f.write("\n")
        f.write("[ System ]\n")
        for i, idx in enumerate(all_atoms):
            f.write(f"{idx:9d}")
            if (i + 1) % 15 == 0:
                f.write("\n")
        f.write("\n")

def _write_mdp_em(path, nsteps=5000):
    with open(path, "w") as f:
        f.write(f"""integrator    = steep
nsteps        = {nsteps}
emtol         = 10.0
emstep        = 0.01
nstlog        = 500
nstenergy     = 100
cutoff-scheme = Verlet
rlist         = 1.0
rcoulomb      = 1.0
rvdw          = 1.0
coulombtype   = PME
vdwtype       = cutoff
fourierspacing = 0.16
pme-order     = 4
pbc           = xyz
""")

def _write_mdp_nvt(path, nsteps, dt=0.002, save_xtc=False, annealing_start=None, ref_t_K=None):
    t_groups = " ".join(_COMPONENTS)
    ref_t_val = ref_t_K if ref_t_K is not None else int(TEMPERATURE)
    ref_t    = " ".join([str(ref_t_val)] * len(_COMPONENTS))
    tau_t    = " ".join(["0.1"] * len(_COMPONENTS))
    nstenergy = max(100, nsteps // 100)
    nstxout   = max(100, nsteps // 50) if save_xtc else 0
    lines = [
        "integrator     = md",
        f"nsteps         = {nsteps}",
        f"dt             = {dt}",
        "nstlog         = 2500",
        f"nstenergy      = {nstenergy}",
        f"nstxout        = {nstxout}",
        "nstvout        = 0",
        "cutoff-scheme  = Verlet",
        "rlist          = 1.0",
        "rcoulomb       = 1.0",
        "rvdw           = 1.0",
        "coulombtype    = PME",
        "vdwtype        = cutoff",
        "fourierspacing = 0.16",
        "pme-order      = 4",
        "pbc            = xyz",
        "tcoupl         = V-rescale",
        f"tc-grps        = {t_groups}",
        f"ref_t          = {ref_t}",
        f"tau_t          = {tau_t}",
        "pcoupl         = no",
    ]
    if annealing_start is not None:
        t_end = nsteps * dt
        n_grp = len(_COMPONENTS)
        lines += [
            "annealing         = " + " ".join(["single"] * n_grp),
            "annealing-npoints = " + " ".join(["2"] * n_grp),
            "annealing-time    = " + " ".join([f"0.0 {t_end:.3f}"] * n_grp),
            "annealing-temp    = " + " ".join([f"{annealing_start:.0f} {int(TEMPERATURE)}"] * n_grp),
        ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

def _grompp(tpr, gro, top, mdp, ndx, workdir):
    cmd = [GMX, "grompp", "-f", mdp, "-c", gro, "-p", top,
           "-n", ndx, "-o", tpr, "-maxwarn", "5"]
    r = subprocess.run(cmd, cwd=workdir, capture_output=True, timeout=120)
    if r.returncode != 0:
        err = r.stderr.decode(errors='replace')
        out = r.stdout.decode(errors='replace')
        log(f"  grompp STDOUT first 2000: {out[:2000]}")
        log(f"  grompp STDERR first 2000: {err[:2000]}")
        log(f"  grompp STDERR last  1500: {err[-1500:]}")
    return r.returncode

def _mpirun_em(deffnm, workdir, hostfile, n_ranks, timeout, label, tpr=None):
    """Multi-rank mdrun for energy minimization.

    em uses steepest descent — -update gpu / -dlb / -npme are invalid here.
    Only -nb gpu is allowed.
    """
    ldpath = os.environ.get("LD_LIBRARY_PATH", "")
    tpr = tpr or f"{deffnm}.tpr"
    cmd = [
        MPIRUN, "-n", str(n_ranks),
        "--hostfile", hostfile,
        "--map-by", "node",
        # RDMA over CX7: rc_mlx5 = Mellanox-optimized; sm = shared mem (intra-node); tcp = fallback
        "-x", "UCX_TLS=rc_mlx5,rc_verbs,sm,self,tcp",
        "-x", "OMPI_MCA_pml=ucx",
        "-x", f"LD_LIBRARY_PATH={ldpath}",
        GMX, "mdrun", "-v",
        "-s", tpr,
        "-deffnm", deffnm,
        "-nb", "gpu",
        "-ntomp", NTOMP,
    ]
    return _run(cmd, workdir, timeout, label)

def _mpirun(deffnm, workdir, hostfile, n_ranks, timeout, label, tpr=None):
    """Multi-rank MD mdrun via UCX over CX7 RoCEv2 RDMA.

    UCX_TLS=rc_mlx5,rc_verbs,sm,self,tcp enables:
      rc_mlx5  = Mellanox-optimized Reliable Connected RDMA on the CX7 fabric
      rc_verbs = generic IB Verbs RDMA fallback
      sm,self  = intra-node shared memory + self-talk
      tcp      = final fallback

    GPUDirect (cuda_copy / gdr_copy) is unavailable in this UCX build,
    so GPU↔GPU data goes via pinned CPU buffers — still RDMA on the wire.

    With -npme 1, one rank does FFT (PME); the rest do PP. The PP↔PME
    halo exchange is what RDMA accelerates.

    -bonded cpu (not gpu) because the system has only single-atom molecules
    with no bonded interactions — GPU bonded kernel has nothing to compute.
    """
    ldpath = os.environ.get("LD_LIBRARY_PATH", "")
    tpr = tpr or f"{deffnm}.tpr"
    cmd = [
        MPIRUN, "-n", str(n_ranks),
        "--hostfile", hostfile,
        "--map-by", "node",
        "-x", "UCX_TLS=rc_mlx5,rc_verbs,sm,self,tcp",
        "-x", "OMPI_MCA_pml=ucx",
        "-x", f"LD_LIBRARY_PATH={ldpath}",
        GMX, "mdrun", "-v",
        "-s", tpr,
        "-deffnm", deffnm,
        "-nb",      "gpu",
        "-pme",     "gpu",
        "-bonded",  "cpu",
        "-update",  "gpu",
        "-npme",    "1",
        "-dlb",     "yes",
        "-ntomp",   NTOMP,
    ]
    return _run(cmd, workdir, timeout, label)

def _ssh_run_1node(deffnm, workdir, timeout, label):
    """Single-rank mdrun on node1 with full GPU residency (no MPI traffic).

    Same GPU offload flags as the 3-node case for a fair like-for-like comparison.
    No -npme/-dlb since they require ≥2 ranks.
    """
    ldpath = os.environ.get("LD_LIBRARY_PATH", "")
    remote_cmd = (
        f"cd {workdir} && "
        f"LD_LIBRARY_PATH={ldpath} "
        f"mpirun -n 1 --mca pml ob1 --mca btl self,tcp "
        f"-x LD_LIBRARY_PATH={ldpath} "
        f"{GMX} mdrun -v -s prod.tpr -deffnm {deffnm} "
        f"-nb gpu -pme gpu -bonded cpu -update gpu -gpu_id 0 -ntomp {NTOMP}"
    )
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
           f"nvidia@{REMOTE_NODES[0]}", remote_cmd]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    elapsed = time.time() - t0
    ok = "OK" if r.returncode == 0 else "FAIL"
    log(f"{label} {ok} ({elapsed:.1f}s)")
    if r.returncode != 0:
        combined = r.stdout.decode(errors='replace') + r.stderr.decode(errors='replace')
        log(f"  output tail: {combined[-800:]}")
    return r.returncode, elapsed

def benchmark_n(n_total):
    log(f"{'='*60}")
    log(f"N = {n_total:,}")
    workdir = tempfile.mkdtemp(prefix=f"bench_{n_total}_")
    log(f"Workdir: {workdir}")

    try:
        gro      = os.path.join(workdir, "system.gro")
        top      = os.path.join(workdir, "system.top")
        ndx      = os.path.join(workdir, "index.ndx")
        em_mdp   = os.path.join(workdir, "em.mdp")
        eq_mdp   = os.path.join(workdir, "eq.mdp")
        prod_mdp = os.path.join(workdir, "prod.mdp")

        n_actual = _write_gro(gro, n_total)
        _write_top(top, n_total)
        _write_ndx(ndx, n_total)
        log(f"System built: N={n_actual:,}, box={_box_size(n_total):.3f}nm")

        # ── EM on 3-node (prep, not benchmarked) ─────────────────────────────
        em_nsteps = min(50000, max(1000, n_total // 10))
        _write_mdp_em(em_mdp, nsteps=em_nsteps)
        em_tpr = os.path.join(workdir, "em.tpr")
        if _grompp(em_tpr, gro, top, em_mdp, ndx, workdir) != 0:
            log("grompp(em) FAILED"); return None
        log(f"grompp(em) OK")
        _sync_to_nodes(workdir, em_tpr)
        rc, _ = _mpirun_em("em", workdir, HOSTFILE_3NODE, 3, timeout=7200, label="mdrun(em)[3node]")
        if rc != 0:
            log("mdrun(em) FAILED — skipping N"); return None
        _pull_from_rank0(workdir, "em.gro")
        em_gro = os.path.join(workdir, "em.gro")
        if not os.path.exists(em_gro) or os.path.getsize(em_gro) < 1000:
            log(f"em.gro missing/truncated ({os.path.getsize(em_gro) if os.path.exists(em_gro) else 0} bytes) — skipping N")
            return None

        # ── Prewarm on 3-node (0.5ps at 10K, settles bad contacts before eq) ──
        prewarm_mdp = os.path.join(workdir, "prewarm.mdp")
        _write_mdp_nvt(prewarm_mdp, nsteps=500, dt=0.001, ref_t_K=10)
        prewarm_tpr = os.path.join(workdir, "prewarm.tpr")
        if _grompp(prewarm_tpr, em_gro, top, prewarm_mdp, ndx, workdir) != 0:
            log("grompp(prewarm) FAILED"); return None
        log("grompp(prewarm) OK")
        _sync_to_nodes(workdir, prewarm_tpr)
        rc, _ = _mpirun("prewarm", workdir, HOSTFILE_3NODE, 3, timeout=3600, label="mdrun(prewarm)[3node]")
        if rc != 0:
            log("mdrun(prewarm) FAILED — skipping N"); return None
        _pull_from_rank0(workdir, "prewarm.gro")
        prewarm_gro = os.path.join(workdir, "prewarm.gro")
        if not os.path.exists(prewarm_gro) or os.path.getsize(prewarm_gro) < 1000:
            log(f"prewarm.gro missing/truncated — skipping N"); return None

        # ── eq on 3-node (10ps, dt=0.001, annealing 10K→T) ────────────────────
        _write_mdp_nvt(eq_mdp, nsteps=10000, dt=0.001, annealing_start=10)
        eq_tpr = os.path.join(workdir, "eq.tpr")
        if _grompp(eq_tpr, prewarm_gro, top, eq_mdp, ndx, workdir) != 0:
            log("grompp(eq) FAILED"); return None
        log("grompp(eq) OK")
        _sync_to_nodes(workdir, eq_tpr)
        rc, _ = _mpirun("eq", workdir, HOSTFILE_3NODE, 3, timeout=14400, label="mdrun(eq)[3node]")
        if rc != 0:
            log("mdrun(eq) FAILED — skipping N"); return None
        _pull_from_rank0(workdir, "eq.gro")
        eq_gro = os.path.join(workdir, "eq.gro")
        if not os.path.exists(eq_gro) or os.path.getsize(eq_gro) < 1000:
            log("eq.gro missing/truncated — skipping N"); return None

        # ── Build prod.tpr (10ps, shared by both benchmark runs) ──────────────
        prod_nsteps = int(BENCH_PS / DT)
        _write_mdp_nvt(prod_mdp, nsteps=prod_nsteps, dt=DT, save_xtc=False)
        prod_tpr = os.path.join(workdir, "prod.tpr")
        if _grompp(prod_tpr, eq_gro, top, prod_mdp, ndx, workdir) != 0:
            log("grompp(prod) FAILED"); return None
        log(f"grompp(prod) OK — running {BENCH_PS}ps benchmarks")
        _sync_to_nodes(workdir, prod_tpr)

        # ── Benchmark run 1: 3-node ────────────────────────────────────────────
        rc3, t_3node = _mpirun("prod3", workdir, HOSTFILE_3NODE, 3,
                                timeout=36000, label=f"BENCH 3-node N={n_total:,}",
                                tpr="prod.tpr")
        t_3node = t_3node if rc3 == 0 else None

        # ── Benchmark run 2: 1-node (SSH directly to node1, no MPI transport) ──
        rc1, t_1node = _ssh_run_1node("prod1", workdir, timeout=36000,
                                      label=f"BENCH 1-node N={n_total:,}")
        t_1node = t_1node if rc1 == 0 else None

        speedup = round(t_1node / t_3node, 3) if (t_1node and t_3node) else None
        result = {
            "n_total":   n_total,
            "bench_ps":  BENCH_PS,
            "t_3node_s": round(t_3node, 2) if t_3node else None,
            "t_1node_s": round(t_1node, 2) if t_1node else None,
            "speedup":   speedup,
        }
        t3s = f"{t_3node:.1f}s" if t_3node else "FAIL"
        t1s = f"{t_1node:.1f}s" if t_1node else "FAIL"
        log(f"RESULT N={n_total:,}: 3-node={t3s}  1-node={t1s}  speedup={speedup}x")
        return result

    except Exception as e:
        log(f"Exception at N={n_total}: {e}")
        return None
    finally:
        # Keep workdir for debugging — only clean up if we got a successful result
        try:
            if 't_3node' in locals() and t_3node is not None and 't_1node' in locals() and t_1node is not None:
                _cleanup_remote(workdir)
                shutil.rmtree(workdir, ignore_errors=True)
                log(f"Cleaned up workdir: {workdir}")
            else:
                log(f"PRESERVED workdir for debug: {workdir}")
        except Exception:
            log(f"PRESERVED workdir for debug: {workdir}")


def main():
    log("GROMACS scaling benchmark: 1-node vs 3-node cluster")
    log(f"N values: {N_VALUES}")
    log(f"Bench prod duration: {BENCH_PS}ps per config")
    _write_hostfiles()

    results = []
    for n in N_VALUES:
        r = benchmark_n(n)
        if r:
            results.append(r)
            with open(OUTPUT_JSON, "w") as f:
                json.dump(results, f, indent=2)
            log(f"Progress saved ({len(results)}/{len(N_VALUES)} N values done)")

    log("=" * 60)
    log("BENCHMARK COMPLETE")
    log(f"{'N':>10}  {'3-node':>10}  {'1-node':>10}  {'speedup':>8}")
    for r in results:
        t3 = f"{r['t_3node_s']:.1f}s" if r['t_3node_s'] else "FAIL"
        t1 = f"{r['t_1node_s']:.1f}s" if r['t_1node_s'] else "FAIL"
        sp = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"
        log(f"{r['n_total']:>10,}  {t3:>10}  {t1:>10}  {sp:>8}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    log(f"Final results → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
