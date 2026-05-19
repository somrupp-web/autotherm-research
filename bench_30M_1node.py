#!/usr/bin/env python3
"""
bench_30M_1node.py — Full N=30M MD pipeline on a SINGLE node (node1 / GPU 1).

Goal: validate whether one DGX Spark GB10 GPU can handle N=30M with PME GPU
offload, before any 3-node comparison run. The whole pipeline (em, prewarm,
eq, prod) runs as one MPI rank SSH'd to node1, with:
  - density 0.5 (softer initial overlap than the standard 0.85)
  - em nsteps 500_000 (was 50_000) so em can actually converge at this scale
  - emtol 100 kJ/mol/nm, emstep 0.005 (looser+careful)
  - -nb gpu -pme gpu -bonded cpu -update gpu (full GPU-resident)
"""

import os, sys, time, json, math, shutil, subprocess, tempfile

GMX     = "/usr/local/gromacs/bin/gmx_mpi"
MPIRUN  = "mpirun"
NTOMP   = "4"
NODE1   = "10.137.203.184"           # the single GPU node we run on

N_TOTAL    = 30_000_000
DENSITY    = 0.5                     # softer than 0.85 default
EM_NSTEPS  = 500_000
EM_TOL     = 100.0
EM_STEP    = 0.005
BENCH_PS   = 200.0
DT         = 0.002
OUTPUT_JSON = "/home/nvidia/autotherm/scaling_results_v3_30M_1node.json"

# ── TIM composition (same as scaling benchmark) ────────────────────────────
COMPOSITION = {'metal': 0.50, 'filler': 0.35, 'binder': 0.15}
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
    print(f"[bench-1n] {msg}", flush=True)

def _ssh_run(workdir, cmd_str, timeout, label):
    """Run a command on NODE1 via SSH inside the workdir. Returns (rc, elapsed)."""
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", f"nvidia@{NODE1}",
           f"cd {workdir} && {cmd_str}"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, timeout=timeout)
    elapsed = time.time() - t0
    ok = "OK" if r.returncode == 0 else "FAIL"
    log(f"{label} {ok} ({elapsed:.1f}s)")
    if r.returncode != 0:
        combined = r.stdout.decode(errors='replace') + r.stderr.decode(errors='replace')
        log(f"  tail: {combined[-800:]}")
    return r.returncode, elapsed

def _push(local_path, workdir):
    subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no",
                    f"nvidia@{NODE1}", f"mkdir -p {workdir}"],
                   capture_output=True, timeout=10)
    subprocess.run(["scp", "-q", local_path, f"nvidia@{NODE1}:{workdir}/"],
                   capture_output=True, timeout=900)

def _pull(workdir, fname, local_path):
    subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no",
                    f"nvidia@{NODE1}", "sync"], capture_output=True, timeout=30)
    subprocess.run(["scp", "-q", "-o", "StrictHostKeyChecking=no",
                    f"nvidia@{NODE1}:{workdir}/{fname}", local_path],
                   capture_output=True, timeout=1200)

# ── System builders (identical to benchmark_scaling.py) ──────────────────────
def _box_size(n_total, density=DENSITY):
    avg_sig = sum(LJ_PARAMS[c]['sigma'] * COMPOSITION[c] for c in _COMPONENTS)
    return (n_total * (avg_sig ** 3) / density) ** (1/3)

def _write_top(path, n_total):
    lines = ["[ defaults ]","; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ",
             "1         2          yes        1.0      1.0","",
             "[ atomtypes ]","; name  at.num  mass  charge  ptype  sigma  epsilon"]
    for c in _COMPONENTS:
        p = LJ_PARAMS[c]
        lines.append(f"  {c:<8} 1  {p['mass']:.2f}  {p['charge']:.3f}  A  {p['sigma']:.4f}  {p['epsilon']:.4f}")
    lines += ["", "[ nonbond_params ]"]
    for (c1, c2), p in CROSS.items():
        lines.append(f"  {c1}  {c2}  1  {p['sigma']:.4f}  {p['epsilon']:.4f}")
    for c in _COMPONENTS:
        n = max(1, int(round(n_total * COMPOSITION[c])))
        lines += ["", "[ moleculetype ]","; name  nrexcl", f"{_RESNAMES[c]}  1", "",
                  "[ atoms ]",
                  "; nr  type  resnr  residue  atom  cgnr  charge  mass",
                  f"  1   {c}   1     {_RESNAMES[c]}   {_SHORT[c]}   1    {LJ_PARAMS[c]['charge']:.3f}    {LJ_PARAMS[c]['mass']:.2f}"]
    lines += ["", "[ system ]", "TIM", "", "[ molecules ]"]
    for c in _COMPONENTS:
        n = max(1, int(round(n_total * COMPOSITION[c])))
        lines.append(f"{_RESNAMES[c]}  {n}")
    with open(path, "w") as f: f.write("\n".join(lines) + "\n")

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
            x = (ix + 0.5) * spacing; y = (iy + 0.5) * spacing; z = (iz + 0.5) * spacing
            aid = (idx + 1) % 100000
            f.write(f"{aid:5d}{_RESNAMES[c]:<5}{_SHORT[c]:<5}{aid:5d}{x:8.3f}{y:8.3f}{z:8.3f}\n")
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
                if (i + 1) % 15 == 0: f.write("\n")
            f.write("\n")
        f.write("[ System ]\n")
        for i, idx in enumerate(all_atoms):
            f.write(f"{idx:9d}")
            if (i + 1) % 15 == 0: f.write("\n")
        f.write("\n")

def _write_mdp_em(path, nsteps=EM_NSTEPS):
    with open(path, "w") as f:
        f.write(f"""integrator     = steep
nsteps         = {nsteps}
emtol          = {EM_TOL}
emstep         = {EM_STEP}
nstlog         = 1000
nstenergy      = 1000
cutoff-scheme  = Verlet
rlist          = 1.0
rcoulomb       = 1.0
rvdw           = 1.0
coulombtype    = PME
vdwtype        = cutoff
fourierspacing = 0.16
pme-order      = 4
pbc            = xyz
""")

def _write_mdp_nvt(path, nsteps, dt=0.002, ref_t_K=None, annealing_start=None):
    t_groups = " ".join(_COMPONENTS)
    ref_t_val = ref_t_K if ref_t_K is not None else int(TEMPERATURE)
    ref_t = " ".join([str(ref_t_val)] * len(_COMPONENTS))
    tau_t = " ".join(["0.1"] * len(_COMPONENTS))
    nstenergy = max(100, nsteps // 100)
    lines = [
        "integrator     = md",
        f"nsteps         = {nsteps}",
        f"dt             = {dt}",
        "nstlog         = 2500",
        f"nstenergy      = {nstenergy}",
        "nstxout        = 0",
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
    with open(path, "w") as f: f.write("\n".join(lines) + "\n")

def _grompp_local(tpr, gro, top, mdp, ndx, workdir):
    r = subprocess.run(
        [GMX, "grompp", "-f", mdp, "-c", gro, "-p", top, "-n", ndx, "-o", tpr, "-maxwarn", "5"],
        cwd=workdir, capture_output=True, timeout=900)
    if r.returncode != 0:
        log(f"  grompp stderr: {r.stderr.decode(errors='replace')[-600:]}")
    return r.returncode

def _mdrun_1node(deffnm, workdir, tpr, timeout, label, em=False):
    """Run mdrun as a single MPI rank via SSH to NODE1. -update gpu for MD; em skips it."""
    ldpath = os.environ.get("LD_LIBRARY_PATH", "")
    flags = "-nb gpu" if em else "-nb gpu -pme gpu -bonded cpu -update gpu"
    cmd_str = (
        f"LD_LIBRARY_PATH={ldpath} "
        f"mpirun -n 1 --mca pml ob1 --mca btl self,tcp "
        f"-x LD_LIBRARY_PATH={ldpath} "
        f"{GMX} mdrun -v -s {tpr} -deffnm {deffnm} {flags} -gpu_id 0 -ntomp {NTOMP}"
    )
    return _ssh_run(workdir, cmd_str, timeout, label)

# ── Pipeline ───────────────────────────────────────────────────────────────
def main():
    log(f"N={N_TOTAL:,}  density={DENSITY}  em_nsteps={EM_NSTEPS}  emtol={EM_TOL}")
    workdir = tempfile.mkdtemp(prefix=f"bench1n_{N_TOTAL}_")
    log(f"Workdir: {workdir}")

    # ── Build inputs locally on node0 ─────────────────────────────────────
    gro = os.path.join(workdir, "system.gro")
    top = os.path.join(workdir, "system.top")
    ndx = os.path.join(workdir, "index.ndx")
    n_actual = _write_gro(gro, N_TOTAL)
    _write_top(top, N_TOTAL)
    _write_ndx(ndx, N_TOTAL)
    log(f"System built: N={n_actual:,}, box={_box_size(N_TOTAL):.3f} nm")

    # ── EM ────────────────────────────────────────────────────────────────
    em_mdp = os.path.join(workdir, "em.mdp"); _write_mdp_em(em_mdp)
    em_tpr = os.path.join(workdir, "em.tpr")
    if _grompp_local(em_tpr, gro, top, em_mdp, ndx, workdir) != 0:
        log("grompp(em) FAILED"); return
    log("grompp(em) OK")
    _push(em_tpr, workdir)
    rc, dt_em = _mdrun_1node("em", workdir, "em.tpr", timeout=36000, label="mdrun(em)[1n]", em=True)
    if rc != 0: log("em FAILED — abort"); return
    _pull(workdir, "em.gro", os.path.join(workdir, "em.gro"))

    # ── Prewarm at 10K (0.5 ps) ───────────────────────────────────────────
    pw_mdp = os.path.join(workdir, "prewarm.mdp"); _write_mdp_nvt(pw_mdp, nsteps=500, dt=0.001, ref_t_K=10)
    pw_tpr = os.path.join(workdir, "prewarm.tpr")
    if _grompp_local(pw_tpr, os.path.join(workdir, "em.gro"), top, pw_mdp, ndx, workdir) != 0:
        log("grompp(prewarm) FAILED"); return
    log("grompp(prewarm) OK")
    _push(pw_tpr, workdir)
    rc, dt_pw = _mdrun_1node("prewarm", workdir, "prewarm.tpr", timeout=7200, label="mdrun(prewarm)[1n]")
    if rc != 0: log("prewarm FAILED — abort"); return
    _pull(workdir, "prewarm.gro", os.path.join(workdir, "prewarm.gro"))

    # ── EQ (10 ps, annealing 10K → T) ─────────────────────────────────────
    eq_mdp = os.path.join(workdir, "eq.mdp"); _write_mdp_nvt(eq_mdp, nsteps=10000, dt=0.001, annealing_start=10)
    eq_tpr = os.path.join(workdir, "eq.tpr")
    if _grompp_local(eq_tpr, os.path.join(workdir, "prewarm.gro"), top, eq_mdp, ndx, workdir) != 0:
        log("grompp(eq) FAILED"); return
    log("grompp(eq) OK")
    _push(eq_tpr, workdir)
    rc, dt_eq = _mdrun_1node("eq", workdir, "eq.tpr", timeout=14400, label="mdrun(eq)[1n]")
    if rc != 0: log("eq FAILED — abort"); return
    _pull(workdir, "eq.gro", os.path.join(workdir, "eq.gro"))

    # ── PROD (200 ps) — THE BENCHMARK ─────────────────────────────────────
    prod_mdp = os.path.join(workdir, "prod.mdp")
    _write_mdp_nvt(prod_mdp, nsteps=int(BENCH_PS / DT), dt=DT)
    prod_tpr = os.path.join(workdir, "prod.tpr")
    if _grompp_local(prod_tpr, os.path.join(workdir, "eq.gro"), top, prod_mdp, ndx, workdir) != 0:
        log("grompp(prod) FAILED"); return
    log("grompp(prod) OK — starting 200 ps benchmark on 1-node")
    _push(prod_tpr, workdir)
    rc, dt_prod = _mdrun_1node("prod", workdir, "prod.tpr", timeout=72000, label="BENCH(prod)[1n]")

    result = {
        "n_total":  N_TOTAL, "density": DENSITY, "bench_ps": BENCH_PS,
        "em_s":      round(dt_em, 1),
        "prewarm_s": round(dt_pw, 1),
        "eq_s":      round(dt_eq, 1),
        "prod_1node_s": round(dt_prod, 1) if rc == 0 else None,
    }
    log(f"FINAL: em={dt_em:.0f}s  prewarm={dt_pw:.0f}s  eq={dt_eq:.0f}s  prod={dt_prod:.0f}s  rc={rc}")
    with open(OUTPUT_JSON, "w") as f: json.dump(result, f, indent=2)
    log(f"Saved → {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
