"""
prepare.py — GROMACS simulation runner and thermal resistance extractor.
DO NOT MODIFY. Qwen only modifies structure.py.

Workflow:
  1. Import structure.py parameters (composition, LJ params, etc.)
  2. Validate all parameters are within physical bounds.
  3. Generate GROMACS input files: .gro (coords), .top (topology), .mdp (params).
  4. Run:  gmx_mpi grompp + gmx_mpi mdrun -nb gpu -pme gpu -bonded gpu
  5. Run:  echo <energy_terms> | gmx_mpi energy -f ener.edr
  6. Parse cross-component LJ energies → compute thermal_resistance proxy.
  7. Print results in the exact format loop.sh expects.

Output format (parsed by loop.sh grep):
  thermal_resistance:   X.XXXXXXE-XX
  thermal_conductivity: X.XXXXXX
  sim_time_s:           NNN.N
"""

import importlib.util, os, sys, subprocess, tempfile, time, math, re, shutil

# ── Load structure.py ─────────────────────────────────────────────────────────
_REPO = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("structure", os.path.join(_REPO, "structure.py"))
_s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s)

N_TOTAL     = int(_s.N_TOTAL)
COMPOSITION = dict(_s.COMPOSITION)
LJ_PARAMS   = dict(_s.LJ_PARAMS)
CROSS       = dict(_s.CROSS_INTERACTIONS)
TEMPERATURE = float(_s.TEMPERATURE)
SIM_TIME_PS = float(_s.SIM_TIME_PS)

GMX      = shutil.which("gmx_mpi") or "/usr/local/gromacs/bin/gmx_mpi"
MPIRUN   = shutil.which("mpirun") or "mpirun"
NTOMP    = str(os.cpu_count() or 20)
SIM_BUDGET_S = 360   # hard wall: kill mdrun if it exceeds this

# ── Multi-node MPI + NCCL config ─────────────────────────────────────────────
HOSTFILE     = "/home/nvidia/autotherm/hostfile_gromacs"  # nodes 1-3 only; node0 reserved for vLLM
N_RANKS      = 3
NCCL_LIB     = "/home/nvidia/nccl_spark_cluster/build/lib"
REMOTE_NODES = ["10.137.203.184", "10.137.203.174", "10.137.203.177"]

def _sync_to_nodes(workdir, *files):
    """Create workdir on nodes 1-3 and copy input files there via SSH/SCP."""
    for node in REMOTE_NODES:
        subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no",
                        f"nvidia@{node}", f"mkdir -p {workdir}"],
                       capture_output=True, timeout=10)
        for f in files:
            subprocess.run(["scp", "-q", f, f"nvidia@{node}:{workdir}/"],
                           capture_output=True, timeout=30)

def _cleanup_remote(workdir):
    """Remove temp workdir from nodes 1-3 after simulation."""
    for node in REMOTE_NODES:
        subprocess.run(["ssh", "-o", "StrictHostKeyChecking=no",
                        f"nvidia@{node}", f"rm -rf {workdir}"],
                       capture_output=True, timeout=10)

def _pull_from_rank0(workdir, *files):
    """Pull output files written by MPI rank 0 (on REMOTE_NODES[0]) back to node0.

    With the 3-node hostfile (nodes 1-3), rank 0 runs on REMOTE_NODES[0] and
    writes all GROMACS output files there. grompp runs locally on node0, so we
    must scp the outputs back before the next grompp step.
    """
    rank0 = REMOTE_NODES[0]
    for fname in files:
        subprocess.run(["scp", "-q", "-o", "StrictHostKeyChecking=no",
                        f"nvidia@{rank0}:{workdir}/{fname}",
                        os.path.join(workdir, fname)],
                       capture_output=True, timeout=30)

def _mpirun_gpu(tpr_deffnm, workdir, timeout, label, extra_flags=None):
    """Run gmx_mpi mdrun on all 4 nodes via NCCL-enabled mpirun."""
    ldpath = f"{NCCL_LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    cmd = [
        MPIRUN, "-n", str(N_RANKS),
        "--hostfile", HOSTFILE,
        "--map-by", "node",
        "--mca", "pml", "ob1",
        "--mca", "btl_tcp_if_include", "enp1s0f0np0",
        "-x", f"LD_LIBRARY_PATH={ldpath}",
        "-x", "NCCL_DEBUG=INFO",
        "-x", "NCCL_SOCKET_IFNAME=enp1s0f0np0",
        "-x", "NCCL_IB_HCA=mlx5_0",
        GMX, "mdrun", "-v", "-deffnm", tpr_deffnm,
        "-nb", "gpu", "-update", "cpu", "-gpu_id", "0", "-ntomp", NTOMP,
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    return _run(cmd, workdir, timeout=timeout, label=label)

# ── Validation ────────────────────────────────────────────────────────────────
BOUNDS = {
    "sigma":   (0.20, 0.60),
    "epsilon": (0.50, 8.00),
    "mass":    (4.0,  200.0),
}

def _check(val, lo, hi, name):
    if not (lo <= val <= hi):
        sys.exit(f"[prepare] INVALID: {name}={val} out of bounds [{lo}, {hi}]")

comp_vals = list(COMPOSITION.values())
if abs(sum(comp_vals) - 1.0) > 1e-6:
    sys.exit(f"[prepare] INVALID: COMPOSITION sums to {sum(comp_vals):.6f}, must be 1.0")
for k, v in COMPOSITION.items():
    _check(v, 0.02, 0.96, f"COMPOSITION[{k}]")
for comp, p in LJ_PARAMS.items():
    for param, (lo, hi) in BOUNDS.items():
        _check(p[param], lo, hi, f"LJ_PARAMS[{comp}][{param}]")
for pair, p in CROSS.items():
    _check(p['sigma'],   0.20, 0.60, f"CROSS_INTERACTIONS[{pair}][sigma]")
    _check(p['epsilon'], 0.50, 8.00, f"CROSS_INTERACTIONS[{pair}][epsilon]")
_check(TEMPERATURE, 280, 480, "TEMPERATURE")
_check(N_TOTAL,     500, 40000, "N_TOTAL")
_check(SIM_TIME_PS, 50,  400,  "SIM_TIME_PS")

print(f"[prepare] Validated. N={N_TOTAL}, T={TEMPERATURE}K, t={SIM_TIME_PS}ps", flush=True)
print(f"[prepare] Composition: " +
      ", ".join(f"{k}={v:.0%}" for k, v in COMPOSITION.items()), flush=True)

# ── Geometry ──────────────────────────────────────────────────────────────────
def _box_size(n_total, density_nm3=0.85):
    """Box side length in nm for given particle count and reduced density."""
    # Use average sigma for reference
    avg_sig = sum(LJ_PARAMS[c]['sigma'] for c in LJ_PARAMS) / len(LJ_PARAMS)
    vol = n_total * (avg_sig ** 3) / density_nm3
    return vol ** (1/3)

BOX_L = _box_size(N_TOTAL)
BOX_AREA = BOX_L * BOX_L   # nm² (one face; used for normalisation)

# ── GROMACS file generators ───────────────────────────────────────────────────
_COMPONENTS = list(COMPOSITION.keys())
_SHORT    = {'metal': 'MM',   'filler': 'GR',   'binder': 'PL'}
_AT_NUM   = {'metal': 49,     'filler': 6,       'binder': 14}
_RESNAMES = {'metal': 'METL', 'filler': 'FILL',  'binder': 'BIND'}

def _write_top(path):
    """Write GROMACS topology .top file."""
    comps = _COMPONENTS
    # Compute cross-interaction pairs (including explicit overrides + LB mixing)
    pairs = {}
    for i, ci in enumerate(comps):
        for cj in comps[i:]:
            key1 = (ci, cj)
            key2 = (cj, ci)
            if key1 in CROSS:
                pairs[(ci, cj)] = CROSS[key1]
            elif key2 in CROSS:
                pairs[(ci, cj)] = CROSS[key2]
            else:
                # Lorentz-Berthelot
                s = (LJ_PARAMS[ci]['sigma']   + LJ_PARAMS[cj]['sigma'])   / 2
                e = math.sqrt(LJ_PARAMS[ci]['epsilon'] * LJ_PARAMS[cj]['epsilon'])
                pairs[(ci, cj)] = {'sigma': s, 'epsilon': e}

    lines = [
        "; GROMACS topology for Thermal Interface Material",
        "[ defaults ]",
        "; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ",
        "  1       2          no         1.0      1.0",
        "",
        "[ atomtypes ]",
        "; name  at.num  mass    charge  ptype  sigma(nm)  epsilon(kJ/mol)",
    ]
    for c in comps:
        p = LJ_PARAMS[c]
        lines.append(f"  {_SHORT[c]:<6} {_AT_NUM.get(c,6):<7} {p['mass']:<8.3f} 0.000   A  "
                     f"{p['sigma']:.6f}  {p['epsilon']:.6f}")

    lines += ["", "[ nonbond_params ]", "; i   j   funct  sigma(nm)  epsilon(kJ/mol)"]
    for (ci, cj), p in pairs.items():
        if ci != cj:
            lines.append(f"  {_SHORT[ci]:<5}{_SHORT[cj]:<5}1  {p['sigma']:.6f}  {p['epsilon']:.6f}")

    # Molecule definitions (each component = single-atom molecule)
    for c in comps:
        lines += [
            "", f"[ moleculetype ]",
            f"; name  nrexcl", f"{c.capitalize():<12} 1",
            "[ atoms ]",
            "; nr  type  resnr  residue  atom  cgnr  charge  mass",
            f"  1   {_SHORT[c]:<5}  1    {_RESNAMES[c]:<5} {_SHORT[c]:<5}   1    0.000  {LJ_PARAMS[c]['mass']:.3f}",
        ]

    lines += ["", "[ system ]", "TIM_System", "", "[ molecules ]"]
    for c in comps:
        n = max(1, int(round(N_TOTAL * COMPOSITION[c])))
        lines.append(f"{c.capitalize():<14} {n}")

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def _write_gro(path):
    """Write GROMACS coordinate .gro file using a simple cubic lattice.

    Lattice placement guarantees inter-atom spacing = BOX_L / cbrt(N_TOTAL),
    which is always >= min_sigma for our density parameter. No atom overlaps,
    so em converges quickly regardless of system size.
    """
    atoms = []
    for c in _COMPONENTS:
        n = max(1, int(round(N_TOTAL * COMPOSITION[c])))
        atoms.extend([(c, i+1) for i in range(n)])

    n_total = len(atoms)
    n_side  = math.ceil(n_total ** (1/3))
    spacing = BOX_L / n_side

    with open(path, 'w') as f:
        f.write(f"TIM System\n{n_total}\n")
        for idx, (c, _) in enumerate(atoms):
            ix = idx % n_side
            iy = (idx // n_side) % n_side
            iz = idx // (n_side ** 2)
            x  = (ix + 0.5) * spacing
            y  = (iy + 0.5) * spacing
            z  = (iz + 0.5) * spacing
            f.write(f"{idx+1:5d}{_RESNAMES[c]:<5}{_SHORT[c]:<5}{idx+1:5d}"
                    f"{x:8.3f}{y:8.3f}{z:8.3f}\n")
        f.write(f"   {BOX_L:.5f}   {BOX_L:.5f}   {BOX_L:.5f}\n")


def _write_ndx(path):
    """Write GROMACS index file with per-component groups."""
    offset = 1
    groups = {}
    for c in _COMPONENTS:
        n = max(1, int(round(N_TOTAL * COMPOSITION[c])))
        groups[c.capitalize()] = list(range(offset, offset + n))
        offset += n
    all_atoms = list(range(1, offset))
    with open(path, 'w') as f:
        for name, indices in groups.items():
            f.write(f"[ {name} ]\n")
            for i, idx in enumerate(indices):
                f.write(f"{idx:6d}")
                if (i + 1) % 15 == 0:
                    f.write('\n')
            f.write('\n')
        f.write("[ System ]\n")
        for i, idx in enumerate(all_atoms):
            f.write(f"{idx:6d}")
            if (i + 1) % 15 == 0:
                f.write('\n')
        f.write('\n')


def _write_mdp(path, mode, nsteps, energygrps=False, save_xtc=False):
    """Write GROMACS MDP parameter file.

    energygrps=False → GPU-compatible (no per-group energy decomposition).
    energygrps=True  → CPU rerun analysis: enables cross-component LJ extraction.
    save_xtc=True    → write compressed trajectory every 250 steps for rerun.
    """
    egroups = ' '.join(c.capitalize() for c in _COMPONENTS)
    nstcalc  = "1"   if energygrps else "10"
    nstenerg = "1"   if energygrps else "250"
    base = {
        'em': [
            "integrator      = steep",
            f"nsteps          = {nsteps}",
            "emtol           = 100.0",
            "emstep          = 0.01",
            "nstlog          = 500",
            "nstenergy       = 100",
            "nstxout         = 0",
            "nstvout         = 0",
        ],
        'nvt': [
            "integrator      = md",
            f"nsteps          = {nsteps}",
            "dt              = 0.002",
            "nstlog          = 2500",
            f"nstenergy       = {nstenerg}",
            "nstxout         = 0",
            "nstvout         = 0",
            f"nstcalcenergy   = {nstcalc}",
            "tcoupl          = V-rescale",
            f"tc-grps         = {egroups}",
            f"tau-t           = " + ' '.join(['0.1'] * len(_COMPONENTS)),
            f"ref-t           = " + ' '.join([str(int(TEMPERATURE))] * len(_COMPONENTS)),
            "pcoupl          = no",
        ],
    }
    extra = []
    if energygrps:
        extra += [f"energygrps      = {egroups}"]
    if save_xtc:
        extra += ["nstxout-compressed = 250"]
    shared = [
        "cutoff-scheme   = Verlet",
        "coulombtype     = cut-off",
        "rcoulomb        = 1.0",
        "vdwtype         = cut-off",
        "rvdw            = 1.0",
        "rlist           = 1.0",
        "nstlist         = 20",
        "pbc             = xyz",
        "constraints     = none",
        "comm-mode       = Linear",
    ]
    with open(path, 'w') as f:
        f.write('\n'.join(base[mode] + extra + shared) + '\n')


def _run(cmd, cwd, timeout=300, label=""):
    """Run a GROMACS command, return (returncode, stdout+stderr)."""
    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[prepare] {label} FAILED (exit={result.returncode}, {elapsed:.1f}s):", flush=True)
        print(result.stderr[-1000:], flush=True)
    else:
        print(f"[prepare] {label} OK ({elapsed:.1f}s)", flush=True)
    return result.returncode, result.stdout + result.stderr


def _parse_energy(edr_path, workdir, terms):
    """Extract average energy terms from .edr using gmx energy.

    Sends indices 1-200 to select all terms, then parses the summary statistics
    table printed to stdout. Uses line.startswith(term) so menu lines (which
    have an index number before the term name) are never mistakenly matched.
    Returns dict {term: average_value_kJ_mol}.
    """
    term_input = '\n'.join(str(i) for i in range(1, 200)) + '\n0\n'
    result = subprocess.run(
        [GMX, "energy", "-f", edr_path,
         "-o", os.path.join(workdir, "energy_out.xvg"), "-xvg", "none"],
        input=term_input, cwd=workdir, capture_output=True, text=True, timeout=60
    )
    values = {}
    for raw_line in (result.stdout + result.stderr).splitlines():
        line = raw_line.strip()
        for term in terms:
            # Only match statistics lines where the term name starts the line.
            # This avoids menu lines like "  23  LJ-SR:Metal-Filler  24  ..."
            # where the first numeric value would be the menu index, not energy.
            if line.startswith(term):
                rest = line[len(term):].strip()
                parts = rest.split()
                for p in parts:
                    if p in ('--', 'nan', 'inf', '-inf'):
                        continue
                    try:
                        values[term] = float(p)
                        break
                    except ValueError:
                        continue
    return values


# ── Main simulation pipeline ──────────────────────────────────────────────────
def run_simulation():
    workdir = tempfile.mkdtemp(prefix="gromacs_tim_")
    print(f"[prepare] Workdir: {workdir}", flush=True)
    t_start = time.time()

    try:
        # Generate input files
        gro  = os.path.join(workdir, "system.gro")
        top  = os.path.join(workdir, "system.top")
        em_mdp  = os.path.join(workdir, "em.mdp")
        nvt_mdp = os.path.join(workdir, "nvt.mdp")

        eq_mdp   = os.path.join(workdir, "eq.mdp")
        anal_mdp = os.path.join(workdir, "analysis.mdp")
        ndx      = os.path.join(workdir, "index.ndx")

        print("[prepare] Generating input files...", flush=True)
        _write_top(top)
        _write_gro(gro)
        _write_ndx(ndx)
        _write_mdp(em_mdp,  'em',  nsteps=5000)
        nvt_steps = max(5000, int(SIM_TIME_PS / 0.002))
        # GPU runs: no energygrps so GPU non-bonded is fully utilised
        _write_mdp(eq_mdp,   'nvt', nsteps=5000,      energygrps=False, save_xtc=False)
        _write_mdp(nvt_mdp,  'nvt', nsteps=nvt_steps, energygrps=False, save_xtc=True)
        # Analysis rerun: energygrps enabled for cross-component LJ extraction
        _write_mdp(anal_mdp, 'nvt', nsteps=1,          energygrps=True,  save_xtc=False)

        # ── Step 1: GPU dynamics (4 nodes, NCCL pressure/force exchange) ────────
        # Energy minimisation
        rc, _ = _run([GMX, "grompp", "-f", em_mdp, "-c", gro, "-p", top,
                      "-o", "em.tpr", "-maxwarn", "5"],
                     workdir, timeout=60, label="grompp(em)")
        if rc != 0: sys.exit("[prepare] grompp(em) failed")

        _sync_to_nodes(workdir, os.path.join(workdir, "em.tpr"))
        rc, _ = _mpirun_gpu("em", workdir, timeout=120, label="mdrun(em)")
        if rc != 0: sys.exit("[prepare] mdrun(em) failed")
        _pull_from_rank0(workdir, "em.gro")

        # NVT equilibration (10 ps)
        rc, _ = _run([GMX, "grompp", "-f", eq_mdp, "-c", "em.gro", "-p", top,
                      "-n", ndx, "-o", "eq.tpr", "-maxwarn", "5"],
                     workdir, timeout=60, label="grompp(eq)")
        if rc != 0: sys.exit("[prepare] grompp(eq) failed")

        _sync_to_nodes(workdir, os.path.join(workdir, "eq.tpr"))
        rc, _ = _mpirun_gpu("eq", workdir, timeout=120, label="mdrun(eq)")
        if rc != 0: sys.exit("[prepare] mdrun(eq) failed")
        _pull_from_rank0(workdir, "eq.gro")

        # NVT production (saves xtc for rerun)
        rc, _ = _run([GMX, "grompp", "-f", nvt_mdp, "-c", "eq.gro", "-p", top,
                      "-n", ndx, "-o", "prod.tpr", "-maxwarn", "5"],
                     workdir, timeout=60, label="grompp(prod)")
        if rc != 0: sys.exit("[prepare] grompp(prod) failed")

        _sync_to_nodes(workdir, os.path.join(workdir, "prod.tpr"))
        rc, _ = _mpirun_gpu("prod", workdir, timeout=SIM_BUDGET_S, label="mdrun(prod)")
        if rc != 0: sys.exit("[prepare] mdrun(prod) failed")
        _pull_from_rank0(workdir, "prod.xtc")

        # ── Step 2: CPU energy audit (rerun, single-rank, node0 only) ───────────
        # Reprocess the GPU trajectory with energygrps. No dynamics — fast.
        rc, _ = _run([GMX, "grompp", "-f", anal_mdp, "-c", "eq.gro", "-p", top,
                      "-n", ndx, "-o", "analysis.tpr", "-maxwarn", "5"],
                     workdir, timeout=60, label="grompp(analysis)")
        if rc != 0: sys.exit("[prepare] grompp(analysis) failed")

        rc, _ = _run([MPIRUN, "-n", "1", GMX, "mdrun", "-v",
                      "-rerun", "prod.xtc", "-deffnm", "analysis", "-ntomp", NTOMP],
                     workdir, timeout=120, label="mdrun(rerun)")
        if rc != 0: sys.exit("[prepare] mdrun(rerun) failed")

        _cleanup_remote(workdir)

        # ── Extract energies ──────────────────────────────────────────────────
        comp_caps = [c.capitalize() for c in _COMPONENTS]
        cross_terms = []
        for i, ci in enumerate(comp_caps):
            for cj in comp_caps[i+1:]:
                cross_terms.append(f"LJ-SR:{ci}-{cj}")

        energy_vals = _parse_energy("analysis.edr", workdir, cross_terms)
        print(f"[prepare] Energy terms found: {energy_vals}", flush=True)

        total_cross = sum(energy_vals.get(t, 0.0) for t in cross_terms)
        if total_cross >= 0:
            all_terms = cross_terms + [t.replace("LJ-SR:", "LJ:") for t in cross_terms]
            energy_vals2 = _parse_energy("analysis.edr", workdir, all_terms)
            total_cross = sum(energy_vals2.get(t, 0.0) for t in all_terms)

        # Coupling strength per unit area [kJ/mol/nm²]
        thermal_coupling = -total_cross / max(BOX_AREA, 1.0)
        if thermal_coupling <= 0:
            thermal_coupling = 0.1   # fallback if extraction failed

        # Convert to thermal resistance proxy [m²K/W]
        # Calibrated so that a well-optimised LJ TIM gives ~1e-8 m²K/W
        # (typical range for real TIMs: 1e-8 to 1e-6 m²K/W)
        K_MODEL = 1.38e-2
        thermal_resistance = K_MODEL / thermal_coupling

        # Also report as thermal conductivity proxy [W/m/K] for intuition
        # λ_proxy = L / (R × A_normalised)
        thermal_conductivity = BOX_L / thermal_resistance

        sim_time_s = time.time() - t_start

        print(f"[prepare] Cross LJ energy: {total_cross:.3f} kJ/mol", flush=True)
        print(f"[prepare] Box area: {BOX_AREA:.2f} nm²", flush=True)
        print(f"[prepare] Thermal coupling: {thermal_coupling:.4f} kJ/mol/nm²", flush=True)

        return thermal_resistance, thermal_conductivity, sim_time_s

    finally:
        # Copy GROMACS logs to /tmp before deleting workdir (for crash debugging)
        import glob as _glob
        for _log in _glob.glob(os.path.join(workdir, "*.log")):
            try:
                shutil.copy(_log, f"/tmp/gmx_last_{os.path.basename(_log)}")
            except Exception:
                pass
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.isfile(GMX):
        sys.exit(f"[prepare] ERROR: gmx_mpi not found at {GMX}. Install GROMACS 2025 with GPU support first.")

    thermal_resistance, thermal_conductivity, sim_time_s = run_simulation()

    print("---")
    print(f"thermal_resistance:   {thermal_resistance:.6E}")
    print(f"thermal_conductivity: {thermal_conductivity:.6f}")
    print(f"sim_time_s:           {sim_time_s:.1f}")
    print(f"n_atoms:              {N_TOTAL}")
    print(f"sim_ps:               {SIM_TIME_PS}")
