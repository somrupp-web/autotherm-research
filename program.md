# Autonomous TIM Research Agent — DGX Spark Cluster

## Task
Optimize a Thermal Interface Material (TIM) for GPU/CPU die-to-heatsink bonding.
Metric: **thermal_resistance** (m²K/W). **Lower is better.**
Target: Reach < 1.0e-8 m²K/W (competitive with best commercial liquid-metal TIMs).

The physical system: A 3-component LJ fluid represents a TIM layer:
- **Metal**: Liquid In-Ga alloy matrix (high intrinsic conductivity, heavy atoms).
- **Filler**: Graphene/BN nanoflake inclusions (phonon superhighway along basal plane).
- **Binder**: Polysiloxane chain segments (adhesion, gap-filling, filler suspension).

GROMACS simulates this system in a periodic box. The judge measures how strongly
the three components couple at their interfaces — stronger coupling means less
phonon scattering, which means lower thermal resistance.

## Rules
- You may ONLY modify `structure.py`
- `prepare.py` is fixed — do not touch it
- Simulation budget: must complete in under 4 minutes total
- `SIM_TIME_PS` should not exceed 400 (prepare.py enforces this)
- Each `COMPOSITION` fraction must be ≥ 0.02 and all three must sum to exactly 1.0

## What structure.py controls

| Parameter | Effect | Safe range |
|-----------|--------|------------|
| `COMPOSITION['metal']` | Volume of InGa matrix | 0.02–0.96 |
| `COMPOSITION['filler']` | Volume of graphene inclusions | 0.02–0.96 |
| `COMPOSITION['binder']` | Volume of polymer binder | 0.02–0.96 |
| `LJ_PARAMS[c]['sigma']` | Effective atomic radius (nm) | 0.20–0.60 |
| `LJ_PARAMS[c]['epsilon']` | Cohesive energy (kJ/mol) | 0.50–8.00 |
| `LJ_PARAMS[c]['mass']` | Atomic/repeat-unit mass (amu) | 4.0–200.0 |
| `CROSS_INTERACTIONS[(i,j)]['epsilon']` | Cross-interface coupling strength | 0.50–8.00 |
| `CROSS_INTERACTIONS[(i,j)]['sigma']` | Cross-interface size mismatch | 0.20–0.60 |
| `TEMPERATURE` | Operating temperature (K) | 280–480 |
| `N_TOTAL` | System size (atoms) | 500–40000 |
| `SIM_TIME_PS` | Production run length (ps) | 50–400 |

## How to lower thermal_resistance

**Principle**: The metric = K_MODEL / thermal_coupling, where thermal_coupling is
the cross-component LJ interaction energy per unit area. Maximize coupling → minimize resistance.

### High-impact strategies

1. **Strengthen cross-interactions** — increase `CROSS_INTERACTIONS` epsilon values.
   The metal-filler interface is most important (both have high individual conductivity;
   if they couple well, heat transfers efficiently across both materials).
   **Best result to date used cross-eps = 7.0 / 6.0 / 5.0 (metal-filler / metal-binder / filler-binder).**
   Explore the 7.0–8.0 range for all three pairs — this is the single highest-impact lever.
   Try: metal-filler eps=7.5, metal-binder eps=7.0, filler-binder eps=6.5.

2. **Optimize size matching** — when `sigma_ij` for a cross-pair is close to the
   arithmetic mean of `sigma_i` and `sigma_j`, the Lennard-Jones potential well is
   deepest and the interface packs most efficiently.

3. **Filler loading** — graphene fillers have high conductivity but only help if
   they form a percolation network. Typical optimal filler fraction is 20–40%.
   Too much (>50%) causes aggregation (modeled as reduced cross-coupling).
   Too little (<10%) doesn't form a connected thermal path.

4. **Metal-binder tuning** — binder must "wet" the metal to avoid delamination.
   Increase `CROSS_INTERACTIONS[('metal','binder')]['epsilon']` to model
   chemical functionalization (silane coupling agents on the metal surface).

5. **Temperature sweep** — lower T (300 K) increases density and coupling in the
   liquid metal; higher T (400 K) can improve binder chain mobility and gap-filling.

6. **System size** — larger N_TOTAL gives more accurate statistics. With 4 GPUs,
   N=16000 is the standard; N=24000-32000 for final refinement.

### Example promising direction
The best-ever result achieved R_th=4.55e-06 with cross-eps = 7.0/6.0/5.0.
Push all three pairs toward the ceiling (8.0): try metal-filler eps=7.5, metal-binder eps=7.0,
filler-binder eps=6.5. Simultaneously use filler fraction ≈ 0.30–0.32 and binder ≈ 0.13–0.15.

---

## CRITICAL — These mistakes cause prepare.py to exit with an error

### 1. COMPOSITION must sum to exactly 1.0
```python
# WRONG — sums to 0.95
COMPOSITION = {'metal': 0.60, 'filler': 0.25, 'binder': 0.10}

# CORRECT
COMPOSITION = {'metal': 0.60, 'filler': 0.25, 'binder': 0.15}
```

### 2. Do not add or remove keys from COMPOSITION, LJ_PARAMS, or CROSS_INTERACTIONS
prepare.py expects exactly 3 components: 'metal', 'filler', 'binder'.
Adding a 4th component or renaming a key will crash the topology generator.

### 3. Print format — loop.sh parses with grep
```
thermal_resistance:   X.XXXXXXE-XX    ← parsed by loop.sh
thermal_conductivity: X.XXXXXX        ← informational
```
Do not add any print() calls to structure.py. Only prepare.py prints results.

### 4. SIM_TIME_PS must stay within budget
If SIM_TIME_PS is too large and the simulation exceeds 4 min wall clock,
prepare.py will kill the job and the iteration is wasted. For exploration, keep
SIM_TIME_PS in the range 80–120 ps. Use 200–300 ps only to confirm a promising
composition (not for routine search iterations).

---

## Results so far
Check `results.tsv` before every iteration. Each row:
```
commit  thermal_resistance  sim_time_s  status  description
```
- `keep`: new best (lower thermal_resistance than all previous keep rows)
- `discard`: ran but didn't improve
- `crash`: prepare.py exited non-zero

**NEVER STOP. Keep iterating.**
