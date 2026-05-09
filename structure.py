"""
structure.py — Thermal Interface Material (TIM) composition.

THIS IS THE FILE QWEN EDITS EACH ITERATION.
Goal: Minimize interfacial_thermal_resistance (m²K/W) — lower is better.

Physical system:
  A copper heat sink sits above a silicon GPU die. Between them is a TIM layer
  made of three components. This file defines those components and how they
  pack together. prepare.py builds the GROMACS simulation from these parameters
  and reports the resulting thermal performance.

The three components:
  METAL   — Liquid metal matrix (InGa alloy). High intrinsic conductivity,
             wets metal surfaces well. Heavy atoms (m≈97 amu).
  FILLER  — Graphene/BN nanoflake inclusions. Extremely high phonon conductivity
             along the basal plane. Acts as a thermal highway if well-dispersed.
  BINDER  — Polymer chain segments (polysiloxane-like). Lower conductivity but
             critical for adhesion, gap-filling, and keeping FILLER suspended.

What you may change:
  - COMPOSITION fractions (must sum exactly to 1.0, each > 0.02)
  - LJ_PARAMS sigma/epsilon for any component (within physical bounds below)
  - CROSS_INTERACTIONS epsilon/sigma for any pair (override Lorentz-Berthelot)
  - TEMPERATURE (300–450 K)
  - N_TOTAL (1000–8000, trade-off between accuracy and runtime)
  - SIM_TIME_PS (50–400 ps; longer = more accurate but slower)

Physical bounds (do NOT violate — prepare.py will reject out-of-range params):
  sigma:   0.20–0.60 nm
  epsilon: 0.50–8.00 kJ/mol
  mass:    4.0–200.0 amu
  TEMPERATURE: 280–480 K
  N_TOTAL: 500–40000
  SIM_TIME_PS: 50–400
  Each COMPOSITION fraction: 0.02–0.96

How prepare.py computes the metric:
  1. Builds a periodic box with N_TOTAL atoms placed at random (then minimised).
  2. Runs short NVT equilibration (10 ps).
  3. Runs NVT production (SIM_TIME_PS) with energy group tracking.
  4. Extracts cross-component LJ interaction energies E_AB, E_AC, E_BC (kJ/mol).
  5. thermal_coupling  = -(E_AB + E_AC + E_BC) / box_area_nm2   [kJ/mol/nm²]
  6. interfacial_thermal_resistance = k_model / thermal_coupling
     where k_model = 1.38e-2  (empirical calibration vs known LJ systems)
  Lower thermal_resistance = better TIM.
  The script also prints thermal_conductivity_proxy as a positive W/m/K estimate.
"""

# ── Composition (must sum to 1.0, each >= 0.02) ───────────────────────────────
N_TOTAL = 16000

COMPOSITION = {
    'metal':   0.60,   # liquid metal (InGa)
    'filler':  0.25,   # graphene/BN nanoflake
    'binder':  0.15,   # polymer binder
}

# ── Per-component Lennard-Jones parameters ─────────────────────────────────────
# sigma [nm], epsilon [kJ/mol], mass [amu]
LJ_PARAMS = {
    'metal':  {'sigma': 0.280, 'epsilon': 4.20, 'mass': 97.0},
    'filler': {'sigma': 0.340, 'epsilon': 2.80, 'mass': 12.0},
    'binder': {'sigma': 0.410, 'epsilon': 1.60, 'mass': 44.0},
}

# ── Cross-interaction overrides (Lorentz-Berthelot mixing used for any pair
#    not listed here: sigma_ij = (s_i+s_j)/2, eps_ij = sqrt(e_i*e_j)) ────────
CROSS_INTERACTIONS = {
    ('metal', 'filler'): {'sigma': 0.310, 'epsilon': 3.50},
    ('metal', 'binder'): {'sigma': 0.345, 'epsilon': 2.20},
    ('filler','binder'): {'sigma': 0.375, 'epsilon': 1.90},
}

# ── Simulation state ───────────────────────────────────────────────────────────
TEMPERATURE  = 350.0   # K  (typical GPU die operating temperature)
SIM_TIME_PS  = 100     # ps (production run; 50–400 allowed)
