# structure.py — Thermal Interface Material (TIM) composition.
# Goal: Minimize interfacial_thermal_resistance (m²K/W) — lower is better.

# ── System size ────────────────────────────────────────────────────────
N_TOTAL = 30000

# ── Composition (must sum to 1.0, each >= 0.02) ────────────────────────
COMPOSITION = {
    'metal':   0.54,
    'filler':  0.33,
    'binder':  0.13,
}

# ── Per-component Lennard-Jones parameters ─────────────────────────────
# sigma [nm], epsilon [kJ/mol], mass [amu]
LJ_PARAMS = {
    'metal':   {'sigma': 0.280, 'epsilon': 6.0, 'mass': 97.0},
    'filler':  {'sigma': 0.340, 'epsilon': 5.0, 'mass': 12.0},
    'binder':  {'sigma': 0.410, 'epsilon': 1.6, 'mass': 44.0},
}

# ── Cross-interaction overrides (Lorentz-Berthelot mixing used for any pair
#    not listed here: sigma_ij = (s_i+s_j)/2, eps_ij = sqrt(e_i*e_j)) ────────
CROSS_INTERACTIONS = {
    ('metal', 'filler'):   {'sigma': 0.310, 'epsilon': 8.0},
    ('metal', 'binder'):   {'sigma': 0.345, 'epsilon': 8.0},
    ('filler', 'binder'):  {'sigma': 0.375, 'epsilon': 8.0},
}

# ── Simulation state ───────────────────────────────────────────────────
TEMPERATURE  = 340.0   # K (GPU die temperature)
SIM_TIME_PS  = 100     # ps (production run)
