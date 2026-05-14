# ── Composition (must sum to 1.0, each >= 0.02) ───────────────────────────────
N_TOTAL = 500000

COMPOSITION = {
    'metal':   0.50,   # liquid metal (InGa)
    'filler':  0.38,   # graphene/BN nanoflake
    'binder':  0.12,   # polymer binder
}

# ── Per-component Lennard-Jones parameters ─────────────────────────────────────
# sigma [nm], epsilon [kJ/mol], mass [amu]
LJ_PARAMS = {
    'metal':  {'sigma': 0.280, 'epsilon': 7.0, 'mass': 97.0},
    'filler': {'sigma': 0.340, 'epsilon': 6.0, 'mass': 12.0},
    'binder': {'sigma': 0.410, 'epsilon': 2.0, 'mass': 44.0},
}

# ── Cross-interaction overrides (Lorentz-Berthelot mixing used for any pair
#    not listed here: sigma_ij = (s_i+s_j)/2, eps_ij = sqrt(e_i*e_j)) ────────
CROSS_INTERACTIONS = {
    ('metal', 'filler'): {'sigma': 0.310, 'epsilon': 8.0},
    ('metal', 'binder'): {'sigma': 0.345, 'epsilon': 8.0},
    ('filler','binder'): {'sigma': 0.375, 'epsilon': 8.0},
}

# ── Simulation state ───────────────────────────────────────────────────────────
TEMPERATURE  = 280.0   # K  (typical GPU die operating temperature)
SIM_TIME_PS  = 100
