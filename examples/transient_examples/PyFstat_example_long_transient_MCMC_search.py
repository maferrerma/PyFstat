#!/usr/bin/env python

import pyfstat
import os
import numpy as np

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 0.5
Delta = 1

minStartTime = 1000000000
duration = 200 * 86400
maxStartTime = minStartTime + duration
Tspan = maxStartTime - minStartTime
tref = minStartTime

DeltaF0 = 6e-7
DeltaF1 = 1e-13

# to make the search cheaper, we exactly target the transientStartTime
# to the injected value and only search over TransientTau
theta_prior = {
    "F0": {"type": "unif", "lower": F0 - DeltaF0 / 2.0, "upper": F0 + DeltaF0 / 2.0},
    "F1": {"type": "unif", "lower": F1 - DeltaF1 / 2.0, "upper": F1 + DeltaF1 / 2.0},
    "F2": F2,
    "Alpha": Alpha,
    "Delta": Delta,
    "transient_tstart": minStartTime + 0.25 * duration,
    "transient_duration": {
        "type": "halfnorm",
        "loc": 0.001 * Tspan,
        "scale": 0.5 * Tspan,
    },
}

ntemps = 2
log10beta_min = -1
nwalkers = 100
nsteps = [100, 100]

outdir = os.path.join("example_data", "long_transient")
if not os.path.isdir(outdir) or not np.any(
    [f.endswith(".sft") for f in os.listdir(outdir)]
):
    raise RuntimeError(
        "Please first run PyFstat_example_make_data_for_long_transient_search.py !"
    )

mcmc = pyfstat.MCMCTransientSearch(
    label="transient_search",
    outdir=outdir,
    sftfilepattern=os.path.join(outdir, "*simulated_transient_signal*sft"),
    theta_prior=theta_prior,
    tref=tref,
    minStartTime=minStartTime,
    maxStartTime=maxStartTime,
    nsteps=nsteps,
    nwalkers=nwalkers,
    ntemps=ntemps,
    log10beta_min=log10beta_min,
    transientWindowType="rect",
)
mcmc.run()
mcmc.plot_corner(label_offset=0.7)
mcmc.print_summary()
