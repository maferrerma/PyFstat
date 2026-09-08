"""
Bilby search: Fully coherent F-statistic
========================================

Directed Bilby search for an isolated CW signal using the PyFstat
fully coherent F-statistic likelihood.

This example mirrors the basic fully coherent MCMC example, but calls
``MCMCSearch.run_bilby()`` instead of ``MCMCSearch.run()``.
"""

import os
import sys

import numpy as np

import pyfstat

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.isdir(os.path.join(repo_root, "pyfstat")):
    sys.path.insert(0, repo_root)

label = "PyFstatExampleFullyCoherentBilbySearch"
outdir = os.path.join("PyFstat_example_data", label)
logger = pyfstat.set_up_logger(label=label, outdir=outdir)

# Properties of the GW data
data_parameters = {
    "sqrtSX": 1e-23,
    "tstart": 1000000000,
    "duration": 20 * 86400,
    "detectors": "H1",
}
tend = data_parameters["tstart"] + data_parameters["duration"]
mid_time = 0.5 * (data_parameters["tstart"] + tend)

# Properties of the signal
depth = 10
signal_parameters = {
    "F0": 30.0,
    "F1": -1e-10,
    "F2": 0,
    "Alpha": np.radians(83.6292),
    "Delta": np.radians(22.0144),
    "tref": mid_time,
    "h0": data_parameters["sqrtSX"] / depth,
    "cosi": 1.0,
}

data = pyfstat.Writer(
    label=label,
    outdir=outdir,
    **data_parameters,
    **signal_parameters,
)
data.make_data()

twoF = data.predict_fstat()
logger.info("Predicted twoF value: {}\n".format(twoF))

DeltaF0 = 5e-7
DeltaF1 = 5e-13
theta_prior = {
    "F0": {
        "type": "unif",
        "lower": signal_parameters["F0"] - DeltaF0 / 2.0,
        "upper": signal_parameters["F0"] + DeltaF0 / 2.0,
    },
    "F1": {
        "type": "unif",
        "lower": signal_parameters["F1"] - DeltaF1 / 2.0,
        "upper": signal_parameters["F1"] + DeltaF1 / 2.0,
    },
    "transient_tstart": {
        "type": "unif",
        "lower": data.tstart,
        "upper": data.tstart + data.duration - 2 * data.Tsft,
    },
    "transient_duration": {
        "type": "unif",
        "lower": 2 * data.Tsft,
        "upper": data.duration - 2 * data.Tsft,
    },
}
for key in "F2", "Alpha", "Delta":
    theta_prior[key] = signal_parameters[key]

search = pyfstat.MCMCTransientSearch(
    label=label,
    outdir=outdir,
    sftfilepattern=data.sftfilepath,
    theta_prior=theta_prior,
    tref=mid_time,
    minStartTime=data_parameters["tstart"],
    maxStartTime=tend,
)
search.transform_dictionary = dict(
    F0=dict(subtractor=signal_parameters["F0"], symbol="$f-f^\\mathrm{s}$"),
    F1=dict(
        subtractor=signal_parameters["F1"], symbol="$\\dot{f}-\\dot{f}^\\mathrm{s}$"
    ),
)

result = search.run_bilby(
    sampler="dynesty",
    sampler_kwargs={
        "nlive": 400,
        "dlogz": 1.0,
    },
    save_pickle=False,
    save_bilby_logs=True,
    save_bilby_progress=True,
)

logger.info("Bilby result object: {}".format(result))
search.print_summary()
search.plot_corner(add_prior=True, truths=signal_parameters)
