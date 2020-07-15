#!/usr/bin/env python

import pyfstat
import os

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 0.5
Delta = 1

tstart = 1000000000
duration = 2 * 86400

transient_tstart = tstart + 0.25 * duration
transient_duration = 0.5 * duration
tref = tstart

h0 = 1e-23
sqrtSX = 1e-22
detectors = "H1,L1"

Tsft = 1800

outdir = os.path.join("example_data", "short_transient")

transient = pyfstat.Writer(
    label="simulated_transient_signal",
    outdir=outdir,
    tref=tref,
    tstart=tstart,
    duration=duration,
    F0=F0,
    F1=F1,
    F2=F2,
    Alpha=Alpha,
    Delta=Delta,
    h0=h0,
    detectors=detectors,
    sqrtSX=sqrtSX,
    transientStartTime=transient_tstart,
    transientTau=transient_tstart,
    transientWindowType="rect",
    Tsft=Tsft,
)
transient.make_data()
