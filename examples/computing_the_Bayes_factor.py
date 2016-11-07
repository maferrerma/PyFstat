from pyfstat import MCMCSearch

F0 = 30.0
F1 = -1e-10
F2 = 0
Alpha = 5e-3
Delta = 6e-2
tref = 362750407.0

tstart = 1000000000
duration = 100*86400
tend = tstart + duration

theta_prior = {'F0': {'type': 'unif', 'lower': F0*(1-1e-6), 'upper': F0*(1+1e-6)},
               'F1': {'type': 'unif', 'lower': F1*(1+1e-2), 'upper': F1*(1-1e-2)},
               'F2': F2,
               'Alpha': Alpha,
               'Delta': Delta
               }

ntemps = 20
log10temperature_min = -2
nwalkers = 100
nsteps = [500, 500]

mcmc = MCMCSearch(label='computing_the_Bayes_factor', outdir='data', 
                  sftfilepath='data/*basic*sft', theta_prior=theta_prior,
                  tref=tref, tstart=tstart, tend=tend, nsteps=nsteps,
                  nwalkers=nwalkers, ntemps=ntemps,
                  log10temperature_min=log10temperature_min)
mcmc.run()
mcmc.plot_corner(add_prior=True)
mcmc.print_summary()
mcmc.compute_evidence()