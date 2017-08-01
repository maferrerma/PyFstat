""" The core tools used in pyfstat """
import os
import logging
import copy
import glob

import numpy as np

# workaround for matplotlib on X-less remote logins
if os.environ.has_key('DISPLAY'):
    import matplotlib.pyplot as plt
else:
    logging.info('No $DISPLAY environment variable found, \
                  so importing matplotlib.pyplot with non-interactive "Agg" backend.')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import scipy.special
import scipy.optimize
import lal
import lalpulsar

import helper_functions
helper_functions.set_up_matplotlib_defaults()
args, tqdm = helper_functions.set_up_command_line_arguments()
earth_ephem, sun_ephem = helper_functions.set_up_ephemeris_configuration()
detector_colors = {'h1': 'C0', 'l1': 'C1'}


def read_par(label=None, outdir=None, filename=None, suffix='par'):
    """ Read in a .par file, returns a dictionary of the values

    Note, can also read in .loudest files
    """
    if filename is None:
        filename = '{}/{}.{}'.format(outdir, label, suffix)
    if os.path.isfile(filename) is False:
        raise ValueError("No file ({}) found".format(filename))
    d = {}
    with open(filename, 'r') as f:
        d = get_dictionary_from_lines(f)
    return d


def get_dictionary_from_lines(lines):
    d = {}
    for line in lines:
        if line[0] not in ['%', '#'] and len(line.split('=')) == 2:
            try:
                key, val = line.rstrip('\n').split('=')
                key = key.strip()
                d[key] = np.float64(eval(val.rstrip('; ')))
            except SyntaxError:
                pass
    return d


def predict_fstat(h0, cosi, psi, Alpha, Delta, Freq, sftfilepattern,
                  minStartTime, maxStartTime, IFO=None, assumeSqrtSX=None,
                  **kwargs):
    """ Wrapper to lalapps_PredictFstat """
    cl_pfs = []
    cl_pfs.append("lalapps_PredictFstat")
    cl_pfs.append("--h0={}".format(h0))
    cl_pfs.append("--cosi={}".format(cosi))
    cl_pfs.append("--psi={}".format(psi))
    cl_pfs.append("--Alpha={}".format(Alpha))
    cl_pfs.append("--Delta={}".format(Delta))
    cl_pfs.append("--Freq={}".format(Freq))

    cl_pfs.append("--DataFiles='{}'".format(sftfilepattern))
    if assumeSqrtSX:
        cl_pfs.append("--assumeSqrtSX={}".format(assumeSqrtSX))
    if IFO:
        cl_pfs.append("--IFO={}".format(IFO))

    cl_pfs.append("--minStartTime={}".format(int(minStartTime)))
    cl_pfs.append("--maxStartTime={}".format(int(maxStartTime)))
    cl_pfs.append("--outputFstat=/tmp/fs")

    cl_pfs = " ".join(cl_pfs)
    helper_functions.run_commandline(cl_pfs)
    d = read_par(filename='/tmp/fs')
    return float(d['twoF_expected']), float(d['twoF_sigma'])


class BaseSearchClass(object):
    """ The base search class, provides general functions """

    earth_ephem_default = earth_ephem
    sun_ephem_default = sun_ephem

    def _add_log_file(self):
        """ Log output to a file, requires class to have outdir and label """
        logfilename = '{}/{}.log'.format(self.outdir, self.label)
        fh = logging.FileHandler(logfilename)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)-8s: %(message)s',
            datefmt='%y-%m-%d %H:%M'))
        logging.getLogger().addHandler(fh)

    def _shift_matrix(self, n, dT):
        """ Generate the shift matrix

        Parameters
        ----------
        n: int
            The dimension of the shift-matrix to generate
        dT: float
            The time delta of the shift matrix

        Returns
        -------
        m: array (n, n)
            The shift matrix
        """

        m = np.zeros((n, n))
        factorial = np.math.factorial
        for i in range(n):
            for j in range(n):
                if i == j:
                    m[i, j] = 1.0
                elif i > j:
                    m[i, j] = 0.0
                else:
                    if i == 0:
                        m[i, j] = 2*np.pi*float(dT)**(j-i) / factorial(j-i)
                    else:
                        m[i, j] = float(dT)**(j-i) / factorial(j-i)
        return m

    def _shift_coefficients(self, theta, dT):
        """ Shift a set of coefficients by dT

        Parameters
        ----------
        theta: array-like, shape (n,)
            vector of the expansion coefficients to transform starting from the
            lowest degree e.g [phi, F0, F1,...].
        dT: float
            difference between the two reference times as tref_new - tref_old.

        Returns
        -------
        theta_new: array-like shape (n,)
            vector of the coefficients as evaluate as the new reference time.
        """

        n = len(theta)
        m = self._shift_matrix(n, dT)
        return np.dot(m, theta)

    def _calculate_thetas(self, theta, delta_thetas, tbounds, theta0_idx=0):
        """ Calculates the set of coefficients for the post-glitch signal """
        thetas = [theta]
        for i, dt in enumerate(delta_thetas):
            if i < theta0_idx:
                pre_theta_at_ith_glitch = self._shift_coefficients(
                    thetas[0], tbounds[i+1] - self.tref)
                post_theta_at_ith_glitch = pre_theta_at_ith_glitch - dt
                thetas.insert(0, self._shift_coefficients(
                    post_theta_at_ith_glitch, self.tref - tbounds[i+1]))

            elif i >= theta0_idx:
                pre_theta_at_ith_glitch = self._shift_coefficients(
                    thetas[i], tbounds[i+1] - self.tref)
                post_theta_at_ith_glitch = pre_theta_at_ith_glitch + dt
                thetas.append(self._shift_coefficients(
                    post_theta_at_ith_glitch, self.tref - tbounds[i+1]))
        self.thetas_at_tref = thetas
        return thetas

    def _get_list_of_matching_sfts(self):
        """ Returns a list of sfts matching the sftfilepattern """
        sftfilepatternlist = np.atleast_1d(self.sftfilepattern.split(';'))
        matches = [glob.glob(p) for p in sftfilepatternlist]
        matches = [item for sublist in matches for item in sublist]
        if len(matches) > 0:
            return matches
        else:
            raise IOError('No sfts found matching {}'.format(
                self.sftfilepattern))


class ComputeFstat(object):
    """ Base class providing interface to `lalpulsar.ComputeFstat` """

    earth_ephem_default = earth_ephem
    sun_ephem_default = sun_ephem

    @helper_functions.initializer
    def __init__(self, tref, sftfilepattern=None, minStartTime=None,
                 maxStartTime=None, binary=False, transient=True, BSGL=False,
                 detectors=None, minCoverFreq=None, maxCoverFreq=None,
                 earth_ephem=None, sun_ephem=None, injectSources=None,
                 injectSqrtSX=None, assumeSqrtSX=None, SSBprec=None):
        """
        Parameters
        ----------
        tref: int
            GPS seconds of the reference time.
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        minStartTime, maxStartTime: float GPStime
            Only use SFTs with timestemps starting from (including, excluding)
            this epoch
        binary: bool
            If true, search of binary parameters.
        transient: bool
            If true, allow for the Fstat to be computed over a transient range.
        BSGL: bool
            If true, compute the BSGL rather than the twoF value.
        detectors: str
            Two character reference to the data to use, specify None for no
            contraint. If multiple-separate by comma.
        minCoverFreq, maxCoverFreq: float
            The min and max cover frequency passed to CreateFstatInput, if
            either is None the range of frequencies in the SFT less 1Hz is
            used.
        earth_ephem, sun_ephem: str
            Paths of the two files containing positions of Earth and Sun,
            respectively at evenly spaced times, as passed to CreateFstatInput.
            If None defaults defined in BaseSearchClass will be used.
        injectSources: dict or str
            Either a dictionary of the values to inject, or a string pointing
            to the .cff file to inject
        injectSqrtSX:
            Not yet implemented
        assumeSqrtSX: float
            Don't estimate noise-floors but assume (stationary) per-IFO
            sqrt{SX} (if single value: use for all IFOs). If signal only,
            set sqrtSX=1
        SSBprec: int
            Flag to set the SSB calculation: 0=Newtonian, 1=relativistic,
            2=relativisitic optimised, 3=DMoff, 4=NO_SPIN

        """

        if earth_ephem is None:
            self.earth_ephem = self.earth_ephem_default
        if sun_ephem is None:
            self.sun_ephem = self.sun_ephem_default

        self.init_computefstatistic_single_point()

    def get_SFTCatalog(self):
        if hasattr(self, 'SFTCatalog'):
            return
        if self.sftfilepattern is None:
            for k in ['minStartTime', 'maxStartTime', 'detectors']:
                if getattr(self, k) is None:
                    raise ValueError('You must provide "{}" to injectSources'
                                     .format(k))
            C1 = getattr(self, 'injectSources', None) is None
            C2 = getattr(self, 'injectSqrtSX', None) is None
            if C1 and C2:
                raise ValueError('You must specify either one of injectSources'
                                 ' or injectSqrtSX')
            SFTCatalog = lalpulsar.SFTCatalog()
            Tsft = 1800
            Toverlap = 0
            Tspan = self.maxStartTime - self.minStartTime
            detNames = lal.CreateStringVector(
                *[d for d in self.detectors.split(',')])
            multiTimestamps = lalpulsar.MakeMultiTimestamps(
                self.minStartTime, Tspan, Tsft, Toverlap, detNames.length)
            SFTCatalog = lalpulsar.MultiAddToFakeSFTCatalog(
                SFTCatalog, detNames, multiTimestamps)
            return SFTCatalog

        logging.info('Initialising SFTCatalog')
        constraints = lalpulsar.SFTConstraints()
        if self.detectors:
            if ',' in self.detectors:
                logging.info('Using all detector data')
            else:
                constraints.detector = self.detectors
        if self.minStartTime:
            constraints.minStartTime = lal.LIGOTimeGPS(self.minStartTime)
        if self.maxStartTime:
            constraints.maxStartTime = lal.LIGOTimeGPS(self.maxStartTime)

        logging.info('Loading data matching pattern {}'.format(
                     self.sftfilepattern))
        SFTCatalog = lalpulsar.SFTdataFind(self.sftfilepattern, constraints)
        detector_names = list(set([d.header.name for d in SFTCatalog.data]))
        self.detector_names = detector_names
        SFT_timestamps = [d.header.epoch for d in SFTCatalog.data]
        self.SFT_timestamps = [float(s) for s in SFT_timestamps]
        if len(SFT_timestamps) == 0:
            raise ValueError('Failed to load any data')
        if args.quite is False and args.no_interactive is False:
            try:
                from bashplotlib.histogram import plot_hist
                print('Data timestamps histogram:')
                plot_hist(SFT_timestamps, height=5, bincount=50)
            except ImportError:
                pass
        if len(detector_names) == 0:
            raise ValueError('No data loaded.')
        logging.info('Loaded {} data files from detectors {}'.format(
            len(SFT_timestamps), detector_names))
        cl_tconv1 = 'lalapps_tconvert {}'.format(int(SFT_timestamps[0]))
        output    = helper_functions.run_commandline(cl_tconv1)
        tconvert1 = output.rstrip('\n')
        cl_tconv2 = 'lalapps_tconvert {}'.format(int(SFT_timestamps[-1]))
        output    = helper_functions.run_commandline(cl_tconv2)
        tconvert2 = output.rstrip('\n')
        logging.info('Data spans from {} ({}) to {} ({})'.format(
            int(SFT_timestamps[0]),
            tconvert1,
            int(SFT_timestamps[-1]),
            tconvert2))
        return SFTCatalog

    def init_computefstatistic_single_point(self):
        """ Initilisation step of run_computefstatistic for a single point """

        SFTCatalog = self.get_SFTCatalog()

        logging.info('Initialising ephems')
        ephems = lalpulsar.InitBarycenter(self.earth_ephem, self.sun_ephem)

        logging.info('Initialising FstatInput')
        dFreq = 0
        if self.transient:
            self.whatToCompute = lalpulsar.FSTATQ_ATOMS_PER_DET
        else:
            self.whatToCompute = lalpulsar.FSTATQ_2F

        FstatOAs = lalpulsar.FstatOptionalArgs()
        FstatOAs.randSeed = lalpulsar.FstatOptionalArgsDefaults.randSeed
        if self.SSBprec:
            logging.info('Using SSBprec={}'.format(self.SSBprec))
            FstatOAs.SSBprec = self.SSBprec
        else:
            FstatOAs.SSBprec = lalpulsar.FstatOptionalArgsDefaults.SSBprec
        FstatOAs.Dterms = lalpulsar.FstatOptionalArgsDefaults.Dterms
        FstatOAs.runningMedianWindow = lalpulsar.FstatOptionalArgsDefaults.runningMedianWindow
        FstatOAs.FstatMethod = lalpulsar.FstatOptionalArgsDefaults.FstatMethod
        if self.assumeSqrtSX is None:
            FstatOAs.assumeSqrtSX = lalpulsar.FstatOptionalArgsDefaults.assumeSqrtSX
        else:
            mnf = lalpulsar.MultiNoiseFloor()
            assumeSqrtSX = np.atleast_1d(self.assumeSqrtSX)
            mnf.sqrtSn[:len(assumeSqrtSX)] = assumeSqrtSX
            mnf.length = len(assumeSqrtSX)
            FstatOAs.assumeSqrtSX = mnf
        FstatOAs.prevInput = lalpulsar.FstatOptionalArgsDefaults.prevInput
        FstatOAs.collectTiming = lalpulsar.FstatOptionalArgsDefaults.collectTiming

        if hasattr(self, 'injectSources') and type(self.injectSources) == dict:
            logging.info('Injecting source with params: {}'.format(
                self.injectSources))
            PPV = lalpulsar.CreatePulsarParamsVector(1)
            PP = PPV.data[0]
            PP.Amp.h0 = self.injectSources['h0']
            PP.Amp.cosi = self.injectSources['cosi']
            PP.Amp.phi0 = self.injectSources['phi0']
            PP.Amp.psi = self.injectSources['psi']
            PP.Doppler.Alpha = self.injectSources['Alpha']
            PP.Doppler.Delta = self.injectSources['Delta']
            PP.Doppler.fkdot = np.array(self.injectSources['fkdot'])
            PP.Doppler.refTime = self.tref
            if 't0' not in self.injectSources:
                PP.Transient.type = lalpulsar.TRANSIENT_NONE
            FstatOAs.injectSources = PPV
        elif hasattr(self, 'injectSources') and type(self.injectSources) == str:
            logging.info('Injecting source from param file: {}'.format(
                self.injectSources))
            PPV = lalpulsar.PulsarParamsFromFile(self.injectSources, self.tref)
            FstatOAs.injectSources = PPV
        else:
            FstatOAs.injectSources = lalpulsar.FstatOptionalArgsDefaults.injectSources
        if hasattr(self, 'injectSqrtSX') and self.injectSqrtSX is not None:
            raise ValueError('injectSqrtSX not implemented')
        else:
            FstatOAs.InjectSqrtSX = lalpulsar.FstatOptionalArgsDefaults.injectSqrtSX
        if self.minCoverFreq is None or self.maxCoverFreq is None:
            fAs = [d.header.f0 for d in SFTCatalog.data]
            fBs = [d.header.f0 + (d.numBins-1)*d.header.deltaF
                   for d in SFTCatalog.data]
            self.minCoverFreq = np.min(fAs) + 0.5
            self.maxCoverFreq = np.max(fBs) - 0.5
            logging.info('Min/max cover freqs not provided, using '
                         '{} and {}, est. from SFTs'.format(
                             self.minCoverFreq, self.maxCoverFreq))

        self.FstatInput = lalpulsar.CreateFstatInput(SFTCatalog,
                                                     self.minCoverFreq,
                                                     self.maxCoverFreq,
                                                     dFreq,
                                                     ephems,
                                                     FstatOAs
                                                     )

        logging.info('Initialising PulsarDoplerParams')
        PulsarDopplerParams = lalpulsar.PulsarDopplerParams()
        PulsarDopplerParams.refTime = self.tref
        PulsarDopplerParams.Alpha = 1
        PulsarDopplerParams.Delta = 1
        PulsarDopplerParams.fkdot = np.array([0, 0, 0, 0, 0, 0, 0])
        self.PulsarDopplerParams = PulsarDopplerParams

        logging.info('Initialising FstatResults')
        self.FstatResults = lalpulsar.FstatResults()

        if self.BSGL:
            if len(self.detector_names) < 2:
                raise ValueError("Can't use BSGL with single detectors data")
            else:
                logging.info('Initialising BSGL')

            # Tuning parameters - to be reviewed
            numDetectors = 2
            if hasattr(self, 'nsegs'):
                p_val_threshold = 1e-6
                Fstar0s = np.linspace(0, 1000, 10000)
                p_vals = scipy.special.gammaincc(2*self.nsegs, Fstar0s)
                Fstar0 = Fstar0s[np.argmin(np.abs(p_vals - p_val_threshold))]
                if Fstar0 == Fstar0s[-1]:
                    raise ValueError('Max Fstar0 exceeded')
            else:
                Fstar0 = 15.
            logging.info('Using Fstar0 of {:1.2f}'.format(Fstar0))
            oLGX = np.zeros(10)
            oLGX[:numDetectors] = 1./numDetectors
            self.BSGLSetup = lalpulsar.CreateBSGLSetup(numDetectors,
                                                       Fstar0,
                                                       oLGX,
                                                       True,
                                                       1)
            self.twoFX = np.zeros(10)
            self.whatToCompute = (self.whatToCompute +
                                  lalpulsar.FSTATQ_2F_PER_DET)

        if self.transient:
            logging.info('Initialising transient parameters')
            self.windowRange = lalpulsar.transientWindowRange_t()
            self.windowRange.type = lalpulsar.TRANSIENT_RECTANGULAR
            self.windowRange.t0Band = 0
            self.windowRange.dt0 = 1
            self.windowRange.tauBand = 0
            self.windowRange.dtau = 1

    def compute_fullycoherent_det_stat_single_point(
            self, F0, F1, F2, Alpha, Delta, asini=None, period=None, ecc=None,
            tp=None, argp=None):
        """ Compute the fully-coherent det. statistic at a single point """

        return self.run_computefstatistic_single_point(
            self.minStartTime, self.maxStartTime, F0, F1, F2, Alpha, Delta,
            asini, period, ecc, tp, argp)

    def run_computefstatistic_single_point(self, tstart, tend, F0, F1,
                                           F2, Alpha, Delta, asini=None,
                                           period=None, ecc=None, tp=None,
                                           argp=None):
        """ Returns twoF or ln(BSGL) fully-coherently at a single point """

        self.PulsarDopplerParams.fkdot = np.array([F0, F1, F2, 0, 0, 0, 0])
        self.PulsarDopplerParams.Alpha = Alpha
        self.PulsarDopplerParams.Delta = Delta
        if self.binary:
            self.PulsarDopplerParams.asini = asini
            self.PulsarDopplerParams.period = period
            self.PulsarDopplerParams.ecc = ecc
            self.PulsarDopplerParams.tp = tp
            self.PulsarDopplerParams.argp = argp

        lalpulsar.ComputeFstat(self.FstatResults,
                               self.FstatInput,
                               self.PulsarDopplerParams,
                               1,
                               self.whatToCompute
                               )

        if self.transient is False:
            if self.BSGL is False:
                return self.FstatResults.twoF[0]

            twoF = np.float(self.FstatResults.twoF[0])
            self.twoFX[0] = self.FstatResults.twoFPerDet(0)
            self.twoFX[1] = self.FstatResults.twoFPerDet(1)
            log10_BSGL = lalpulsar.ComputeBSGL(twoF, self.twoFX,
                                               self.BSGLSetup)
            return log10_BSGL/np.log10(np.exp(1))

        self.windowRange.t0 = int(tstart)  # TYPE UINT4
        self.windowRange.tau = int(tend - tstart)  # TYPE UINT4

        FS = lalpulsar.ComputeTransientFstatMap(
            self.FstatResults.multiFatoms[0], self.windowRange, False)

        if self.BSGL is False:
            twoF = 2*FS.F_mn.data[0][0]
            if np.isnan(twoF):
                return 0
            else:
                return twoF

        FstatResults_single = copy.copy(self.FstatResults)
        FstatResults_single.lenth = 1
        FstatResults_single.data = self.FstatResults.multiFatoms[0].data[0]
        FS0 = lalpulsar.ComputeTransientFstatMap(
            FstatResults_single.multiFatoms[0], self.windowRange, False)
        FstatResults_single.data = self.FstatResults.multiFatoms[0].data[1]
        FS1 = lalpulsar.ComputeTransientFstatMap(
            FstatResults_single.multiFatoms[0], self.windowRange, False)

        self.twoFX[0] = 2*FS0.F_mn.data[0][0]
        self.twoFX[1] = 2*FS1.F_mn.data[0][0]
        log10_BSGL = lalpulsar.ComputeBSGL(
                2*FS.F_mn.data[0][0], self.twoFX, self.BSGLSetup)

        return log10_BSGL/np.log10(np.exp(1))

    def calculate_twoF_cumulative(self, F0, F1, F2, Alpha, Delta, asini=None,
                                  period=None, ecc=None, tp=None, argp=None,
                                  tstart=None, tend=None, npoints=1000,
                                  ):
        """ Calculate the cumulative twoF along the obseration span
        Params
        ------
        F0, F1, F2, Alpha, Delta: float
            Parameters at which to compute the cumulative twoF
        asini, period, ecc, tp, argp: float
            Binary parameters at which to compute the cumulative twoF (default
            to None)
        tstart, tend: int
            GPS times to restrict the range of data used - automatically
            truncated to the span of data available
        npoints: int
            Number of points to compute twoF along the span

        Note: the minimum cumulatibe twoF is hard-coded to be computed over
        the first 6 hours from either the first timestampe in the data (if
        tstart is smaller than it) or tstart.

        """
        SFTminStartTime = self.SFT_timestamps[0]
        SFTmaxStartTime = self.SFT_timestamps[-1]
        tstart = np.max([SFTminStartTime, tstart])
        min_tau = np.max([SFTminStartTime - tstart, 0]) + 3600*6
        max_tau = SFTmaxStartTime - tstart
        taus = np.linspace(min_tau, max_tau, npoints)
        twoFs = []
        if self.transient is False:
            self.transient = True
            self.init_computefstatistic_single_point()
        for tau in taus:
            twoFs.append(self.run_computefstatistic_single_point(
                tstart=tstart, tend=tstart+tau, F0=F0, F1=F1, F2=F2,
                Alpha=Alpha, Delta=Delta, asini=asini, period=period, ecc=ecc,
                tp=tp, argp=argp))

        return taus, np.array(twoFs)

    def calculate_pfs(self, label, outdir, N=15, IFO=None, pfs_input=None):

        if pfs_input is None:
            if os.path.isfile('{}/{}.loudest'.format(outdir, label)) is False:
                raise ValueError(
                    'Need a loudest file to add the predicted Fstat')
            loudest = read_par(label, outdir, suffix='loudest')
            pfs_input = {key: loudest[key] for key in
                         ['h0', 'cosi', 'psi', 'Alpha', 'Delta', 'Freq']}
        times = np.linspace(self.minStartTime, self.maxStartTime, N+1)[1:]
        times = np.insert(times, 0, self.minStartTime + 86400/2.)
        out = [predict_fstat(minStartTime=self.minStartTime, maxStartTime=t,
                             sftfilepattern=self.sftfilepattern, IFO=IFO,
                             **pfs_input) for t in times]
        pfs, pfs_sigma = np.array(out).T
        return times, pfs, pfs_sigma

    def plot_twoF_cumulative(self, label, outdir, ax=None, c='k', savefig=True,
                             title=None, add_pfs=False, N=15,
                             injectSources=None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        if injectSources:
            pfs_input = dict(
                h0=injectSources['h0'], cosi=injectSources['cosi'],
                psi=injectSources['psi'], Alpha=injectSources['Alpha'],
                Delta=injectSources['Delta'], Freq=injectSources['fkdot'][0])
        else:
            pfs_input = None

        taus, twoFs = self.calculate_twoF_cumulative(**kwargs)
        ax.plot(taus/86400., twoFs, label='All detectors', color=c)
        if len(self.detector_names) > 1:
            detector_names = self.detector_names
            detectors = self.detectors
            for d in self.detector_names:
                self.detectors = d
                self.init_computefstatistic_single_point()
                taus, twoFs = self.calculate_twoF_cumulative(**kwargs)
                ax.plot(taus/86400., twoFs, label='{}'.format(d),
                        color=detector_colors[d.lower()])
            self.detectors = detectors
            self.detector_names = detector_names

        if add_pfs:
            times, pfs, pfs_sigma = self.calculate_pfs(
                label, outdir, N=N, pfs_input=pfs_input)
            ax.fill_between(
                (times-self.minStartTime)/86400., pfs-pfs_sigma, pfs+pfs_sigma,
                color=c,
                label=r'Predicted $\langle 2\mathcal{F} \rangle\pm $ 1-$\sigma$ band',
                zorder=-10, alpha=0.2)
            if len(self.detector_names) > 1:
                for d in self.detector_names:
                    times, pfs, pfs_sigma = self.calculate_pfs(
                        label, outdir, IFO=d.upper(), N=N, pfs_input=pfs_input)
                    ax.fill_between(
                        (times-self.minStartTime)/86400., pfs-pfs_sigma,
                        pfs+pfs_sigma, color=detector_colors[d.lower()],
                        alpha=0.5,
                        label=(
                            'Predicted $2\mathcal{{F}}$ 1-$\sigma$ band ({})'
                            .format(d.upper())),
                        zorder=-10)

        ax.set_xlabel(r'Days from $t_{{\rm start}}={:.0f}$'.format(
            kwargs['tstart']))
        if self.BSGL:
            ax.set_ylabel(r'$\log_{10}(\mathrm{BSGL})_{\rm cumulative}$')
        else:
            ax.set_ylabel(r'$\widetilde{2\mathcal{F}}_{\rm cumulative}$')
        ax.set_xlim(0, taus[-1]/86400)
        ax.legend(frameon=False, loc=2, fontsize=6)
        if title:
            ax.set_title(title)
        if savefig:
            plt.tight_layout()
            plt.savefig('{}/{}_twoFcumulative.png'.format(outdir, label))
            return taus, twoFs
        else:
            return ax


class SemiCoherentSearch(BaseSearchClass, ComputeFstat):
    """ A semi-coherent search """

    @helper_functions.initializer
    def __init__(self, label, outdir, tref, nsegs=None, sftfilepattern=None,
                 binary=False, BSGL=False, minStartTime=None,
                 maxStartTime=None, minCoverFreq=None, maxCoverFreq=None,
                 detectors=None, earth_ephem=None, sun_ephem=None,
                 injectSources=None, assumeSqrtSX=None, SSBprec=None):
        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to.
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, and start and end of the data.
        nsegs: int
            The (fixed) number of segments
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.

        For all other parameters, see pyfstat.ComputeFStat.
        """

        self.fs_file_name = "{}/{}_FS.dat".format(self.outdir, self.label)
        if self.earth_ephem is None:
            self.earth_ephem = self.earth_ephem_default
        if self.sun_ephem is None:
            self.sun_ephem = self.sun_ephem_default
        self.transient = True
        self.init_computefstatistic_single_point()
        self.init_semicoherent_parameters()

    def init_semicoherent_parameters(self):
        logging.info(('Initialising semicoherent parameters from {} to {} in'
                      ' {} segments').format(
            self.minStartTime, self.maxStartTime, self.nsegs))
        self.transient = True
        self.whatToCompute = lalpulsar.FSTATQ_2F+lalpulsar.FSTATQ_ATOMS_PER_DET
        self.tboundaries = np.linspace(self.minStartTime, self.maxStartTime,
                                       self.nsegs+1)

    def run_semi_coherent_computefstatistic_single_point(
            self, F0, F1, F2, Alpha, Delta, asini=None,
            period=None, ecc=None, tp=None, argp=None,
            record_segments=False):
        """ Returns twoF or ln(BSGL) semi-coherently at a single point """

        if hasattr(self, 'SFT_timestamps'):
            if self.tboundaries[0] < self.SFT_timestamps[0]:
                logging.debug(
                    'Semi-coherent start time {} before first SFT timestamp {}'
                    .format(self.tboundaries[0], self.SFT_timestamps[0]))
            if self.tboundaries[-1] > self.SFT_timestamps[-1]:
                logging.debug(
                    'Semi-coherent end time {} after last SFT timestamp {}'
                    .format(self.tboundaries[-1], self.SFT_timestamps[-1]))

        self.PulsarDopplerParams.fkdot = np.array([F0, F1, F2, 0, 0, 0, 0])
        self.PulsarDopplerParams.Alpha = Alpha
        self.PulsarDopplerParams.Delta = Delta
        if self.binary:
            self.PulsarDopplerParams.asini = asini
            self.PulsarDopplerParams.period = period
            self.PulsarDopplerParams.ecc = ecc
            self.PulsarDopplerParams.tp = tp
            self.PulsarDopplerParams.argp = argp

        lalpulsar.ComputeFstat(self.FstatResults,
                               self.FstatInput,
                               self.PulsarDopplerParams,
                               1,
                               self.whatToCompute
                               )

        #if self.transient is False:
        #    if self.BSGL is False:
        #        return self.FstatResults.twoF[0]
        #    twoF = np.float(self.FstatResults.twoF[0])
        #    self.twoFX[0] = self.FstatResults.twoFPerDet(0)
        #    self.twoFX[1] = self.FstatResults.twoFPerDet(1)
        #    log10_BSGL = lalpulsar.ComputeBSGL(twoF, self.twoFX,
        #                                       self.BSGLSetup)
        #    return log10_BSGL/np.log10(np.exp(1))

        detStat = 0
        if record_segments:
            self.detStat_per_segment = []
        for tstart, tend in zip(self.tboundaries[:-1], self.tboundaries[1:]):
            self.windowRange.t0 = int(tstart)  # TYPE UINT4
            self.windowRange.tau = int(tend - tstart)  # TYPE UINT4

            FS = lalpulsar.ComputeTransientFstatMap(
                self.FstatResults.multiFatoms[0], self.windowRange, False)

            if self.BSGL is False:
                d_detStat = 2*FS.F_mn.data[0][0]
            else:
                FstatResults_single = copy.copy(self.FstatResults)
                FstatResults_single.lenth = 1
                FstatResults_single.data = self.FstatResults.multiFatoms[0].data[0]
                FS0 = lalpulsar.ComputeTransientFstatMap(
                    FstatResults_single.multiFatoms[0], self.windowRange, False)
                FstatResults_single.data = self.FstatResults.multiFatoms[0].data[1]
                FS1 = lalpulsar.ComputeTransientFstatMap(
                    FstatResults_single.multiFatoms[0], self.windowRange, False)

                self.twoFX[0] = 2*FS0.F_mn.data[0][0]
                self.twoFX[1] = 2*FS1.F_mn.data[0][0]
                log10_BSGL = lalpulsar.ComputeBSGL(
                        2*FS.F_mn.data[0][0], self.twoFX, self.BSGLSetup)
                d_detStat = log10_BSGL/np.log10(np.exp(1))
            if np.isnan(d_detStat):
                logging.debug('NaNs in semi-coherent twoF treated as zero')
                d_detStat = 0
            detStat += d_detStat
            if record_segments:
                self.detStat_per_segment.append(d_detStat)

        return detStat


class SemiCoherentGlitchSearch(BaseSearchClass, ComputeFstat):
    """ A semi-coherent glitch search

    This implements a basic `semi-coherent glitch F-stat in which the data
    is divided into segments either side of the proposed glitches and the
    fully-coherent F-stat in each segment is summed to give the semi-coherent
    F-stat
    """

    @helper_functions.initializer
    def __init__(self, label, outdir, tref, minStartTime, maxStartTime,
                 nglitch=0, sftfilepattern=None, theta0_idx=0, BSGL=False,
                 minCoverFreq=None, maxCoverFreq=None, assumeSqrtSX=None,
                 detectors=None, earth_ephem=None, sun_ephem=None,
                 SSBprec=None, injectSources=None):
        """
        Parameters
        ----------
        label, outdir: str
            A label and directory to read/write data from/to.
        tref, minStartTime, maxStartTime: int
            GPS seconds of the reference time, and start and end of the data.
        nglitch: int
            The (fixed) number of glitches; this can zero, but occasionally
            this causes issue (in which case just use ComputeFstat).
        sftfilepattern: str
            Pattern to match SFTs using wildcards (*?) and ranges [0-9];
            mutiple patterns can be given separated by colons.
        theta0_idx, int
            Index (zero-based) of which segment the theta refers to - uyseful
            if providing a tight prior on theta to allow the signal to jump
            too theta (and not just from)

        For all other parameters, see pyfstat.ComputeFStat.
        """

        self.fs_file_name = "{}/{}_FS.dat".format(self.outdir, self.label)
        if self.earth_ephem is None:
            self.earth_ephem = self.earth_ephem_default
        if self.sun_ephem is None:
            self.sun_ephem = self.sun_ephem_default
        self.transient = True
        self.binary = False
        self.init_computefstatistic_single_point()

    def compute_nglitch_fstat(self, F0, F1, F2, Alpha, Delta, *args):
        """ Returns the semi-coherent glitch summed twoF """

        args = list(args)
        tboundaries = ([self.minStartTime] + args[-self.nglitch:]
                       + [self.maxStartTime])
        delta_F0s = args[-3*self.nglitch:-2*self.nglitch]
        delta_F1s = args[-2*self.nglitch:-self.nglitch]
        delta_F2 = np.zeros(len(delta_F0s))
        delta_phi = np.zeros(len(delta_F0s))
        theta = [0, F0, F1, F2]
        delta_thetas = np.atleast_2d(
                np.array([delta_phi, delta_F0s, delta_F1s, delta_F2]).T)

        thetas = self._calculate_thetas(theta, delta_thetas, tboundaries,
                                       theta0_idx=self.theta0_idx)

        twoFSum = 0
        for i, theta_i_at_tref in enumerate(thetas):
            ts, te = tboundaries[i], tboundaries[i+1]

            twoFVal = self.run_computefstatistic_single_point(
                ts, te, theta_i_at_tref[1], theta_i_at_tref[2],
                theta_i_at_tref[3], Alpha, Delta)
            twoFSum += twoFVal

        if np.isfinite(twoFSum):
            return twoFSum
        else:
            return -np.inf

    def compute_glitch_fstat_single(self, F0, F1, F2, Alpha, Delta, delta_F0,
                                    delta_F1, tglitch):
        """ Returns the semi-coherent glitch summed twoF for nglitch=1

        Note: OBSOLETE, used only for testing
        """

        theta = [F0, F1, F2]
        delta_theta = [delta_F0, delta_F1, 0]
        tref = self.tref

        theta_at_glitch = self._shift_coefficients(theta, tglitch - tref)
        theta_post_glitch_at_glitch = theta_at_glitch + delta_theta
        theta_post_glitch = self._shift_coefficients(
            theta_post_glitch_at_glitch, tref - tglitch)

        twoFsegA = self.run_computefstatistic_single_point(
            self.minStartTime, tglitch, theta[0], theta[1], theta[2], Alpha,
            Delta)

        if tglitch == self.maxStartTime:
            return twoFsegA

        twoFsegB = self.run_computefstatistic_single_point(
            tglitch, self.maxStartTime, theta_post_glitch[0],
            theta_post_glitch[1], theta_post_glitch[2], Alpha,
            Delta)

        return twoFsegA + twoFsegB
