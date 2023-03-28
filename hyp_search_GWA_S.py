from __future__ import division
import sys, os
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.linalg as sl
from enterprise.signals.parameter import Uniform
import enterprise
from enterprise.pulsar import Pulsar
import enterprise.signals.parameter as parameter
from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals.selections import Selection
from enterprise.signals import white_signals
from enterprise.signals import gp_signals
from enterprise.signals.deterministic_signals import Deterministic
import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
from enterprise_extensions.sampler import JumpProposal, get_parameter_groups, save_runtime_info
sys.path.append('./GW_hyp/')
from gw_hyp import hyp_pta_res

def hms_to_rad(hh, mm, ss):
    sgn = np.sign(hh)
    return sgn * (sgn * hh + mm / 60 + ss / 3600) * np.pi / 12


def dms_to_rad(dd, mm, ss):
    sgn = np.sign(dd)
    return sgn * (sgn * dd + mm / 60 + ss / 3600) * np.pi / 180


datadir ='./run/gwhyp_sims_try/'

parfiles = sorted(glob.glob(datadir + '/*.par'))
timfiles = sorted(glob.glob(datadir + '/*.tim'))

psrs = []
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t)
    psrs.append(psr)
    

# find the maximum time span to set GW frequency sampling
tmin = [p.toas.min() for p in psrs]
tmax = [p.toas.max() for p in psrs]
Tspan = np.max(tmax) - np.min(tmin)

##### parameters and priors #####

# white noise parameters
# in this case we just set the value here since all efacs = 1 
# for the MDC data
#efac = parameter.Constant(1.0)

efac = parameter.Constant(1.0)


# white noise
ef = white_signals.MeasurementNoise(efac=efac)


# timing model
tm = gp_signals.TimingModel()

##########################################################################

from gw_hyp import hyp_pta_res

RA_GW = hms_to_rad(4, 0, 0)
DEC_GW = dms_to_rad(-45, 0, 0)



tref1 = (max(psr.toas)+min(psr.toas))/2


inc0=0;q0=1;M0=1e10


def memory_block_hyp(
    cos_gwtheta=np.sin(DEC_GW),
    gwphi=RA_GW,
    psi=0,
    cos_inc=np.cos(inc0),
    log10_M=np.log10(M0),
    q=q0,
    b=Uniform(50,100)("hyp_b"),
    e0=Uniform(1.1,1.3)("hyp_e"),
    log10_S=Uniform(-7,-5)("hyp_S"),
    tref=tref1):
    return Deterministic(hyp_pta_res(cos_gwtheta=cos_gwtheta,gwphi=gwphi,psi=psi,cos_inc=cos_inc
                                     ,log10_M=log10_M,q=q,b=b,e0=e0,log10_S=log10_S,tref=tref),name="hyp")


hyp = memory_block_hyp()

#############################
# full model is sum of components
model = ef + tm +hyp


# initialize PTA
pta = signal_base.PTA([model(psr) for psr in psrs])


# initial parameters
xs = {par.name: par.sample() for par in pta.params}

# dimension of parameter space
ndim = len(xs)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.01**2)

# set up jump groups by red noise groups 
ndim = len(xs)
groups  = [range(0, ndim)]
groups.extend(map(list, zip(range(0,ndim,2), range(1,ndim,2))))

sampler = ptmcmc(ndim, pta.get_lnlikelihood, pta.get_lnprior, cov, groups=groups, 
                 outDir='./run/chains/mdc/open1_hyp/')

# sampler for N steps
N = int(1e5)
x0 = np.hstack(p.sample() for p in pta.params)

sampler.sample(x0, N, SCAMweight=30, AMweight=15, DEweight=50)

