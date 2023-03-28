import numpy as np
import matplotlib.pyplot as plt
import os, sys
import enterprise
from enterprise.signals.signal_base import function as enterprise_function, PTA
from enterprise.signals.deterministic_signals import Deterministic
from enterprise.signals.parameter import Uniform
from enterprise.signals.gp_signals import MarginalizingTimingModel
from enterprise.signals.white_signals import MeasurementNoise
from enterprise.pulsar import Pulsar
import enterprise
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc
import nestle

sys.path.append('./GW_hyp/')
from gw_hyp import hyp_pta_res
import json
import os, shutil

import libstempo as lst
import libstempo.plot as lstplot
import libstempo.toasim as toasim




code = sys.argv[0]
index = int(sys.argv[1])


datadir=f"tempo2_fake_tim/"

file=np.loadtxt('psrlist.txt',dtype=object)

psrname=file[index]

parfile = f"{datadir}/"+psrname+".par"
timfile = f"{datadir}/"+psrname+".tim"

psr = Pulsar(parfile, timfile)


def hms_to_rad(hh, mm, ss):
    sgn = np.sign(hh)
    return sgn * (sgn * hh + mm / 60 + ss / 3600) * np.pi / 12


def dms_to_rad(dd, mm, ss):
    sgn = np.sign(dd)
    return sgn * (sgn * dd + mm / 60 + ss / 3600) * np.pi / 180


from gw_hyp import hyp_pta_res


RA_GW = hms_to_rad(4, 0, 0)
DEC_GW = dms_to_rad(-45, 0, 0)
tref1 = (max(psr.toas)+min(psr.toas))/2


inc0=0;M0=1e10;q0=1;b0=70;e0=1.15;S0=1e-6;

hyp_gw =hyp_pta_res(
    cos_gwtheta=np.sin(DEC_GW),
    gwphi=RA_GW,
    psi=0,
    cos_inc=np.cos(inc0),
    log10_M=np.log10(M0),
    q=q0,
    b=b0,
    e0=e0,
    tref=tref1,
    log10_S=np.log10(S0))


hyp_gw_fn = hyp_gw(name="hyp_gw", psr=psr)

res = hyp_gw_fn()

toas = psr.toas / (365.25*24*3600)
t0 = np.mean(toas)




if not os.path.exists('run'):
        os.mkdir('run')


output_dir = ["run/gwhyp_sims_try","run/injected_plots"]
for i in range(len(output_dir)):
    # if os.path.exists(output_dir[i]):
    #     shutil.rmtree(output_dir[i])
    if not os.path.exists(output_dir[i]):
        os.mkdir(output_dir[i])



psrl = lst.tempopulsar(parfile=parfile, timfile=timfile)
print(psrl.name)





def save_psr_sim(psr1, savedir):
    print("Writing simulated data for", psr1.name)
    psr1.savepar(f"{savedir}/{psr1.name}_simulate.par")
    psr1.savetim(f"{savedir}/{psr1.name}_simulate.tim")
    lst.purgetim(f"{savedir}/{psr1.name}_simulate.tim")




day_to_s = 24 * 3600





def add_gwhyp_1psr(psr):
    toas = (psr.toas() * day_to_s).astype(float)

    signal = (
        np.array(res
            
        )/day_to_s
        
    )

    psr.stoas[:] += signal

    return signal



toasim.make_ideal(psrl)

toasim.add_efac(psrl, 1)


signal = add_gwhyp_1psr(psrl)


lstplot.plotres(psrl, label="Residuals")
plt.plot(psrl.toas(), signal * day_to_s * 1e6, c="k", label="Injected signal")
plt.title(psrl.name)
plt.legend()
plt.savefig('run/injected_plots/simulated_'+psrname+'.png',dpi=200)
psrl.fit()

save_psr_sim(psrl, output_dir[0])
