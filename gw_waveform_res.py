from enterprise.signals.signal_base import function as enterprise_function, PTA
from constants import *
import numpy as np
from numpy import sin, cos, cosh, sqrt, pi
from scipy.integrate import odeint
from getx import get_x
from gw_functions import rx, phitx, phiv, rtx
import matplotlib.pyplot as plt
from eval_max import get_max, Fomg
import antenna_pattern as ap
from rr_method_enl import solve_rr

from hypmik3pn import get_u, get_u_v2
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from astropy.cosmology import Planck18


def amp(M,q,e,b,z,order):
    eta=q/(1+q)**2
    x0=get_x(e,eta,b,order)[0]
    dis=M*dsun
    Time=M*tsun
    D_GW = 1e6 * Planck18.luminosity_distance(z).value * pc # meter
    scale=D_GW/dis
    return x0*eta/scale*Time

def get_hyp_waveform(q,e,b,ti_dim,tf_dim,t_step,inc,order):
    eta=q/(1+q)**2
    
    x0=get_x(e,eta,b,3)[0]
    n0=x0**(3/2)
    t_arr=np.linspace(ti_dim,tf_dim,t_step)
    
    l_i=n0*ti_dim


    y0=[e,n0,l_i]
    sol2=solve_rr(eta,b,y0,ti_dim,tf_dim,t_arr)
    earr,narr,larr=sol2
    
    xarr=narr**(2/3) 
    uarr=get_u_v2(larr,earr,eta,xarr,3)



    step=len(t_arr)
    hp_arr=np.zeros(step)
    hx_arr=np.zeros(step)
    X=np.zeros(step)
    Y=np.zeros(step)
    for i in range(step):
        et=earr[i]
        u=uarr[i]
        x=narr[i]**(2/3) 
        phi=phiv(eta,et,u,x,order)
        r1=rx(eta,et,u,x,order)
        z=1/r1
        phit=phitx(eta,et,u,x,order)
        rt=rtx(eta,et,u,x,order)
        phi=phiv(eta,et,u,x,order)
        phi=phiv(eta,et,u,x,order)
        r1=rx(eta,et,u,x,order)
        X[i]=r1*cos(phi)
        Y[i]=r1*sin(phi)
        hp_arr[i]=(-(sin(inc)**2*(z-r1**2*phit**2-rt**2)+(1+cos(inc)**2)*((z
        +r1**2*phit**2-rt**2)*cos(2*phi)+2*r1*rt*phit*sin(2*phi))))
        hx_arr[i]=(-2*cos(inc)*((z+r1**2*phit**2-rt**2)*sin(2*phi)-2*r1*rt*phit*cos(2*phi)))
    Hp=hp_arr/x0
    Hx=hx_arr/x0
    return Hp-Hp[0],Hx-Hx[0]

@enterprise_function
def hyp_pta_res(toas,
    theta,
    phi,
    cos_gwtheta,
    gwphi,
    psi,
    cos_inc,
    log10_M,
    q,
    b,
    e0,
    log10_z,
    tref,
    interp_steps=1000
):

    order = 3

    M = 10**log10_M # Solar mass
    z = 10**log10_z 

    Time=M*tsun

    ts = (toas - tref)/Time

    # ti, tf, tzs in seconds, in source frame
    ti = min(ts)
    tf = max(ts)
    tz_arr = np.linspace(ti, tf, interp_steps)
    delta_t_arr = (tz_arr[1]-tz_arr[0]) 

    tzs = ts
    

    inc = np.arccos(cos_inc)

    gwra = gwphi
    gwdec = np.arcsin(cos_gwtheta)

    psrra = phi
    psrdec = np.pi/2 - theta

    hp_arr, hx_arr = get_hyp_waveform(q,e0,b,ti,tf,interp_steps,inc,order)


    cosmu, Fp, Fx = ap.antenna_pattern(gwra, gwdec, psrra, psrdec)

    c2psi = np.cos(2*psi)
    s2psi = np.sin(2*psi)
    Rpsi = np.array([[c2psi, -s2psi],
                     [s2psi, c2psi]])
    h_arr = np.dot([Fp,Fx], np.dot(Rpsi, [hp_arr,hx_arr]))

    # Integrate over time in SSB frame
    s_arr = cumtrapz(h_arr, initial=0)*delta_t_arr

    s_spline = CubicSpline(tz_arr, s_arr)
    s_pre = s_spline(ts)

    

    return s_pre , amp(M,q,e0,b,z,order)
    




