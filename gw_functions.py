import numpy as np
from numpy import pi, sinh, cosh, tanh, arctanh, tan, cos, sqrt, arctan, sin
from constants import *




#Define in which Post Newtonian Order we are interested in, for eg. order=n refers to nPN order.
def PNx(pn0,pn1,pn2,pn3,initial,x,order):
    s=0
    pn=[pn0,pn1,pn2,pn3]
    for i in range(order+1):
        s+=pn[i]*x**(initial+i)
    return s

def rx(eta,et,u,x,order):
    
    a0=(-1+et*cosh(u))

    a1= 1/6*(-18+2*eta+et*(-6+7*eta)*cosh(u))
    a2=((-216+534*eta+8*eta**2-2*et**2*(4*eta**2+15*eta+36)+et*(et**2-1)*(35*eta**2-
    231*eta+72)*cosh(u))/(72*et**2-72))

    a3=(1/181440/(et**2-1)**2*(-4233600+12143736*eta-348705*pi**2*eta-761040*eta
    **2+4480*eta**3+280*et**4*(16*eta**3+90*eta**2-81*eta+432)-et**2*(3144960+81*(1435*pi
    **2-134336)*eta+3437280*eta**2+8960*eta**3)+140*et*(et**2-1)**2*(49*eta**3-3933*eta**2
    +7047*eta-864)*cosh(u)))


    return PNx(a0,a1,a2,a3,-1,x,order)


def rtx(eta,et,u,x,order):
    a0=et*sinh(u)/(-1+et*cosh(u))
    a1=sinh(u)*et*(-6+7*eta)/(-6+6*et*cosh(u))
    a2=(35/72*et*(et**3*(eta**2-33/5*eta+72/35)*cosh(u)**3-3*et**2*(eta**2-33/5*eta+72/35)*
    cosh(u)**2+96/35*et*(eta**2-57/16*eta-27/8)*cosh(u)+(9/35*eta**2-27/7*eta)*et**2-
    eta**2+3/7*eta+468/35)*sinh(u)/(-1+et*cosh(u))**4)
    a3=(41/64*et*(196/3321*(et-1)*(et+1)*(eta**3-3933/49*eta**2+7047/49*eta-864/49)*et**5*
    cosh(u)**5-980/3321*(et-1)*(et+1)*(eta**3-3933/49*eta**2+7047/49*eta-864/49)*et**4*
    cosh(u)**4-2*((-1196/3321*eta**3+10720/369*eta**2-2436/41*eta+1360/123)*et**2+560/
    123+1196/3321*eta**3-7408/369*eta**2+(pi**2+42796/4305)*eta)*et**3*cosh(u)**3+et**2*(
    (80/123*eta**3-232/41*eta**2+192/41*eta)*et**4+(-2080/123-8224/3321*eta**3+24680/
    369*eta**2+(pi**2-95696/1435)*eta)*et**2+13600/123+6064/3321*eta**3-2720/369*eta**2+
    (-336992/1435+5*pi**2)*eta)*cosh(u)**2-2*((-58/123*eta**3-338/41*eta**2+1462/41*eta
    )*et**4+(-5200/123+1346/3321*eta**3+7538/369*eta**2+(pi**2-20166/1435)*eta)*et**2+
    10960/123+220/3321*eta**3+5440/369*eta**2+(-243988/1435+2*pi**2)*eta)*et*cosh(u)+(
    -92/41*eta-52/41*eta**3+292/41*eta**2)*et**6+(3008/41*eta+272/123*eta**3-1320/41*
    eta**2)*et**4+(-6112/123-3328/3321*eta**3+12728/369*eta**2+(pi**2-1856/35)*eta)*et**2
    +9952/123+196/3321*eta**3+3148/369*eta**2+(-14396/123+pi**2)*eta)*sinh(u)/(et**2-1)
    /(-1+et*cosh(u))**6)
    return PNx(a0,a1,a2,a3,1/2,x,order)



def phitx(eta,et,u,x,order):
    a0=(et**2-1)**(1/2)/(-1+et*cosh(u))**2
    a1=((et*(-1+eta)*cosh(u)-3+(-eta+4)*et**2)/(et**2-1)**(1/2)/(-1+et*cosh(u))**3)
    a2=(1/12*(-14*et**3*((eta**2+5*eta-3/7)*et**2-4/7*eta**2+1/7*eta-18/7)*cosh(u)**3+17*et**
    2*((48/17+eta**2-eta)*et**4+(-66/17-4/17*eta**2+8*eta)*et**2-108/17+5/17*eta**2+97/
    17*eta)*cosh(u)**2-et*(et**4*(eta**2-97*eta-12)+(16*eta**2+188*eta+102)*et**2+eta**2+
    125*eta-216)*cosh(u)+(-12*eta**2-18*eta)*et**6+(20*eta**2-26*eta-60)*et**4+(-2*eta**
    2+68*eta+162)*et**2+48*eta-144)/(et**2-1)**(3/2)/(-1+et*cosh(u))**5)
    a3=(1/6720*(12915*((64/123*eta**3+2912/369*eta**2-104/9*eta+64/41)*et**4+(736/41-160/
    369*eta**3+2432/369*eta**2+(pi**2-216592/4305)*eta)*et**2+128/41+64/369*eta**3+128/
    369*eta**2+(1/3*pi**2-1112/123)*eta)*et**5*cosh(u)**5+8610*et**4*((-116/123*eta**3+92
    /123*eta**2+44/41*eta)*et**6+(736/41+116/123*eta**3-2564/123*eta**2+(pi**2-4448/105)
    *eta)*et**4+(-5200/41-132/41*eta**3-12332/123*eta**2+(370220/861-11/2*pi**2)*eta)*
    et**2-2496/41+52/41*eta**3+1124/123*eta**2+(616088/4305-11/2*pi**2)*eta)*cosh(u)**4-
    34440*et**3*((-1/123*eta**3+245/123*eta**2-694/123*eta-144/41)*et**6+(1096/41-55/
    123*eta**3+53/123*eta**2+(pi**2-204698/4305)*eta)*et**4+(-2248/41-119/123*eta**3-
    7705/123*eta**2+(-7/4*pi**2+159434/861)*eta)*et**2-2184/41+55/123*eta**3+189/41*eta
    **2+(-17/4*pi**2+576538/4305)*eta)*cosh(u)**3-8610*((-32/123*eta**3+48/41*eta**2-40/
    41*eta)*et**8+(1728/41-28/123*eta**3-908/41*eta**2+(pi**2-172708/4305)*eta)*et**6+(-
    7776/41+180/41*eta**3-1316/123*eta**2+(-9*pi**2+2116892/4305)*eta)*et**4+(6816/41-
    12/41*eta**3+30772/123*eta**2+(6*pi**2-1072548/1435)*eta)*et**2+13152/41+12/41*eta**
    3+484/123*eta**2+(22*pi**2-660244/861)*eta)*et**2*cosh(u)**2+17220*et*(-8/123*eta*(
    eta**2+75*eta+379/2)*et**8+(864/41+34/41*eta**3+1538/123*eta**2+(pi**2-132248/4305)*
    eta)*et**6+(-2912/41-118/123*eta**3-3974/123*eta**2+(-5*pi**2+229366/861)*eta)*et**4
    +(968/41+146/123*eta**3+9142/123*eta**2+(11/4*pi**2-376388/1435)*eta)*et**2+4560/41
    -2/123*eta**3+734/123*eta**2+(25/4*pi**2-977078/4305)*eta)*cosh(u)+(5040*eta**3+
    20160*eta**2-7560*eta)*et**10+(-26320*eta**3-6720*eta**2+241640*eta)*et**8+(-120960+
    42000*eta**3-142240*eta**2+(-8610*pi**2-95584)*eta)*et**6+(336000-26320*eta**3+
    355040*eta**2+(34440*pi**2-1401544)*eta)*et**4+(-3360+2240*eta**3-404320*eta**2+(-
    21525*pi**2+1368504)*eta)*et**2-21525*pi**2*eta-13440*eta**2+810320*eta-504000)/(et
    **2-1)**(5/2)/(-1+et*cosh(u))**7)
    return PNx(a0,a1,a2,a3,3/2,x,order)

def vH(eta,et,u,x,order):
    ephi=ephiet(eta,et,x,order)*et
    return (2*arctan(((ephi+1)/(ephi-1))**(1/2)*tanh(1/2*u)))

def ephiet(eta,et,x,order):
    a0= 1
    a1= (-4+eta)
    a2=((4*eta**2+260*eta-2016+et**2*(41*eta**2-659*eta+1152))/(96*et**2-96))
    a3=(1/26880/(et**2-1)**2*(-4139520-20*(861*pi**2-178748)*eta+155680*eta**2+3*et**2*(
    806400+(1435*pi**2-430016)*eta+161140*eta**2-6300*eta**3)+70*et**4*(15*eta**3-1915*
    eta**2+11233*eta-12288)))
    return PNx(a0,a1,a2,a3,0,x,order)

def phiv(eta,et,u,x,order):
    v=vH(eta,et,u,x,order)
    a0=v
    a1=3*v/(et**2-1)
    a2=(-1/32*(8*v*(-78+28*eta+et**2*(-51+26*eta))+4*et**2*(3*eta**2-19*eta-1)*sin(2*v
    )+et**3*eta*(-1+3*eta)*sin(3*v))/(et**2-1)**2)

    a3=(1/26880/(et**2-1)**3*(et**2*(84000+1180064*eta-30135*pi**2*eta-442400*eta**2+
    10080*eta**3+280*et**2*(93*eta**3-781*eta**2+886*eta+24))*sin(2*v)+et**3*eta*(113208
    -4305*pi**2-101780*eta+7140*eta**2+35*et**2*(129*eta**2-137*eta+33))*sin(3*v)+210*v
    *(16*et**4*(65*eta**2-110*eta+156)+18240+4*(123*pi**2-6344)*eta+896*eta**2+et**2*(
    28128+3*(41*pi**2-9280)*eta+5120*eta**2))+140*et**4*eta*(15*eta**2-57*eta+82)*sin(4
    *v)+105*et**5*eta*(5*eta**2-5*eta+1)*sin(5*v)))
    return PNx(a0,a1,a2,a3,0,x,order)







def get_gw_ra_dec(a1,b1,e1,d1):
    psrra=     15*(a1+b1/60)*pi/180
    psrdec =   (e1+e1/np.abs(e1)*d1/60)*pi/180
    return psrra, psrdec