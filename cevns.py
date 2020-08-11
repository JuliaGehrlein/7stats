#calculation of CEvNS process
import numpy as np
import cevns_accelerators 
import coherent_CsI_real as csir
from scipy.integrate import quad,dblquad
import scipy.stats

#form factor (Klein Nystrand), take eq. 2.5 from http://inspirehep.net/record/1664530/files/Scholz_uchicago_0330D_14161.pdf (arxiv: 1904.01155)    
def formfac(Er,A,Z,bindingE):
    a=0.7 #in fm
    Ra=A**(1/3)*1.2 #in fm
    rho=3*A/(4*np.pi*Ra**3)
    #nucleus mass
    ma=Z*938.27208816+(A-Z)*939.56542052-bindingE#subtract binding energy, all in MeV
    #energy transfer in keV and fm
    qkev=np.sqrt(2*ma*Er*1e3)#in keV
    q=1/1.97*1e-5*qkev#in fm
    #that's the form factor
    f=4*np.pi*rho/(A*q**3)*(np.sin(q*Ra)-q*Ra*np.cos(q*Ra))*1/(1+a**2*q**2)
    return f

#CEvNS cross section heavy  mediator
#dsigma/dEr (from 1701.04828)
def crosssec_nsi(Er, Enu,A,Z,bindE,epsuxx=0,epsdxx=0,epsuxy=0,epsdxy=0,epsuxz=0,epsdxz=0 ):
    GF=1.1663787*1e-5 #GeV^-2
    M=(Z*938.27208816+(A-Z)*939.56542052-bindE)*1e-3#in GeV
    #weak charge from 1806.01310
    rhop=1.0082
    sz=0.23129
    k=0.9972
    lul=-0.0031
    ldl=-0.0025
    ldr=7.5*1e-5
    lur=0.5*7.5*1e-5
    gpV=rhop*(0.5-2*k*sz)+2*lul+2*lur+ldl+ldr
    gnV=-0.5*rhop+lul+lur+2*ldl+2*ldr
    Q=(Z*(gpV+2*epsuxx+epsdxx)+(A-Z)*(gnV+epsuxx+2*epsdxx))**2
    Q+=(Z*(2*epsuxy+epsdxy)+(A-Z)*(epsuxy+2*epsdxy))**2
    Q+=(Z*(2*epsuxz+epsdxz)+(A-Z)*(epsuxz+2*epsdxz))**2 #actually Q^2/4
    enu=Enu*1e-3 #in GeV
    #take subleading kinetic terms into account 1910.04951
    kin=max(0,(2-2*Er*1e-6/enu+(Er*1e-6/enu)**2-(M*Er*1e-6)/(enu*enu)))
    dsigdEr=GF**2/(2*np.pi)*Q*formfac(Er,A,Z,bindE)**2*M*kin#in GeV^(-3)
    return dsigdEr*1e-6*(1.97*1e-14)**2
    #convert from GeV^(-3) to cm^2/kev

#minimal neutrino energy
def E_numin(A,Z,Er,bindE):
    ma=(Z*938.27208816+(A-Z)*939.56542052-bindE)*1e-3 #in GeV
    return np.maximum(np.sqrt(ma*Er/2.),0) 

#rate for nu_e
def ratee( A,Z,type,bindE,R,pot,Q,Y,PEmin,PEmax,mass,time,epsuee=0,epsdee=0,epsuem=0,epsdem=0,epsuet=0,epsdet=0 ):
    if(type=="accelerator"):
        ratee=lambda enu,PE: cevns_accelerators.fluxnue(enu,R,pot)*crosssec_nsi(PE/(Q*Y), enu,A,Z,bindE,epsuee,epsdee,epsuem,epsdem,epsuet,epsdet )
    else:
        ratee=0
    ma=(Z*938.27208816+(A-Z)*939.56542052-bindE)*1.79*1e-30 #in kg
    res=dblquad(ratee,PEmin,PEmax,lambda PE:E_numin(A,Z,PE/(Q*Y),bindE),lambda PE:53., epsrel=1e-4)[0]/ma
    
    return mass*time*res*86400.0*1/(Q*Y)


#rate for prompt nu_mu 
def ratemp( A,Z,type,bindE,R,pot,Q,Y,PEmin,PEmax,mass,time,epsumm=0,epsdmm=0,epsuem=0,epsdem=0,epsumt=0,epsdmt=0 ):
    if (type=="accelerator"):
        ratemp= lambda PE: cevns_accelerators.fluxnumu(29.79,R,pot)*crosssec_nsi(PE/(Q*Y), 29.79,A,Z,bindE,epsumm,epsdmm,epsuem,epsdem,epsumt,epsdmt )
    else:
        ratemp=0
    ma=(Z*938.27208816+(A-Z)*939.56542052-bindE)*1.79*1e-30 #in kg

    return mass*time*quad(ratemp,PEmin,PEmax, epsrel=1e-6)[0]/ma*86400.0*1/(Q*Y)



#rate for delayed nu_mu
def ratemd( A,Z,type,bindE,R,pot,Q,Y,PEmin,PEmax,mass,time,epsumm=0,epsdmm=0,epsuem=0,epsdem=0,epsumt=0,epsdmt=0 ):
    if(type=="accelerator"):
        ratemd=lambda enu,PE: cevns_accelerators.fluxnumubar(enu,R,pot)*crosssec_nsi(PE/(Q*Y), enu,A,Z,bindE,epsumm,epsdmm,epsuem,epsdem,epsumt,epsdmt )
    else:
        ratemd=0
    ma=(Z*938.27208816+(A-Z)*939.56542052-bindE)*1.79*1e-30 #in kg                                                                                  
    res=dblquad(ratemd,PEmin,PEmax,lambda PE:E_numin(A,Z,PE/(Q*Y),bindE),lambda PE:53, epsrel=1e-4)[0]/ma
    return mass*time*res*86400.0*1/(Q*Y)

