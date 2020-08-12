import numpy as np
import matplotlib.pyplot  as plt
from scipy.integrate import quad
import cevns
import cevns_accelerators as acc
import coherent_CsI_real as csir
from scipy.interpolate import interp1d
from scipy.optimize import minimize

#load data                                                                                             
dat=acc.load_file("datafiles_in/Co_data.dat")
#load neutrons                                                                                                                                                     
neut=acc.load_file("datafiles_in/total_neutron.dat")
n=acc.load_neutron(neut)
#load bkg                                                                                                                                                         
ACon=np.loadtxt("datafiles_in/full_AC_on.dat")
ACoff=np.loadtxt("datafiles_in/full_AC_off.dat")
COoff=np.loadtxt("datafiles_in/full_CO_off.dat")
#produce 2D background file
bkg1dsdata=acc.bkgreal_2ds(ACoff,ACon,COoff)


#calculate TS for 2 timing and 1 energy bin
def chi2_2t1e(n,ac,meas,epseu,epsmu,alpha,beta,gamma):
    chi=0
    sig=csir.signalcomp_lmad(epseu,epsmu)
    sig_n=csir.add_lists_2(sig,n,alpha,beta)
    #uncomment the next two lines if smearing is required
#    pred=acc.data_eff(sig_n)                                                                                                                                                                             
#    smeared=acc.smearing_fct(sig_n)                                                                                                                                                                      
    pred=acc.data_eff(sig_n)

    #add bkg                                                                                                                                                                                              
    lista=csir.add_lists_1(pred,ac,gamma)
    obss=csir.switch_te(meas)

    #rebin from 144 bins to 2 bins                                                                                                                                                                                               
    preddev=csir.rebin_list_1E2t(lista)
    obssev=csir.rebin_list_1E2t(obss)

    for i in range(len(preddev)):
        if preddev[i,2]>=0:
            numevents = preddev[i, 2]
        else:
            numevents=10000 #in case the minimizer leads to a negative number of events     
        numobs = obssev[i, 2]
        if numobs == 0:
            add = numevents - numobs
        else:
            add = numevents - numobs + numobs*np.log(numobs/numevents)

        chi += add
    return  2*chi+(alpha/0.28)**2+(beta/0.25)**2+(gamma/0.06)**2

def mini_2t1e(epseu,epsmu,n,meas,bkgdata):
    minres= minimize(lambda x: chi2_2t1e( n,bkgdata,meas,epseu,epsmu,x[0],x[1],x[2]),(0.0,0.0,0.0),method='SLSQP',tol=1e-5,options={"maxiter":1e3})
    return minres

results=np.zeros((len(np.arange(-1,1.05,0.05))*len(np.arange(-1,1.25,0.05)),6))                       
kk=0  
for epseu in np.arange(-1,1.05,0.05):
   for epsmu in np.arange(1,1.25,0.05):
      res=mini_2t1e(epseu,epsmu,n,dat,bkg1dsdata)

      results[kk,0]=epseu
      results[kk,1]=epsmu
      results[kk,2]=res.fun
      results[kk,3]=res.x[0]
      results[kk,4]=res.x[1]
      results[kk,5]=res.x[2]
      
      kk=kk+1

      np.savetxt("datafiles_out/chi2_lmad_2t1e_nomarg_nosmear_gauss006.txt",results)
