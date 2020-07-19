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
bkg1dsdata=acc.bkgreal_2ds(ACoff,ACon,COoff)

#minimizes chi2 with 2 timing, 1 energy bin over all NSI parameters and 3 pull terms
def mini_2t1e(epsee,n,meas,bkgdata):
    minres= minimize(lambda x: csir.chi2_bins( n,bkgdata,meas,epsee,x[0],x[1],x[2],x[3],x[4],x[5],x[6]),(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),method='SLSQP',tol=1e-5,options={"maxiter":1e3})
    return minres

results=np.zeros((len(np.arange(-1,1.02,0.02)),9))                       
kk=0  
#loop over eps_ee and write to fule
for epsee in np.arange(-1,1.02,0.02):
      res=mini_2t1e(epsee,n,dat,bkg1dsdata)

      results[kk,0]=epsee
      results[kk,1]=res.fun
      results[kk,2]=res.x[0]
      results[kk,3]=res.x[1]
      results[kk,4]=res.x[2]
      results[kk,5]=res.x[3]
      results[kk,6]=res.x[4]
      results[kk,7]=res.x[5]
      results[kk,8]=res.x[6]
      kk=kk+1
#      print(" I'm at ee ",epsee)
#      print("with chi^2 ", res.fun)
      print("ch",res)
  #    np.savetxt("datafiles_out/data_gauss/chi2_eeonly_2t1e_marg_nosmear_gauss006.txt",results)
