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

#print(csir.chi2_1d(n,bkg1dsdata,dat,0,0,0,0))

def mini_2t1e(epsee,n,meas,bkgdata):
    minres= minimize(lambda x: csir.chi2_bins( n,bkgdata,meas,epsee,0.0,0.0,0.0,0.0,x[0],x[1],x[2]),(0.0,0.0,0.0),method='SLSQP',tol=1e-5,options={"maxiter":1e3})
    return minres

results=np.zeros((len(np.arange(-1,1.02,0.02)),5))                       
kk=0  

for epsee in np.arange(-1,1.02,0.02):
      res=mini_2t1e(epsee,n,dat,bkg1dsdata)

      results[kk,0]=epsee
      results[kk,1]=res.fun
      results[kk,2]=res.x[0]
      results[kk,3]=res.x[1]
      results[kk,4]=res.x[2]
      
      kk=kk+1
#      print(" I'm at ee ",epsee)
#      print("with chi^2 ", res.fun)
      print("ch",res)
  #    np.savetxt("datafiles_out/data_gauss/chi2_eeonly_2t1e_nomarg_nosmear_gauss006.txt",results)
