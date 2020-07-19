import numpy as np
import matplotlib.pyplot  as plt
from scipy.integrate import quad
import cevns
import cevns_accelerators as acc
import coherent_CsI_real as csir
import cevns_statistics as stat
from scipy.interpolate import interp1d
from scipy.optimize import minimize
#setting up COHERENT                                                                                                                                                
pot= 0.08*(1.76e23/308.1)/(24.0*60.0*60.0)
R=1930.0 #in cm                                                                                                                                                

 
A_Cs=133
Z_Cs=55
bindE_Cs=1118.532104
Q=0.0878
Y=13.348
A_I=127
Z_I=53
bindE_I=1072.580322
mass=14.6
time=308.1
f_Cs=(A_Cs)/(A_Cs+A_I)
f_I=(A_I)/(A_Cs+A_I)

#best fit
epseu=    2.02366521e-01
epsmu=  -2.95015935e-02
  


sig=csir.signalcomp_lmad(epseu,epsmu)
#chi2=147.7320555091213
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
#print( minimize(lambda x: csir.chi2_1d( n,bkg1dsdata,dat,x[0],x[1],x[2],x[3]),(0.0,0.0,0.0,0.0),method='SLSQP',tol=1e-5,options={"maxiter":1e3}))
#print(csir.chi2_bins(n,bkg1dsdata,dat,epsee,  epsmm,epsem,epset,epsmt,0,0,0))

def randompred(sig,n,bkg):
#        alpha=stat.pulldist(0.28)
 #       beta=stat.pulldist(0.25)
        alpha=np.random.normal(0, 0.28)
        beta=np.random.normal(0, 0.25)
        sig_n=csir.add_lists_2(sig,n,alpha,beta)
 #       smeared=acc.smearing_fct(sig_n)
        pred=acc.data_eff(sig_n)
        #add bkg                          
        lista=csir.add_lists_1(pred,bkg,0)
        return lista


def randomdata(lista):
        arrout = np.zeros((2, 3))
        bins=csir.rebin_list_1E2t(lista)
        for i in range(len(bins)):
                if bins[i,2]>=0:
                        arrout[i,0] =bins[i,0]
                        arrout[i,1]=bins[i,1]
                        arrout[i,2]=np.random.poisson(bins[i, 2])
                        prob=0
                else:
                        prob=1e4
        return prob,arrout


def chi2_bins(n,ac,meas,epseu,epsmu,alpha,beta,gamma):
        chi=0
        sig=csir.signalcomp_lmad(epseu,epsmu)
        sig_n=csir.add_lists_2(sig,n,alpha,beta)
#        smeared=acc.smearing_fct(sig_n)
        pred=acc.data_eff(sig_n)
        #add bkg                                                                                                                                                                                        
        lista=csir.add_lists_1(pred,ac,gamma)
        preddev=csir.rebin_list_1E2t(lista)
        
        for i in range(len(preddev)):
                numevents = preddev[i, 2]
                numobs = meas[i, 2]
                if numobs == 0:
                        add = numevents - numobs
                else:
                        add = numevents - numobs + numobs*np.log(numobs/numevents)

                chi += add
        return 2*chi+(alpha/0.28)**2+(beta/0.25)**2+(gamma/0.06)**2
  #      return 2*chi+ 2*(alpha - np.log(alpha + 1.))/0.28**2+2*(beta - np.log(beta + 1.))/0.25**2+2*(gamma - np.log(gamma + 1.))/0.06**2

def mini_bins(epseu,epsmu,n,meas,bkgdata):
    minres= minimize(lambda x: chi2_bins( n,bkgdata,meas,epseu,epsmu,x[0],x[1],x[2]),(0.0,0.0,0.0),method='SLSQP',tol=1e-5,options={"maxiter":1e3})
    return minres


#number of tests
nn=2000
fileout=np.zeros((nn,1))
for i in range(nn):
        random=randompred(sig,n,bkg1dsdata)
        pp,outt=randomdata(random)
        if pp<10:
                res=mini_bins(epseu,epsmu,n,outt,bkg1dsdata)
                fileout[i,0]=res.fun
#                print("min",res.fun)
                np.savetxt("datafiles_out/result_pval_nsi_lmad_2t1e_nomarg_nosmear_gauss006_5.txt",fileout)
#print(fileout)
