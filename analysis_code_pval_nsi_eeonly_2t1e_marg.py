#MC simulation assuming all NSI parameters to be non-zero simultaneously
import numpy as np
import matplotlib.pyplot  as plt
from scipy.integrate import quad
import cevns
import cevns_accelerators as acc
import coherent_CsI_real as csir
import cevns_statistics as stat #needed for the new pull term parametrization
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

#best fit values of the NSI parameters
#calculation further down
epsee=   9.56045515e-02
epsmm=  -1.38976619e-02
epsem=   8.49049946e-05
epset=2.37678040e-05
epsmt=  -1.85888604e-03

#calculate the best fit signal
sig=csir.signalcomp(epsee,epsmm,epsem,epset,epsmt)
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

#calculation of best fit NSI parameters
#print( minimize(lambda x: csir.chi2_bins( n,bkg1dsdata,dat,x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]),(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0),method='SLSQP',tol=1e-5,options={"maxiter":1e3}))

#take pull term from normal distribution (or self defined distribution for the new pull term parametrization)
def randompred(sig,n,bkg):
    #    alpha=stat.pulldist(0.28)#if new pull terms
     #   beta=stat.pulldist(0.25)#
        alpha=np.random.normal(0, 0.28)#Gaussian pull terms
        beta=np.random.normal(0, 0.25)
        sig_n=csir.add_lists_2(sig,n,alpha,beta)
        pred=acc.data_eff(sig_n)
        #add bkg                          
        lista=csir.add_lists_1(pred,bkg,0)
        return lista

#apply Poisson fluctuations to the total number of observed events in each bin
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

#calculate chi2
def chi2_bins(n,ac,meas,epsee,epsmm,epsem,epset,epsmt,alpha,beta,gamma):
        chi=0
        sig=csir.signalcomp(epsee,epsmm,epsem,epset,epsmt)
        sig_n=csir.add_lists_2(sig,n,alpha,beta)
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
    #uncomment the next line if new pull term parametrization is used
#        return 2*chi+ 2*(alpha - np.log(alpha + 1.))/0.28**2+2*(beta - np.log(beta + 1.))/0.25**2+2*(gamma - np.log(gamma + 1.))/0.06**2

def mini_bins(epsee,n,meas,bkgdata):
    minres= minimize(lambda x: chi2_bins( n,bkgdata,meas,epsee,x[0],x[1],x[2],x[3],x[4],x[5],x[6]),(0.0,0.0,0.0,0.0,0.0,0.0,0.0),method='SLSQP',tol=1e-5,options={"maxiter":1e3})
    return minres

#number of tests
nn=2000
fileout=np.zeros((nn,1))
for i in range(nn):
        random=randompred(sig,n,bkg1dsdata)
        pp,outt=randomdata(random)
        if pp<10:
                res=mini_bins(epsee,n,outt,bkg1dsdata)
                fileout[i,0]=res.fun
                np.savetxt("datafiles_out/data_gauss/result_pval_nsi_eeonly_2t1e_marg_nosmear_gauss006_1.txt",fileout)

