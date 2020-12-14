#code to calculate the PDF needed for the FC procedure
#as an example for only non-zero eps_ee^V
import numpy as np
import matplotlib.pyplot  as plt
from scipy.integrate import quad
import cevns
import cevns_accelerators as acc
import coherent_CsI_real as csir

from scipy.interpolate import interp1d
from scipy.optimize import minimize
#setting up COHERENT                                                                                                                                                
pot= 0.08*(1.76e23/308.1)/(24.0*60.0*60.0) #proton on target
R=1930.0 #distance to detector in cm
#the atomic numbers of Cs, I and their binding energies
A_Cs=133
Z_Cs=55
bindE_Cs=1118.532104
Q=0.0878
Y=13.348
A_I=127
Z_I=53
bindE_I=1072.580322
mass=14.6 #mass of CsI detector
time=308.1 #exposure time
#weighted contributions of Cs, I 
f_Cs=(A_Cs)/(A_Cs+A_I)
f_I=(A_I)/(A_Cs+A_I)

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

#function to generate a random prediction
def randompred(sig,alphabf,betabf,gammabf,n,bkg):
        sig_n=csir.add_lists_2(sig,n,alphabf,betabf) #adds signal and neutron background
        smeared=acc.smearing_fct(sig_n) #applies smearing 
        pred=acc.data_eff(smeared) #applies efficiency
        #add bkg                          
        lista=csir.add_lists_1(pred,bkg,gammabf)
        return lista

#function to apply Poisson fluctuations in each bin
def randomdata(lista):
        arrout = np.zeros((144, 3))
        bins=lista
        for i in range(len(bins)):
                if bins[i,2]>=0:
                        arrout[i,0] =bins[i,0]
                        arrout[i,1]=bins[i,1]
                        arrout[i,2]=np.random.poisson(bins[i, 2])
                        prob=0
                else:
                        prob=1e4
        return prob,arrout

#function to calculate TS between data and prediction
def chi2_bins(n,ac,meas,epsee,epsmm,epsem,epset,epsmt,alpha,beta,gamma):
        chi=0
        sig=csir.signalcomp(epsee,epsmm,epsem,epset,epsmt) #calculate signal for given NSI parameters
        sig_n=csir.add_lists_2(sig,n,alpha,beta) #add signal and neutron background
        smeared=acc.smearing_fct(sig_n) #apply smearing and efficiency
        pred=acc.data_eff(smeared)
        #add bkg                                                                                                                                                                    
        lista=csir.add_lists_1(pred,ac,gamma)
        #that's the prediction 
        preddev=lista
        for i in range(len(preddev)):
                if preddev[i,2]>=0:
                        numevents = preddev[i, 2]
                else:
                        numevents=10000 #in case the minimizer goes to a negative number of events   
                numobs = meas[i, 2] #that's the data
                if numobs == 0: #if data is zero
                        add = numevents - numobs
                else:
                        add = numevents - numobs + numobs*np.log(numobs/numevents)

                chi += add
       # return 2*chi+(alpha/0.28)**2+(beta/0.25)**2+(gamma/0.171)**2 #uncomment if Gaussian pull terms are used
        if (alpha + 1.)<=0 or (beta + 1.)<=0 or (gamma + 1.)<=0: #if asymmetric pull terms are used prevent that the log(0) case
                return np.abs(alpha*1000.0)+np.abs(beta*1000.0)+np.abs(gamma*1000.0)+2*chi
        else:
                return 2*chi+ 2*(alpha - np.log(alpha + 1.))/0.28**2+2*(beta - np.log(beta + 1.))/0.25**2+2*(gamma - np.log(gamma + 1.))/0.171**2

#the following function assume only non-zero eps_ee^V, if other NSI parameters are considered make this trivial change
#minimizer function for TS where one minizes over NSI parameter
def mini_bins(n,meas,bkgdata):
    minres= minimize(lambda x: chi2_bins( n,bkgdata,meas,x[0],0,0.0,0.0,0.0,x[1],x[2],x[3]),(0.0,0.0,0.0,0.0),method='SLSQP',tol=1e-5,options={"maxiter":1e3})
    return minres
#minimizer function with NSI parameter fixed in TS
def mini_binsfixeps(epsbf,n,meas,bkgdata):
    minres= minimize(lambda x: chi2_bins( n,bkgdata,meas,epsbf,0,0.0,0.0,0.0,x[0],x[1],x[2]),(0.0,0.0,0.0),method='SLSQP',tol=1e-5,options={"maxiter":1e3})
    return minres
#generate signal assume all NSI parameters apart from eps_ee^V are zero                                              
epsmm=0.0
epsem=0.0
epset=0.0
epsmt=0.0
epseetry=0.25 #test value of eps_ee^V
#determine best fit values of nuisance parameters for given value of eps_ee^V using the measured data
resbf= minimize(lambda x: csir.chi2_144bins_newpull(n,bkg1dsdata,dat,epseetry,0,0.0,0.0,0.0,x[0],x[1],x[2]),(0.0,0.0,0.0),method='SLSQP',tol=1e-5,options={"maxiter":1e3})  
alphabf=resbf.x[0]
betabf=resbf.x[1]
gammabf=resbf.x[2]
#generate signal prediction
sigeps=csir.signalcomp(epseetry,epsmm,epsem,epset,epsmt)
                                                                                                                                                       
nn=250 #number of tests
fileout=np.zeros((nn,4))
for i in range(nn):
        random=randompred(sigeps,alphabf,betabf,gammabf,n,bkg1dsdata) #calculate random prediction based on test value of eps_ee^V and best fit values for nuisance parameters
        pp,outt=randomdata(random)
        if pp<10:
                res=mini_bins(n,outt,bkg1dsdata) #TS where one marginalizes over eps_ee^V
                resepsfix=mini_binsfixeps(epseetry,n,outt,bkg1dsdata) #TS where eps_ee^V is fixed
                fileout[i,0]=epseetry
                fileout[i,1]=res.x[0]
                fileout[i,2]=res.fun
                fileout[i,3]=resepsfix.fun
                np.savetxt("datafiles_out/result_histo_nsi_eeonly_" + str(epseetry) + "_12t12e_nomarg_smear_newpull017.txt",fileout)  

