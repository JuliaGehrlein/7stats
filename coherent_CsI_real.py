#COHERENT CsI related physics
import numpy as np
import cevns_accelerators as acc
from sympy.solvers import solve
from sympy import Symbol
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import cevns_accelerators as acc
import cevns
from scipy.optimize import minimize

#efficiency from 1804.09459
def eff(PE):
    a=0.6655
    k=0.4942
    x0=10.8507-0.3995
    f=a/(1+np.exp(-k*(PE-x0)))
    if (PE<5):
        ret=0
    else:
        if(PE>= 5 and PE<6):
            ret=0.5*f
        else:
            ret=f
    return ret

#adds signal and neutron background taking the normalization pull terms into account
def add_lists_2(sig,n,alpha,beta):
    predbins=np.zeros((len(sig),3))
    for i in range(len(sig)):
        predbins[i,0]=sig[i,0]
        predbins[i,1]=sig[i,1]
        predbins[i,2]=(1+alpha)*sig[i,2]+(1+beta)*n[i,2]
    return predbins

#adds signal+neutron background to the background taking the normalization pull term for the background into account
def add_lists_1(sig,bkg,gamma):
    predbins=np.zeros((len(sig),3))
    normalisation=308.1/(153.5*2+308.1) #to account for the fact that the background has more data
    for i in range(len(sig)):
        predbins[i,0]=sig[i,0]
        predbins[i,1]=sig[i,1]
        predbins[i,2]=sig[i,2]+(bkg[i,2]*normalisation)*(1+gamma)
    return predbins

#switches t and e positions in list
def switch_te(lista):
    listfinal=np.zeros((len(lista),3))
    for i in range(len(lista)):
        listfinal[i,0]=lista[i,1]
        listfinal[i,1]=lista[i,0]
        listfinal[i,2]=lista[i,2]
    return listfinal

#rebins with 1 E bin, 2 t bins (from 0-1 mu s, 1-6 mu s)
def rebin_list_1E2t(lista):
    t1=0
    t2=0
    listfinal=np.zeros((2,3))
    for i in range(len(lista)):
        if lista[i,0]<1:
            t1+=lista[i,2]
        else:
            t2+=lista[i,2]
    listfinal[0,0]=1
    listfinal[0,1]=29
    listfinal[0,2]=t1
    listfinal[1,0]=6
    listfinal[1,1]=29
    listfinal[1,2]=t2
    return listfinal

#rebins with 2 E bin (6-18 PE, 19-30 PE), 2 t bins (0-1 mu s, 1-6 mu s)                                                                       
def rebin_list_2E2t(lista):
    t1e1=0
    t2e1=0
    t1e2=0
    t2e2=0
    listfinal=np.zeros((4,3))
    for i in range(len(lista)):
        if lista[i,1]<18:
            if lista[i,0]<1:
                t1e1+=lista[i,2]
            else:
                t2e1+=lista[i,2]
        else:
            if lista[i,0]<1:
                t1e2+=lista[i,2]
            else:
                t2e2+=lista[i,2]

    listfinal[0,0]=1
    listfinal[0,1]=18
    listfinal[0,2]=t1e1
    listfinal[1,0]=6
    listfinal[1,1]=18
    listfinal[1,2]=t2e1
    listfinal[2,0]=1
    listfinal[2,1]=29
    listfinal[2,2]=t1e2
    listfinal[3,0]=6
    listfinal[3,1]=29
    listfinal[3,2]=t2e2
    return listfinal

#rebins with 4 E bin (6- 18, 19-22,23-26,27-30), 2 t bins (0-1 mu s, 1-6 mu s)                                        
def rebin_list_4E2t(lista):
    t1e1=0
    t2e1=0
    t1e2=0
    t2e2=0
    t1e3=0
    t2e3=0
    t1e4=0
    t2e4=0
    listfinal=np.zeros((8,3))
    for i in range(len(lista)):
        if lista[i,1]<18:
            if lista[i,0]<1:
                t1e1+=lista[i,2]
            else:
                t2e1+=lista[i,2]
        else:
            if lista[i,1]==19 or lista[i,1]==21:
                if lista[i,0]<1:
                    t1e2+=lista[i,2]
                else:
                    t2e2+=lista[i,2]
            else:
                if lista[i,1]==23 or lista[i,1]==25:
                    if lista[i,0]<1:
                        t1e3+=lista[i,2]
                    else:
                        t2e3+=lista[i,2]
                else:
                    if lista[i,0]<1:
                        t1e4+=lista[i,2]
                    else:
                        t2e4+=lista[i,2]
    listfinal[0,0]=1
    listfinal[0,1]=18
    listfinal[0,2]=t1e1
    listfinal[1,0]=6
    listfinal[1,1]=18
    listfinal[1,2]=t2e1
    listfinal[2,0]=1
    listfinal[2,1]=22
    listfinal[2,2]=t1e2
    listfinal[3,0]=6
    listfinal[3,1]=22
    listfinal[3,2]=t2e2
    listfinal[4,0]=1
    listfinal[4,1]=26
    listfinal[4,2]=t1e3
    listfinal[5,0]=6
    listfinal[5,1]=26
    listfinal[5,2]=t2e3
    listfinal[6,0]=1
    listfinal[6,1]=29
    listfinal[6,2]=t1e4
    listfinal[7,0]=6
    listfinal[7,1]=29
    listfinal[7,2]=t2e4
    return listfinal




#signal computation for LMA D
def signalcomp_lmad(epseu,epsmu):
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

    numbins=33
    PE_bins =np.linspace(4, 37,34 )
    bins_I_ed = np.zeros(numbins)
    bins_Cs_ed = np.zeros(numbins)
    bins_result_ed=np.zeros((numbins,2))

    bins_I_md = np.zeros(numbins)
    bins_Cs_md = np.zeros(numbins)
    bins_result_md=np.zeros((numbins,2))

    bins_I_mp = np.zeros(numbins)
    bins_Cs_mp = np.zeros(numbins)
    bins_result_mp=np.zeros((numbins,2))

    epsuee=epseu
    epsdee=0.0
    epsuem=0.0
    epsdem=0.0
    epsdet=0.0
    epsuet=0.0
    epsumm=epsmu
    epsdmm=0.0
    epsumt=0.0
    epsdmt=0.0
    k=0
    for i in range(len(PE_bins)-1):
        bins_Cs_ed[i] = f_Cs*cevns.ratee( A_Cs,Z_Cs,"accelerator",bindE_Cs,R,pot,Q,Y,PE_bins[i],PE_bins[i+1],mass,time,epsuee,epsdee,epsuem,epsdem,epsuet,epsdet )
        bins_I_ed[i] = f_I*cevns.ratee( A_I,Z_I,"accelerator",bindE_I,R,pot,Q,Y,PE_bins[i],PE_bins[i+1],mass,time,epsuee,epsdee,epsuem,epsdem,epsuet,epsdet )
        bins_result_ed[k,1] = bins_I_ed[i] + bins_Cs_ed[i]
        bins_result_ed[k,0]=PE_bins[i]
        bins_Cs_md[i] = f_Cs*cevns.ratemd( A_Cs,Z_Cs,"accelerator",bindE_Cs,R,pot,Q,Y,PE_bins[i],PE_bins[i+1],mass,time,epsumm,epsdmm,epsuem,epsdem,epsumt,epsdmt)
        bins_I_md[i] = f_I*cevns.ratemd( A_I,Z_I,"accelerator",bindE_I,R,pot,Q,Y,PE_bins[i],PE_bins[i+1],mass,time,epsumm,epsdmm,epsuem,epsdem,epsumt,epsdmt)
        bins_result_md[k,1] = bins_I_md[i] + bins_Cs_md[i]
        bins_result_md[k,0]=PE_bins[i]
        bins_Cs_mp[i] = f_Cs*cevns.ratemp( A_Cs,Z_Cs,"accelerator",bindE_Cs,R,pot,Q,Y,PE_bins[i],PE_bins[i+1],mass,time,epsumm,epsdmm,epsuem,epsdem,epsumt,epsdmt)
        bins_I_mp[i] = f_I*cevns.ratemp( A_I,Z_I,"accelerator",bindE_I,R,pot,Q,Y,PE_bins[i],PE_bins[i+1],mass,time,epsumm,epsdmm,epsuem,epsdem,epsumt,epsdmt)
        bins_result_mp[k,1] = bins_I_mp[i] + bins_Cs_mp[i]
        bins_result_mp[k,0]=PE_bins[i]
        k=k+1

    sig=acc.timing_energy_2D(bins_result_ed,bins_result_md,bins_result_mp)
    return sig

#signal computation for vector NSI with heavy mediator
def signalcomp(epsee,epsmm,epsem,epset,epsmt):
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
    
    numbins=33
    PE_bins =np.linspace(4, 37,34 )
    bins_I_ed = np.zeros(numbins)
    bins_Cs_ed = np.zeros(numbins)
    bins_result_ed=np.zeros((numbins,2))

    bins_I_md = np.zeros(numbins)
    bins_Cs_md = np.zeros(numbins)
    bins_result_md=np.zeros((numbins,2))

    bins_I_mp = np.zeros(numbins)
    bins_Cs_mp = np.zeros(numbins)
    bins_result_mp=np.zeros((numbins,2))

    epsuee=epsee
    epsdee=epsee
    epsuem=epsem
    epsdem=epsem
    epsdet=epset
    epsuet=epset
    epsumm=epsmm
    epsdmm=epsmm
    epsumt=epsmt
    epsdmt=epsmt
    k=0
    for i in range(len(PE_bins)-1):
        bins_Cs_ed[i] = f_Cs*cevns.ratee( A_Cs,Z_Cs,"accelerator",bindE_Cs,R,pot,Q,Y,PE_bins[i],PE_bins[i+1],mass,time,epsuee,epsdee,epsuem,epsdem,epsuet,epsdet )
        bins_I_ed[i] = f_I*cevns.ratee( A_I,Z_I,"accelerator",bindE_I,R,pot,Q,Y,PE_bins[i],PE_bins[i+1],mass,time,epsuee,epsdee,epsuem,epsdem,epsuet,epsdet )
        bins_result_ed[k,1] = bins_I_ed[i] + bins_Cs_ed[i]
        bins_result_ed[k,0]=PE_bins[i]
        bins_Cs_md[i] = f_Cs*cevns.ratemd( A_Cs,Z_Cs,"accelerator",bindE_Cs,R,pot,Q,Y,PE_bins[i],PE_bins[i+1],mass,time,epsumm,epsdmm,epsuem,epsdem,epsumt,epsdmt)
        bins_I_md[i] = f_I*cevns.ratemd( A_I,Z_I,"accelerator",bindE_I,R,pot,Q,Y,PE_bins[i],PE_bins[i+1],mass,time,epsumm,epsdmm,epsuem,epsdem,epsumt,epsdmt)
        bins_result_md[k,1] = bins_I_md[i] + bins_Cs_md[i]
        bins_result_md[k,0]=PE_bins[i]
        bins_Cs_mp[i] = f_Cs*cevns.ratemp( A_Cs,Z_Cs,"accelerator",bindE_Cs,R,pot,Q,Y,PE_bins[i],PE_bins[i+1],mass,time,epsumm,epsdmm,epsuem,epsdem,epsumt,epsdmt)
        bins_I_mp[i] = f_I*cevns.ratemp( A_I,Z_I,"accelerator",bindE_I,R,pot,Q,Y,PE_bins[i],PE_bins[i+1],mass,time,epsumm,epsdmm,epsuem,epsdem,epsumt,epsdmt)
        bins_result_mp[k,1] = bins_I_mp[i] + bins_Cs_mp[i]
        bins_result_mp[k,0]=PE_bins[i]
        k=k+1

    sig=acc.timing_energy_2D(bins_result_ed,bins_result_md,bins_result_mp)
    return sig


#chi2 computation using 144 bins
def chi2_144bins(n,ac,meas,epsee,epsmm,epsem,epset,epsmt,alpha,beta,gamma):
    chi=0
    #add signal and neutron bkg                                                                                                                                                         
    sig=signalcomp(epsee,epsmm,epsem,epset,epsmt)
    sig_n=add_lists_2(sig,n,alpha,beta)
    smeared=acc.smearing_fct(sig_n)
    pred=acc.data_eff(smeared)    
    #add bkg                                                                                                                                
    lista=add_lists_1(pred,ac,gamma)
    for i in range(len(pred)):
        numevents = lista[i, 2]
        numobs = meas[i, 2]
        if numobs == 0:
            add = numevents - numobs
        else:
            add = numevents - numobs + numobs*np.log(numobs/numevents)

        chi += add
    return 2*chi+(alpha/0.28)**2+(beta/0.25)**2+(gamma/0.171)**2
#    return 2*chi+ 2*(alpha - np.log(alpha + 1.))/0.28**2+2*(beta - np.log(beta + 1.))/0.25**2+2*(gamma - np.log(gamma + 1.))/0.171**2

#chi2 for 1 bin
def chi2_1d(n,ac,meas,epsee,epsmm,epsem,epset,epsmt,alpha,beta,gamma):
    chi=0
    numevents1d=0 
    numobs1d=0
    sig=signalcomp(epsee,epsmm,epsem,epset,epsmt)
    sig_n=add_lists_2(sig,n,alpha,beta)
    pred=acc.data_eff(sig_n)  
    #add bkg                                                                                                                                                                            
    lista=add_lists_1(pred,ac,gamma)
    for i in range(len(lista)):
        numevents1d +=lista[i, 2]
        numobs1d += meas[i, 2]
    chi=numevents1d - numobs1d + numobs1d*np.log(numobs1d/numevents1d)
    return  2*chi+(alpha/0.28)**2+(beta/0.25)**2+(gamma/0.049)**2
#    return 2*chi+ 2*(alpha - np.log(alpha + 1.))/0.28**2+2*(beta - np.log(beta + 1.))/0.25**2+2*(gamma - np.log(gamma + 1.))/0.049**2

#chi2 compuation for 2 timing, 1 energy bin
def chi2_bins(n,ac,meas,epsee,epsmm,epsem,epset,epsmt,alpha,beta,gamma):
    chi=0
    sig=signalcomp(epsee,epsmm,epsem,epset,epsmt)
    sig_n=add_lists_2(sig,n,alpha,beta)
    pred=acc.data_eff(sig_n)
    #add bkg                                                                                                                                                                            
    lista=add_lists_1(pred,ac,gamma)
    obss=switch_te(meas)
    
    #rebin
    preddev=rebin_list_1E2t(lista)
    obssev=rebin_list_1E2t(obss)

    for i in range(len(preddev)):
        numevents = preddev[i, 2]
        numobs = obssev[i, 2]
        if numobs == 0:
            add = numevents - numobs
        else:
            add = numevents - numobs + numobs*np.log(numobs/numevents)

        chi += add
    return  2*chi+(alpha/0.28)**2+(beta/0.25)**2+(gamma/0.06)**2     
#    return 2*chi+ 2*(alpha - np.log(alpha + 1.))/0.28**2+2*(beta - np.log(beta + 1.))/0.25**2+2*(gamma - np.log(gamma + 1.))/0.06**2

#chi2 for 2 timing, 2 energy bins
def chi2_2t2ebins(n,ac,meas,epsee,epsmm,epsem,epset,epsmt,alpha,beta,gamma):
    chi=0
    sig=signalcomp(epsee,epsmm,epsem,epset,epsmt)
    sig_n=add_lists_2(sig,n,alpha,beta)
    smeared=acc.smearing_fct(sig_n)
    pred=acc.data_eff(smeared)
    #add bkg                                                                                                                                                                            
    lista=add_lists_1(pred,ac,gamma)
    obss=switch_te(meas)
    #rebin                                                                                                                                                                             
    preddev=rebin_list_2E2t(lista)
    obssev=rebin_list_2E2t(obss)
    for i in range(len(preddev)):
        numevents = preddev[i, 2]
        numobs = obssev[i, 2]
        if numobs == 0:
            add = numevents - numobs
        else:
            add = numevents - numobs + numobs*np.log(numobs/numevents)

        chi += add
    return  2*chi+(alpha/0.28)**2+(beta/0.25)**2+(gamma/0.0698)**2                                                                                                  
#    return 2*chi+ 2*(alpha - np.log(alpha + 1.))/0.28**2+2*(beta - np.log(beta + 1.))/0.25**2+2*(gamma - np.log(gamma + 1.))/0.0698**2

#chi2 for 2 timing, 4 energy bins
def chi2_2t4ebins(n,ac,meas,epsee,epsmm,epsem,epset,epsmt,alpha,beta,gamma):
    chi=0
    sig=signalcomp(epsee,epsmm,epsem,epset,epsmt)
    sig_n=add_lists_2(sig,n,alpha,beta)
    smeared=acc.smearing_fct(sig_n)
    pred=acc.data_eff(smeared)
    #add bkg                                                                                                                                                                           
    lista=add_lists_1(pred,ac,gamma)
    obss=switch_te(meas)
    #rebin                                                                                                                                                                             
    preddev=rebin_list_4E2t(lista)
    obssev=rebin_list_4E2t(obss)                  
    for i in range(len(preddev)):
        numevents = preddev[i, 2]
        numobs = obssev[i, 2]
        if numobs == 0:
            add = numevents - numobs
        else:
            add = numevents - numobs + numobs*np.log(numobs/numevents)

        chi += add
    return  2*chi+(alpha/0.28)**2+(beta/0.25)**2+(gamma/0.0855)**2
#    return 2*chi+ 2*(alpha - np.log(alpha + 1.))/0.28**2+2*(beta - np.log(beta + 1.))/0.25**2+2*(gamma - np.log(gamma + 1.))/0.0855**2         
