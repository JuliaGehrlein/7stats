#accelerator specific neutrino flux
import numpy as np
from scipy.interpolate import interp1d
import coherent_CsI_real as co
from scipy.stats import poisson
from scipy.optimize import minimize
import scipy.integrate as integrate

#flux per 1/cm^2 s
#anti nu mu flux
def fluxnumubar(Enu,R,pot):
    mmu=105.6583745 #MeV, from pdg
    return 64.0/mmu*((Enu/mmu)**2*(3./4.-Enu/mmu))*1/(4.*np.pi*R**2)*pot
#nu e flux
def fluxnue(Enu,R,pot):
    mmu=105.6583745 #MeV, from pdg                                                                                                        
    return 192.0/mmu*((Enu/mmu)**2*(0.5-Enu/mmu))*1/(4.*np.pi*R**2)*pot

#flux from pion decay at Enu=29.79 MeV (prompt)
def fluxnumu(Enu,R,pot):
    if(Enu>29.78 and Enu<29.8):
        return 1*1/(4.*np.pi*R**2)*pot
    else:
        return 0

def load_file(data):
    a=np.loadtxt(data)
    return a

#function which produces a 2D timing-energy array for the signal
def timing_energy_2D(fluxed,fluxmd,fluxmp):
    flux_d = load_file("datafiles_in/data_delayed_cut.dat")
    flux_p = load_file("datafiles_in/data_prompt_cut.dat")
    bin_signal_tot=np.zeros((33*12,3))
    k=0
    for i in range(len(flux_d)):
        for j in range(len(fluxed)):
            totsig=(fluxed[j,1])*flux_d[i,1]+(fluxmd[j,1])*flux_d[i,1]+(fluxmp[j,1])*flux_p[i,1]
            bin_signal_tot[k,0]=flux_d[i,0]
            bin_signal_tot[k,1]=fluxmp[j,0]
            bin_signal_tot[k,2]=totsig
            k=k+1
    return bin_signal_tot

#function which adds signal and background including the normalization pull terms
def add_lists(sig,n,ac,alpha,beta,gamma):
    predbins=np.zeros((len(sig),3))
    for i in range(len(sig)):
        predbins[i,0]=sig[i,0]
        predbins[i,1]=sig[i,1]
        predbins[i,2]=(1+alpha)*sig[i,2]+(1+beta)*n[i,2]+(1+gamma)*ac[i,2]
    return predbins

#function which applies the effiency to the signal
def data_eff(file):
    k=0
    finalres=np.zeros((144,3))
    for en in range(6,30,2):
        for t in np.arange (0.25,6.25,0.5):
            for i in range (len(file)-1):
                if file[i,0]==t and file[i,1]==en:
                    res=(file[i,2]*co.eff(file[i,1]))+(file[i+1,2]*co.eff(file[i+1,1]))
                    finalres[k,0]=file[i,0]
                    finalres[k,1]=file[i,1]+1
                    finalres[k,2]=res
                    k=k+1
    return finalres

#function which applies Poisson fluctions
def smearing_poisson_1b(mean,events,pe):
    mu=mean
    return events*(poisson.pmf(pe,mu))

#smearing function, reads in the 2D energy timing array 
def smearing_fct(file_in):
    Pe_res=np.zeros((12*25,3))
    PE_bins = np.linspace(6, 30, 25)
    k=0
    res=0
    for k in range(12):
        for j in range(32):
            mean=file_in[j+32*k,1]+0.5
            events=file_in[j+32*k,2]
            for i in range(25):                                                                                                                               
                Pe_res[i+25*k,0]=file_in[j+32*k,0]                                                                                                            
                Pe_res[i+25*k,1]=(PE_bins[i])                                                                                                                
                Pe_res[i+25*k,2]+=smearing_poisson_1b(mean,events,PE_bins[i])   
    return Pe_res


#function which rescales the sum of all 3 backgrounds to the current exposure
def bkgtemplate_2d(bkgfile):
    bkg2d=np.zeros((33*12,3))
    k=0
    for j in range(len(bkgfile)):
        for i in np.arange( 1, 34):
            bkg2d[k,0]=bkgfile[j,0]
            bkg2d[k,1]=i+3
            bkg2d[k,2]=bkgfile[j,i]*308.1/(308.1 + 153.5*2)
            k=k+1
    return bkg2d

#function which applies the energy and timing cuts on the neutron background
def load_neutron(neut):
    neu=np.zeros((33*12,3))
    k=0
    for i in range(len(neut)):
        if neut[i,1]>=4 and neut[i,1]<=36 and neut[i,0]<=5.75:
            neu[k,0]=neut[i,0]
            neu[k,1]=neut[i,1]
            neu[k,2]=neut[i,2]
            k=k+1
    return neu

#function to produce the 1d timing distribution taking the cuts in energy into account
def make_timing_dist_cutE(file_in,file_out):
    k=0
    for j in range(0,12):
        i=0
        st=0
        while i <= len(file_in)-1:
            if file_in[i,1]==0.25+j/2 and file_in[i,0]>=6 and file_in[i,0]<30:
                st=st+file_in[i,2]

            i=i+1

        file_out[k,0]=0.25+j/2
        file_out[k,1]=st
        k=k+1
    return file_out

#function to produce the 1d energy distribution taking the cuts in time into account 
def make_energy_dist_cutt(file_in,file_out):
    k=0
    for j in range(7,30,2):
        i=0
        st=0
        while i <= len(file_in)-1:
            if file_in[i,0]==j and file_in[i,1]<6:
                st=st+file_in[i,2]

            i=i+1

        file_out[k,0]=j
        file_out[k,1]=st
        k=k+1
    return file_out

#adds the three backgrounds together
def bkgunweighted_time(ACoff,ACon,COoff):
    bkgtsumnow=np.zeros((len(ACoff),3))
    for i in range(len(ACoff)):
        bkgtsumnow[i,0]=ACoff[i,0]
        bkgtsumnow[i,1]=ACoff[i,1]
        bkgtsumnow[i,2]=ACoff[i,2]+ACon[i,2]+COoff[i,2]
    return bkgtsumnow

#adds the three backgrounds together
def bkgunweighted_energy(ACoff,ACon,COoff):
    bkgesumnow=np.zeros((144,3))
    k=0
    for i in range(len(ACoff)):
        for j in range(12):
            if ACoff[i,1]==0.25+j/2 and ACoff[i,0]>=6 and ACoff[i,0]<30:
                bkgesumnow[k,0]=ACoff[i,0]
                bkgesumnow[k,1]=ACoff[i,1]
                bkgesumnow[k,2]=ACoff[i,2]+ACon[i,2]+COoff[i,2]
                k=k+1
    return bkgesumnow

#produces the 2D timing-energy distribution from the 1D timing and 1D energy distributions
def bkgreal_2ds(ACoff,ACon,COoff):
    timefile=np.zeros((12,2))
    energyfile=np.zeros((12,2))
    bkgenergy_data= bkgunweighted_energy(ACoff,ACon,COoff)
    bkgtime_data= bkgunweighted_time(ACoff,ACon,COoff)
    make_energy_dist_cutt(bkgenergy_data,energyfile)
    make_timing_dist_cutE(bkgtime_data,timefile)
    bkg2ds_data=np.zeros((12*12,3))
    k=0
    intt=0
    for ii in range(12):
        intt+=timefile[ii,1]
    for j in range(12):
        for i in range(12 ):
            rr=timefile[i,1]*energyfile[j,1]/intt
            bkg2ds_data[k,0]=timefile[i,0]
            bkg2ds_data[k,1]=energyfile[j,0]
            bkg2ds_data[k,2]=rr
            k=k+1
    return bkg2ds_data
    
    
