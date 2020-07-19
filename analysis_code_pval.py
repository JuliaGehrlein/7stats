#code to calculate the p value
import numpy as np
from scipy.stats import chi2
def pvalcalc(filein,histofile):
        k=len(histofile)
        l=0
        fileout=np.zeros((len(filein),2))
        for m in range(len(filein)):
                right=0
                for i in range(k):
                        if histofile[i]>=filein[m,1]:
                                right+=1
                fileout[l,0]=filein[m,0]
                fileout[l,1]=right/k
                l=l+1
        return fileout


pvalnsiee2t1enomarg=np.loadtxt("datafiles_out/pval_ee_nomarg_2t1e_gauss_final.txt")
epsee2t1enomarg=np.loadtxt("datafiles_out/chi2_eeonly_2t1e_nomarg_nosmear_gauss006.txt")
np.savetxt("datafiles_out/results_eeonly_2t1e_nomarg_nosmear_gauss006.txt",pvalcalc(epsee2t1enomarg,pvalnsiee2t1enomarg))
