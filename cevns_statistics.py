#functions related to the new pull term distribution
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import numpy as np
#pull distribution 
def pdfdist(pullin,sig):
        return  np.exp(-(pullin-np.log(pullin+1))/sig**2)

def pulldist(sigma):
 #calculate random number                                                                                                                                                               
        k=-1
        while(k<0):
                r1=2.7*np.random.rand()-1
                p1=pdfdist(r1,sigma)
                r2=np.random.rand()
                if r2<=p1:
                        k=1
                        res=r2
                else:
                        k=-1
        return res
