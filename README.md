<img src="logo.png" alt="logo"
	title="7stats logo" width="350" height="250" />
  
# 7stats 
 
| arXiv |
|:-----:|
|[![arXiv](https://img.shields.io/badge/arXiv-2008.06062-orange.svg)](https://arXiv.org/abs/2008.06062)|



## Introduction

`7stats` is a tool which allows for the calculation of the p value and confidence levels using a Feldmann-Cousins approach for standard and new physics scenarios in the CEvNS process.
As an explicit example we demonstrate the p value calculation with a Monte Carlo simulation for the SM, and the calculation of confidence levels with the FC approach 
for neutrino non-standard interactions with a heavy mediator (mass above 100 MeV) using the COHERENT CsI data.

## Usage
The code is written in python3, numpy and scipy are needed for the calculations.

The data files for the COHERENT CsI data are included as well. If you use this data please cite 
`Science 357 (2017) no.6356, 1123-1126,[arXiv:1708.01294 [nucl-ex]] `
and `arXiv:1804.09459 [nucl-ex]`.


## Example code

### Auxillary code
There are three code files which contain auxillary functions needed to calculate the number of CEvNS signal events at COHERENT CsI. 

`cevns.py` contains the functions specific to the CEvNS process like the cross section for heavy NSI mediators.

`cevns_accelerators.py` contains the funtions which are specific to CEvNS from SNS neutrinos like the energy structure of the different neutrino flavors coming 
from the SNS beam, the function used for photoelectron smearing, the CsI detector efficiency and functions needed for the background treatment.

`coherent_CsI_real.py` contains the functions which calculate the signal at COHERENT CsI, apply the dector efficiency, or rebin the data to the desired number of bins. 

`cevns_statistics.py` is only needed if one wishes to use the new pull term parametrization. In this case the chi2 function and the functions for the MC calculation
need to be changed as well.

### 	Scan code
The files `scan_nsi_csir_eeonly_2t1e_nomarg_ee.py`, `scan_nsi_csir_eeonly_2t1e_marg_ee.py`, `scan_lmad_2t1e.py`
demonstrate the calculation of the test statistic using two timing bins, scanning over one NSI parameter (or two in the case of LMA Dark) and marginalizing over the pull terms 
and the remaining NSI parameters (in the case of `scan_nsi_csir_eeonly_2t1e_marg_ee.py`). 

### Monte Carlo simulation
The files `analysis_code_pval_nsi_eeonly_2t1e_nomarg.py`, `analysis_code_pval_nsi_eeonly_2t1e_marg.py`, `analysis_code_pval_nsi_lmad_2t1e.py` demonstrate the Monte Carlo
simulation which is used to calculate the p value. 

`analysis_code_pval.py` contains the function to calculate the p value. It uses the output data file from the scan code as well as the one from the MC simulation. 
We include in `datafile_out` the output of the MC simulation for several NSI scenarios as well as the output of the scan code for these cases.

## Results 
For convience we include the datafiles containing the results of the MC estimation for the p values of various NSI models (only one non-zero NSI parameter, allowing 
for all NSI parameters to be non-zero simultanously and only non-zero eps_ee^u and eps_mm^u for LMA Dark) using two timing bins. These files can be found in the 
`datafiles_pval` folder 
The first column is the value of the NSI parameter, the second column is the p value.

## Bugs and Features
If any bugs are identified or any features suggested, please use the tools (issues, pull requests, etc.) on github.

## Reference
If you use this code please reference **[arXiv:2008.06062](https://arxiv.org/pdf/2008.06062.pdf)**

