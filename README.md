# High-z candles
A Bayesian Python code to confront the quasar data set with models beyond the standard model of elementary particle physics and models beyond the $\Lambda$CDM standard cosmology. 

# Abstract
The Hubble diagram of quasars, as candidates to ``standardizable" candles,
has been used to measure the expansion history of the Universe at late times,
up to very high redshifts ($z \sim 7$). It has been shown that this history, as
inferred from the quasar dataset, deviates at $\gtrsim 3 \sigma$ level from the
concordance ($\Lambda$CDM) cosmology model preferred by the cosmic microwave
background (CMB) and other datasets. In this article, we investigate whether
new physics beyond $\Lambda$CDM (B$\Lambda$CDM) or beyond the Standard Model
(BSM) could make the quasar data consistent with the concordance model. We
first show that an effective redshift-dependent relation between the quasar UV
and X-ray luminosities, complementing previous phenomenological work in the
literature, can potentially remedy the discrepancy. Such a redshift dependence
can be realized in a BSM model with axion-photon conversion in the
intergalactic medium (IGM), although the preferred parameter space could be in
mild tension with various other astrophysical constraints on axions, depending
on the specific assumptions made regarding the IGM magnetic field. We briefly
discuss a variation of the axion model that could evade these astrophysical
constraints. On the other hand, we show that models beyond $\Lambda$CDM such as
one with a varying dark energy equation of state ($w$CDM) or the
phenomenological cosmographic model with a polynomial expansion of the
luminosity distance, cannot alleviate the tension.

# How to run

Requirements
-----------------------------------------

1. Python  
2. numpy  
3. scipy  
4. emcee  
5. corner  


How to run the MCMC
-----------------------------------------

In the terminal:

	$ python cosmo_axions_run.py -L likelihoods/ -o path/to/your/chain/output/ -i inputs/the_param_file.param -N number_of_points -w number_of_walkers
	
As a rule of thumb, a good number to start most runs is `-N 40000 -w 100`. You can adjust the number depending on the convergence test during analysis. Be careful that analyzing an unfinished chain will likely break the run. Therefore, it is better to copy the unfinished chain to a different folder and analyze from there. 


How to analyze the chain
-----------------------------------------
After the runs are finished, you can analyze them with:

	$ python cosmo_axions_analysis.py -i path/to/your/chain/output/

Once the analysis is done, if you wanna output the contours in ma-ga space from the frequentist likelihood ratio test, do:

	$ python bin_chi2.py -c path/to/your/chain/output/ -b number_of_ma-ga_bins

where the argument with flag -b bins the ma-ga parameter space in order to minimize the chi2 in each bin. A value of ~50 is good enough.

For the best fit point, it can be extracted by the `parse()` from `bin_chi2.py`. A sample is given below: 

	(bf_chi2,
	 x_mesh,
	 y_mesh,
	 chi2_mins,
	 idx_mins_global,
	 x_arr, y_arr,
	 delta_arr,
	 _,
	 pts, 
	 blobs) = parse(directory="path_to_chain",
						chain_name="chain_1.h5",
						x_name="OmL",
						y_name="h0",
						bins=10)
The specific location of the best fit in the chain is output, which can then be located in the flat chain with `pts[<index_of_best_fit>`. 




Bibtex entry
-----------------------------------------
If you use this code or find it in any way useful for your research, please cite [Sun, Buen-Abad, Fan (2023)](https://arxiv.org/abs/2309.xxxxx). The BibTeX entry is:

	@article{Sun:2023,
	    author = "Sun, Chen and Buen-Abad, Manuel A. and Fan, JiJi",
	    title = "{Probing New Physics with High-Redshift Quasars: Axions and Non-standard Cosmology}",
	    eprint = "2309.xxxxx",
	    archivePrefix = "arXiv",
	    primaryClass = "astro-ph.CO",
	    month = "09",
	    year = "2023"
	}


The main routine and the routine of fitting SNIa is based on [Buen-Abad, Fan, & Sun (2020)](https://arxiv.org/abs/2011.05993). Please also consider citing this publication with the following BibTeX entry:

	@article{Buen-Abad:2020zbd,
	    author = "Buen-Abad, Manuel A. and Fan, JiJi and Sun, Chen",
	    title = "{Constraints on Axions from Cosmic Distance Measurements}",
	    eprint = "2011.05993",
	    archivePrefix = "arXiv",
	    primaryClass = "hep-ph",
	    month = "11",
	    year = "2020"
	}
