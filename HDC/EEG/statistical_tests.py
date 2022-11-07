"""
===========
Statistical tests file
===========
For Student t-test with cluster permutation correction
"""
import os
import numpy as np
from mne.stats import (permutation_cluster_1samp_test, permutation_cluster_test)
from scipy import stats as stats
from parameters import n_permutations, dict_tfce
from mne.stats import ttest_ind_no_p


def set_threshold(n_subjects, p_threshold, tfce = False):
    """Set threshold for significant samples selection. 
    Higher the threshold, more severe the selection

    Parameters
    ----------
    n_subjects : int
        number of subjects
    p_threshold : float
        when selecting normal threshold, threshold is student distribution value at 1-p_threshold
    tfce : bool
        if True, use a TFCE statistical test insted of basic cluster permutation test

    Return
    ---------
    threshold : a dict or a float
        threshold for cluster permutation test
    """
    if tfce:
        threshold = dict_tfce
    else:
        p_threshold = 0.05
        threshold = -stats.distributions.t.ppf(p_threshold / 2., n_subjects - 1)
        print('t_threshold', threshold)

    return threshold

def cluster_permutation_test_1sample(X, connectivity = None, tails = 0, 
                                    tfce = False):
    """
    Cluster permutation t-test on 1 sample

    Parameters
    ----------
    X : numpy array
        array of shape (n subjects x n sources/sensors)
    connectivity : scipy matrix
        shape (n_channels, n_channels). Default is None.
    tails : int
            If tail is 1, the statistic is thresholded above threshold. 
            If tail is -1, the statistic is thresholded below threshold. 
            If tail is 0 (default), the statistic is thresholded on both 
            sides of the distribution.
    tfce : bool
        if True, use a TFCE statistical test insted of basic cluster permutation test
    
    Return
    ----------
    clu : tuple 
        made of T_obs, clusters, cluster_p_values, H0.
        T-statistic observed for all variables, list of clusters, array of 
        p-values for each cluster, Max cluster level stats observed under permutation.
    """    
    n_subjects = X.shape[0]
    threshold = set_threshold(n_subjects, p_threshold = 0.005, tfce = tfce)
    
    clu = permutation_cluster_1samp_test(X, n_permutations = n_permutations,
                                        connectivity = connectivity,
                                        threshold=threshold, tail=tails,
                                        out_type = 'indices')

    return clu

def cluster_permutation_test_2sample(X1, X2, connectivity = None, tails = 0,
                                    tfce = False):
    """
    Cluster permutation t-test on 2 samples (asd and td)

    Parameters
    ----------
    X : numpy array
        array of shape (n subjects x n sources/sensors)
    connectivity : scipy matrix
        shape (n_channels, n_channels). Default is None.
    tails : int
            If tail is 1, the statistic is thresholded above threshold. 
            If tail is -1, the statistic is thresholded below threshold. 
            If tail is 0 (default), the statistic is thresholded on both 
            sides of the distribution.
    tfce : bool
        if True, use a TFCE statistical test insted of basic cluster permutation test
    
    Return
    ----------
    clu : tuple 
        made of T_obs, clusters, cluster_p_values, H0.
        T-statistic observed for all variables, list of clusters, array of 
        p-values for each cluster, Max cluster level stats observed under permutation.
    """
    n_subjects = X1.shape[0]
    threshold = set_threshold(n_subjects, p_threshold = 0.005, tfce = tfce)

    if len(X1.shape) < 3:
        X1 = X1.reshape(X1.shape[0], X1.shape[1], 1)
        X2 = X2.reshape(X2.shape[0], X2.shape[1], 1)
    X1 = np.transpose(X1, [0, 2, 1])
    X2 = np.transpose(X2, [0, 2, 1])
    
    clu = permutation_cluster_test([X1, X2], connectivity = connectivity, 
                                    n_permutations = n_permutations, 
                                    threshold=threshold, tail = tails, 
                                    stat_fun = ttest_ind_no_p)
    
    return clu
