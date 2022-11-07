"""
===========
Some useful functions
===========
"""
import os
import numpy as np
from parameters import exclus_file
import analyses_individuelles as source


def get_epochs_in_condition(cond, epochs_HDC, epochs_RS):
    """Return epochs in the desired condition

        Parameters
        ----------
        cond : str
            Desired condition
        epochs_HDC : mne.Epochs object
            epochs in HDC condition
        epochs_RS : mne.Epochs object
            epochs in RS condition
        
        Return
        ----------
        epoch_condition : mne.Epochs object
            epochs in condition 'cond'
    """
    if cond == 'HDC':
        #take only the interaction period
        epoch_condition = epochs_HDC["INTE"]
    elif cond in ['eyeo', "eo", 'RS']:
        epoch_condition = source.get_RS_condition(epochs_RS, "eo")
    else:
        epoch_condition = source.get_RS_condition(epochs_RS, "ec")
    
    return epoch_condition


def check_list_cond(liste_cond):
    """Check that list cond was correctly specified"""

    if liste_cond in [["HDC", "RS"], ["HDC", "eyeo"], ["eyec", "eyeo"], ["ec", "eo"]]:
        liste_cond.reverse()
    if liste_cond not in [["RS", "HDC"],["eyeo", "HDC"], ["eyeo", "eyec"], ["eo", "ec"]]:
        return None
    
    return liste_cond, liste_cond[0]

def sujets_exclusion_psds(path_to_figures):
    """
        Read a subject exclusion information
    """
    indices_asd, indices_td = [], []
    with open(os.path.join(path_to_figures + "/"+ exclus_file), "r") as f:
        for line in f:
            row = line.rstrip('\n').split(",")
            groupe, indice = row[0], int(row[1])
            if groupe == "td":
                indices_td.append(indice)
            elif groupe == "asd":
                indices_asd.append(indice)

    return indices_td, indices_asd



def add_to_array(array, values, axis = 0):
    if array is None:
        array = values
    else:
        array = np.concatenate((array, values), axis = axis)

    return array

def zscore(M2, M1, S1):
    return np.nan_to_num((M2 - M1)/ S1)

def cohen_d(X1, X2):
    """
        Calculates cohen d effect size between X1 and X2
        X1 and X2 : shape n subjects x n channels"""

    # calculate the size of samples
    n1, n2 = X1.shape[0], X2.shape[0]
    # calculate the variance of the samples
    s1, s2 = np.nanvar(X1, axis=0, ddof=1), np.nanvar(X2, axis=0, ddof=1)
    # calculate the pooled standard deviation
    s = np.power(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2), 0.5)
    # calculate the means of the samples
    u1, u2 = np.nanmean(X1, axis=0), np.nanmean(X2, axis=0)
    # calculate the effect size
    cohen = (u2 - u1) / s
    cohen[cohen == -np.inf] = 0
    cohen = np.nan_to_num(cohen)

    return cohen

def save_results(level, liste_cond, frequency, liste_participants,
                tfce = None, path_to_figures = "", method = "",
                score = "", participants = "all", **kwargs):
    """
        Save results in npz file for later plotting
    """

    #save values
    title = "{}_{}_vs_{}_{}_{}_{}_{}".format(level, liste_cond[1], liste_cond[0], frequency, method, score, participants)

    if tfce:
        title += "_tfce"

    k = 0
    while os.path.isfile(path_to_figures + "/" + title + str(k) + ".npz"):
        k += 1
    outfile = path_to_figures + "/" + title + str(k)
    #save arrays in outfile
    np.savez(outfile, **{k: v for k, v in kwargs.items()})

    return outfile
