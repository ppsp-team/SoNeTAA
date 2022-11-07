"""
===========
spectral_density file
===========
Script for spectral density calculation (called by analyses_groupes)
"""

import mne
import numpy as np
from mne.time_frequency import psd_welch
from parameters import freq_bands

def power_spectral_density_welch(epochs, frequency):
    """Compute PSD with Welch method. AT SENSOR LEVEL 
        
        Parameters
        ----------
        epochs : mne.Epochs object
            epochs to compute PSD over
        frequency : str
            frequency band of interest
        
        Return
        ----------
        psds : numpy array
            array of PSDs, shape (n epochs x n sensors x n frequencies)
         freqs : list
            list of frequencies
         sensors : list
            list of sensors
        """
    # define frequencies of interest
    fmin, fmax = freq_bands[frequency]

    count = len(epochs)
    if count > 0:
        picks = mne.pick_types(epochs.info, eeg=True)
        psds, freqs = psd_welch(epochs, fmin=fmin, fmax=fmax,
                                picks=picks, n_jobs=1,
                                n_overlap=250, n_fft=1000, n_per_seg=500)
        psds = 10 * np.log10(psds) #sensors x epochs x freq
        sensors = epochs.info['ch_names']
    else:
        psds = np.nan
        freqs = np.nan
        sensors = np.nan

    return psds, freqs, sensors


def power_spectral_density_multitapper(epochs, inverse_op, method, 
                                        lambda2, frequency, label = None,
                                        return_type = ""):
    """Calculates PSD over all epochs in a frequency band using multitapper method
        AT SOURCE LEVEL
        Return a generator of PSDs to save time

        Parameters
        ----------
        epochs : mne.Epochs
            epoched data
        inverse_op :
            inverse operator to compute PSD at source level
        method : str
            method to solve inverse problem
        lambda2 : float
            parameter to solve inverse problem
        frequency : str
        
        label : str
            anatomic region to compute PSD in
        return_type : str
            return type of PSD, either a list of arrays or a list of source estimate 
        
        Return
        ----------
        list_data : list
            either a list of arrays or a list of source estimate
        stc.times : list
            list of frequencies
    """
    # define frequencies of interest
    fmin, fmax = freq_bands[frequency]
    bandwidth = 4.  # The bandwidth of the multi taper windowing function in Hz. 
    psds = mne.minimum_norm.compute_source_psd_epochs(epochs, 
                                inverse_op, lambda2 = lambda2,
                                 method = method, fmin = fmin, fmax = fmax,
                                 bandwidth = bandwidth, label = label,
                                 return_generator = True, verbose = True)
    return psds 

