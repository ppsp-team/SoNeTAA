import mne
import numpy as np
from scipy import signal
from parameters import freq_bands

def connectivity_one_epoch(stc1, stc2, sfreq, frequency):
    """ Calculates connectivity for one epoch between stc1 and stc2
    
        Parameters
        ----------
        stc : MixedSourceEstimate object
            The source estimates (for each label)
        epochs : mne.Epoch object
            the epoched EEg
        inverse_op : instance of InverseOperator
            The inverse operator. 
        subject_mri : str
             Name of the sub directory containing the subject MRI data
        subjects_dir_mri : str
            Path to the directory containing the subject MRI data (or the template MRI data)
        frequency : str
            The frequency band in which to compute the connectivity.
            Can be 'alpha', 'beta', 'gamma', 'theta', 'delta'.
        method : str
            method to compute the connectivity. Can be 'coh' or 'pli'
    """
    fmin, fmax = freq_bands[frequency]

    #connectivity shape : labels x labels x freq
    conn = np.zeros(((stc1.shape[0], stc2.shape[0], 100)))
    different = True
    if (stc1 == stc2).all():
        different = False
        for i in range(stc1.shape[0]):
            conn[i,i,:] = 1 

    for i in range(stc1.shape[0]): #i de 0 à 69
        for j in range(stc2.shape[0]):
            if different or (not different and i != j and np.all(conn[i,j,:] == 0)):
                freq, Cxy = signal.coherence(stc1[i], stc2[j], fs = sfreq, nperseg = 125, 
                                            nfft = 250, noverlap = 62)
                if (different and i==0 and j==0) or (not different and i ==0 and j ==1):
                    #sélection des bonnes fréquences
                    ind = [x for x in range(len(freq)) if freq[x] >= fmin and freq[x] <= fmax]
                    conn = conn[:,:,:len(ind)]
                #remplissage de la matrice
                value = [round(Cxy[x],2) for x in ind]               
                conn[i,j,:] = value
                if not different:
                    conn[j,i,:]  = value

    freq = [freq[x] for x in ind]
    #average conn over frequencies
    conn = np.average(conn, axis = 2)

    return conn, freq

def get_average_stc_over_epochs(epoch_condition, inverse_op, lambda2, method):
    #NB: better to apply inverse op to average of epochs but somehow results differ 
    stcs = mne.minimum_norm.apply_inverse_epochs(epoch_condition, inverse_op, lambda2, method,
                                                return_generator=True)
    #mean over stcs
    s =0
    for (n, x) in enumerate(stcs):
        s += x.data
    x.data = s/(n+1)

    return x 

