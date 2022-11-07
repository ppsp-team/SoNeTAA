"""
===========
Parameters file
===========
Configuration parameters for the study.
"""
import os
import mne
from socket import getfqdn
from mne.datasets import sample
from config import path_to_eeg_group

user = os.getlogin()  # Username of the user running the scripts
host = getfqdn()  # Hostname of the machine running the scripts

#configuration
data_path = sample.data_path()
subjects_dir = os.path.join(data_path, 'subjects')
mne.utils.set_config("SUBJECTS_DIR", subjects_dir, set_env=True)

###############################################################################################

#Parameters set by user when performing analyses at the subject level

method = "eLORETA" #method to use for source reconstruction. 
coreg = 'auto' #ask for automated coregistration
redo = True #if source reconstruction performed before, rerun it or not
plot = False #save various plots during source reconstruction in subfolder 'FIGURES'
path_to_mri = None #path to the subject MRI

###############################################################################################

#Parameters for inverse solution computation
snr = 1.0 
lambda2 = 1.0 / snr ** 2

freq_bands = {
    "delta": (0.5, 4.),
    "theta": (3., 8.),
    "alpha": (8., 13.),
    "low-alpha": (8., 10.),
    "high-alpha": (11., 13.),
    "beta": (12.5, 30.),
    "gamma": (25., 140.),
    "all": (0., 140.),
}

###############################################################################################
#Parameters for group-level analyses
frequency = "alpha"
level = 'source' #compute PSD/connectivity at sensor or source level - set to None if no analyses
consider_MRI = False #TEMPORARY: even for subjects with available MRI, use template MRI
folder_group = 'RESULTATS-GROUPE' #name of folder to store results
subject_to = 'fsaverage' #name of the subject to morph sources estimates to
plot_group = True
n_epochs = None #limit nb of epochs to gain time (None if no limit)

#set paths
path_to_results = os.path.join(path_to_eeg_group, folder_group) #results for group analyses
path_results_source_rec = os.path.join(path_to_eeg_group, 'Subjects_source_reconstruction') #results for source rec

#parameters for statistics 
n_permutations = 2000
tfce = True #whether or not to perform TFCE
dict_tfce = dict(start=0., step=0.2) #found no difference in results so far between step 0.5 and step 2 (time reasonnable)

exclus_file = "exclus_cohen.txt"

