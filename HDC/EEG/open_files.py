"""
Manage file names / looks for useful files such as MRI data

There are 2 classes:
-File_paths_sonetaa : will recover file paths for a sonetaa subject 
(specificaly path to the raw file & the preprocessed file, in order to create 
a new preprocessed epoch file with the right montage)
-File_paths : when you already have an epoch file, set paths to store the results 
& intermediates files of source reconstruction
"""

import os
import mne
from mne.datasets import sample
from parameters import path_to_eeg_group
from traitement_sources import inverse_operator

#Path to the directory containing the mne data
mne_data_path = sample.data_path()

class File_paths_sonetaa():
    """Object containing all the file paths for one sonetaa subject
    User has to provide either folder_to_save, either subject_name

    Parameters
    ----------
    folder_to_save : str
        Path the folder with EEG data containing MFF and PREPROCESSED folders 
        & where results will be stored. 
    choix : str
        'HDC' or 'RS'. Defaults to None if you don't have a folder with EEG data
    path to mri : str
        Path to the folder with MRI data of the subject. None is use a template MRI.
    subject_name : str 
        Subject id, to recover subject paths when there are folders MFF and PREPROCESSED 
        containing data for several subjects
    """
    def __init__(self, folder_to_save = None, choix = None, path_to_mri = None,
                subject_name = None):
        
        self.mne_data_path = mne_data_path
        if subject_name:
            self.path_to_mff = os.path.join(path_to_eeg_group, 'MFF')
            #folder where preprocessed file is stored
            self.path_to_preprocessed = os.path.join(path_to_eeg_group, 'PREPROCESSED')
            #folder where new preprocessed file will be stored
            self.path_to_preprocessed2 = os.path.join(path_to_eeg_group, 'PREPROCESSED2')
            
        elif folder_to_save:
            self.data_path_to_subject = folder_to_save
            self.path_to_mff = os.path.join(self.data_path_to_subject, 'MFF')
            #folder where preprocessed file is stored
            self.path_to_preprocessed = os.path.join(path_to_eeg_group, 'PREPROCESSED')
            #folder where new preprocessed file will be stored
            self.path_to_preprocessed2 = os.path.join(path_to_eeg_group, 'PREPROCESSED2')
        
        if not os.path.exists(self.path_to_preprocessed2):
            os.mkdir(self.path_to_preprocessed2)        
        
        #find raw file name
        self.f_raw = find_raw(self.path_to_mff, choix, subject_name)
        #find preprocessed file name
        self.preprocessed_fname, self.f_epochs = find_preprocessed(self.path_to_preprocessed,
                                                                    self.path_to_preprocessed2, 
                                                                    choix, subject_name)
        #check if the subject has an MRI
        self.mri_dispo = True if path_to_mri else False

    def open_raw_file(self):
        """Creates or open an epoched EEG raw file (FIF format)

            Returns
            -------
            raw : object mne.Raw
                EEG raw data
        """
        raw = mne.io.read_raw_egi(self.f_raw, preload = True)
        
        return raw

    def open_epochs_file(self, raw):
        """Creates or open an epoched EEG file (FIF format) if already exists

            Returns
            -------
            epochs : object mne.Epochs
                EEG epoched data
        """
        if not os.path.exists(self.f_epochs):
            montage = mne.channels.read_dig_egi(self.f_raw +'/coordinates.xml')

            montage.ch_names = raw.info['ch_names'][0:129]

            if self.mri_dispo == False:
                for digpoint in montage.dig:
                    digpoint['r'] = [round(element/100,4) for element in digpoint['r']]

            epochs = mne.read_epochs(self.preprocessed_fname)

            #  set the EEG electrode locations
            epochs.set_montage(montage)

            #interpolating bad channels
            epochs = epochs.interpolate_bads()
            #save the epochs in a FIF file
            epochs.save(fname = self.f_epochs, overwrite = True)

        else:
            epochs = mne.read_epochs(self.f_epochs)

        return epochs

class File_paths():
    """Object containing all the file paths for one subject

    Parameters
    ----------
    data_path : str
        Path to the folder with all the data
    folder_to_save : str
        Path the folder with EEG data & where results will be stored.
    choix : str
        'HDC' or 'RS'. Defaults to None if you don't have a folder with EEG data
    """
    def __init__(self, folder_to_save, method):
        
        self.mne_data_path = mne_data_path
        self.data_path_to_subject = os.path.join(folder_to_save)
        self.data_path_to_results = os.path.join(self.data_path_to_subject, 'RESULTATS/')
        self.data_path_to_figures = os.path.join(self.data_path_to_subject, 'FIGURES/')
        
        #creation des dossiers si n'existent pas
        if not os.path.exists(self.data_path_to_subject):
            os.mkdir(self.data_path_to_subject)
        if not os.path.exists(self.data_path_to_results):
            os.mkdir(self.data_path_to_results) 
        if not os.path.exists(self.data_path_to_figures): 
            os.mkdir(self.data_path_to_figures) 

        #save the file paths in variables
        self.f_epochs = os.path.join(self.data_path_to_results, 'epochs2-epo.fif')
        self.matrix = os.path.join(self.data_path_to_results, 'matrix-trans.fif')
        self.matrix_auto = os.path.join(self.data_path_to_results, 'matrix-auto-trans.fif')
        self.source_space = os.path.join(self.data_path_to_results, 'src-src.fif')
        self.fwd_file = os.path.join(self.data_path_to_results, 'fwd-fwd.fif')
        self.bem_surfaces = os.path.join(self.data_path_to_results, 'surfaces-bem.fif')
        self.bem_file = os.path.join(self.data_path_to_results, 'model-bem.fif')
        self.cov = os.path.join(self.data_path_to_results, 'cov-cov.fif')
        self.inv = os.path.join(self.data_path_to_results, 'inv-inv.fif')
        
        self.f_stc = os.path.join(self.data_path_to_results, 'eLORETA-evoked-stc.h5')
        
    def set_path_to_mri(self, path_to_mri = None, subject = None):
        """Unnecessary complicated function just to check if subject MRI was provided

            Parameters
            ----------
            path_to_mri : str
                path to the MRI data to use. If provided without subject, automatically
                recover what looks like an MRI in path_to_mri folder.
            subject : str
                Name of the MRI subject to use. If provided with path_to_mri, subject name
                will be searched for in path_to_mri folder. If provided without path_to_mri,
                subject name will be searched for in mne sample data folder.
                If no path_to_mri and subject are provided, set MRI to mne 'sample' brain.
        """
        #save paths to the MRI files
        if path_to_mri and subject:
            self.subjects_dir_mri = path_to_mri
            self.subject_mri = subject
            if path_to_mri == os.path.join(self.mne_data_path,'subjects'):
                self.mri_dispo = False
            else:
                self.mri_dispo = True
        elif not path_to_mri and not subject:
            #set paths to the template MRI
            self.mri_dispo = False
            self.subjects_dir_mri = os.path.join(self.mne_data_path, 'subjects')
            self.subject_mri = 'sample'
        elif path_to_mri and not subject:
            self.mri_dispo = True    
            liste_dossiers = os.listdir(self.subjects_dir_mri)
            #recovering the right MRI
            for dossier in liste_dossiers:
                if 'isotrope' in dossier:
                    self.subject_mri = dossier
        else:
            self.subjects_dir_mri = os.path.join(self.mne_data_path,'subjects')
            self.subject_mri = subject
            self.mri_dispo = False

        self.mri_fiducials = os.path.join(self.subjects_dir_mri, self.subject_mri, 
                                        'bem', str(self.subject_mri + '-fiducials.fif'))

    def change_mri_subject(self, subject_to = None):
        """Changes the subject mri name
        Called after coregistration & rescaling the MRI

        Parameters
        ----------
        subject_to : str
            name of the new mri subject. Defaults to None.
            if None, looks for the most recent created folder in subjects_dir_mri.
        """
        if  subject_to:
            self.subject_mri = subject_to
        else: #we look for the most recent MRI directory /!\ may cause error
            liste_dossiers = os.listdir(self.subjects_dir_mri)
            dirs_sort = sorted(liste_dossiers, 
                                key=lambda x: os.path.getctime(self.subjects_dir_mri + '/' + x), 
                                reverse=True)
            dirs_sort = [d for d in dirs_sort if 'sample' in d]            
            latest_subdir = dirs_sort[0]
            self.subject_mri = latest_subdir

    def change_matrix(self):
        """Changes the matrix file name to fi
        Called after coregistration
        """
        for truc in os.walk(self.data_path_to_results):
            liste_files = truc[2]
            for name_file in liste_files:
                if 'trans' in name_file:
                    self.matrix = self.data_path_to_results + name_file
    

    def erase_file(self, file_name):
        if os.path.exists(file_name):
            os.remove(file_name)

    def clean_files(self, all_files = False):
        """Erase useless files in directory

            Parameters
            ----------
            all_files : bool
                if True, erases all files in directory
        """
        #in case matrix was registered under a different name during previous manual coregistration
        self.change_matrix()

        for file_name in [self.f_epochs, self.matrix, self.matrix_auto, self.source_space,
                            self.fwd_file, self.bem_surfaces, self.bem_file, self.cov]:
            self.erase_file(file_name)

        if all_files == True:
            self.erase_file(self.f_stc)
            self.erase_file(self.inv)
    
def recover_epochs_sonetaa(path_or_name, choix, 
                            path_to_mri = None, clean = None):
    """Recovers files for source reconstruction 
    
        Parameters
        ----------
        path_or_name : str
            subject EEG C07 code OR path to the subject EEG data
        choix : str
            EEG file to analyze. Can be 'HDC' or 'RS'.
        
        Return
        ----------
        epochs : mne.Epochs
            Epochs object
    """
    print('Sujet :', path_or_name, ' Condition :', choix)

    # récupération des noms des fichiers de l'EEG

    if os.path.isdir(path_or_name):
        files = File_paths_sonetaa(folder_to_save = path_or_name, choix = choix, 
                                    path_to_mri = path_to_mri)
    else:
        files = File_paths_sonetaa(choix = choix, path_to_mri = path_to_mri,
                                    subject_name = path_or_name)
    if clean:
        if os.path.exists(files.f_epochs):
            os.remove(files.f_epochs)
    
    #create epochs from raw file / or open epoch fil
    if not os.path.exists(files.f_epochs):
        raw = files.open_raw_file()
        epochs = files.open_epochs_file(raw)
    else:
        epochs = mne.read_epochs(files.f_epochs)

    return epochs 

def recover_inverse_op(path_or_name, choix, method, path_results_subjects = None):
    """Recovers inverse op 
    
        Parameters
        ----------
        path_or_name : str
            subject EEG C07 code OR path to the subject EEG data
        choix : str
            EEG file to analyze. Can be 'HDC' or 'RS'.
        
        Return
        ----------
    """
    # récupération des noms des fichiers de l'EEG
    if os.path.isdir(path_or_name):
        files = File_paths(os.path.join(path_or_name, choix), method)
    else:
        path_or_name_eeg2 = os.path.join(path_results_subjects, path_or_name)
        files = File_paths(os.path.join(path_or_name_eeg2, choix), method)
    inverse_op = inverse_operator(files.inv)

    return inverse_op 

def find_raw(path_to_mff, choix, subject_name = None):
    """Find a subject raw file given the path to a MFF folder
    """
    file_list = os.listdir(path_to_mff)
    for f in file_list:
        if subject_name:
            if choix in f and subject_name in f:
                f_raw = os.path.join(path_to_mff, f)
        elif choix in f:
            f_raw = os.path.join(path_to_mff, f)
    
    return f_raw

def find_preprocessed(path_to_preprocessed, path_to_preprocessed2, choix, subject_name = None):
    """Find a subject preprocessed file given the path to a PREPROCESSED folder
    """
    file_list = os.listdir(os.path.join(path_to_preprocessed))

    for f in file_list:
        if subject_name:
            if subject_name in f and str(choix + '-epo.fif') in f: #and '.gz' not in f: #^pb fif et gz
                preprocessed_fname = os.path.join(path_to_preprocessed, f)
                name = f
        else:
            if str(choix + '-epo.fif') in f: #and '.gz' not in f: #^pb fif et gz
                preprocessed_fname = os.path.join(path_to_preprocessed, f)
                name = f
    subject_name = name.replace('-epo.fif.gz','')
    f_epochs = os.path.join(path_to_preprocessed2, str(subject_name + '2-epo.fif'))
    
    return preprocessed_fname, f_epochs
    