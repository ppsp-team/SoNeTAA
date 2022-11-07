""" 
    Calls source reconstruction functions for a single subject
"""

import os
import open_files as op
import coregistration as cor
import traitement_sources as fct
import visualization as viz
from mne import read_source_estimate, read_cov

os.environ['ETS_TOOLKIT'] = 'qt4'
os.environ['QT_API'] = 'pyqt5'

def get_RS_condition(epochs_RS, cond):
    if cond is "eo":
        k = [k for k in epochs_RS.event_id.keys() if 'eo' in k.lower()]
    elif cond is "ec":
        k = [k for k in epochs_RS.event_id.keys() if 'ec' in k.lower()]
    return epochs_RS[k]

def source_reconstruction(epochs, method, lambda2, folder_to_save, 
                            path_to_mri = None, coreg = 'auto', 
                            plot = False, redo = False, noise_cov = None):
    """Reconstruct sources for one epoch file. 
        Automatically uses subject MRI or a template MRI.
    
        Parameters
        ----------
        epochs : mne.Epochs
            Epochs object
        method : str
            chosen method for source reconstruction
        folder_to_save : str
            Path to the folder where you want to store your results. 
        path_to_mri : str
            Path to the folder containing MRI data. Defaults to None.
        coreg : str
            'auto' (default) or manual.
        plot : str
            whether or not to plot the graphs. Defaults to False.
        redo : bool
            If True, erase the previous source reconstruction files and do it from the start.
            Default to False.

        Return
        --------
        stcs_evoked :
        noise_cov : noise covariance matrix
    """
    # récupération des noms des fichiers / creation du répertoire pour stocker fichiers
    files = op.File_paths(folder_to_save, method)
    files.set_path_to_mri(path_to_mri)

    #erase files to do it all over again
    if redo == True:
        files.clean_files(all_files = True)

    #save epochs in a f_epochs file (needed for coregistration)
    if not os.path.exists(files.f_epochs):
        epochs.save(fname = files.f_epochs)

    #if source estimates file already exists, the function does not do anything
    if os.path.exists(files.f_stc):
        print('Reconstruction de sources déjà effectuée pour le sujet ' + folder_to_save + '!')
        stcs_evoked = read_source_estimate(files.f_stc)
        if not noise_cov:   
            noise_cov = read_cov(files.cov)
        return stcs_evoked, noise_cov

    # moyenne des époques
    evoked = epochs.average()
    
    # visualisation
    if plot:
        viz.visualization_epochs(files, epochs)
        viz.visualization_evoked(files, evoked)
    
    if coreg == 'auto': #coregistration manuelle
        trans = cor.coregistration_automated(files)
        if plot:
            viz.check_alignment(epochs.info, trans, files.subject_mri, files.subjects_dir_mri,
                                files.data_path_to_figures, 'alignment_after_automated_coreg')
    
    #bem
    bem = fct.boundary_element_model(epochs, files.bem_file, files.bem_surfaces, 
                                    files.subject_mri, files.subjects_dir_mri, 
                                    files.data_path_to_figures, plot)

    # espace des sources apres la coreg car creation du source space a partir de l'irm (rescalé)
    src = fct.creation_source_space(files.source_space, files.subjects_dir_mri, 
                                    files.subject_mri, bem, files.data_path_to_figures,
                                    plot)

    # forward solution (During fwd calculation, vertices can be removed from src)
    fwd = fct.forward_solution(epochs, files.fwd_file, files.subject_mri, 
                                files.subjects_dir_mri, bem, src, trans)

    if plot:
        viz.check_alignment(epochs.info, trans, files.subject_mri, files.subjects_dir_mri,
                            files.data_path_to_figures, 'alignment_src_bem', 
                            src, bem)

    # compute covariance matrix between epochs, if cov matrix not provided by user
    if not noise_cov:
        #get epochs in RS eyes open condition
        epochs_EO = get_RS_condition(epochs, cond = "eo")
        noise_cov = fct.noise_covariance(files.cov, epochs_EO, files.data_path_to_figures, 
                                        plot = plot)

    # Compute inverse solution and stcs for average of epochs (useless, can be removed)
    stcs_evoked = fct.inverse_solution(files.f_stc, evoked, epochs, method, 
                                        lambda2, files.inv, fwd, noise_cov)

    #clean the created files not needed anymore
    files.clean_files()

    return stcs_evoked, noise_cov

