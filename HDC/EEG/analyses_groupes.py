"""
===========
Group analysis file
===========
Script for group level analyses.
(Source reconstruction for all subjects and statistical tests)
"""

import mne
import os
import numpy as np
import open_files as op 
from mne.channels import find_ch_adjacency

import source_reconstruction as source
from tools import (add_to_array, zscore, save_results, 
                    check_list_cond, get_epochs_in_condition, 
                    sujets_exclusion_psds)
from open_files import recover_epochs_sonetaa
from spectral_density import (power_spectral_density_multitapper, power_spectral_density_welch)
from morphing import (morph_surface_source_space, morph_volume_source_space, 
                    compute_morph_surface, compute_morph_volume)
from statistical_tests import (cluster_permutation_test_1sample, 
                                cluster_permutation_test_2sample) 
from parameters import (subjects_dir)

#global exclus_file
#global subjects_dir

################################################################################################################
#### SOURCE RECONSTRUCTION
################################################################################################################


def source_reconstruction_group(path_to_eeg_group, path_results_source_rec, list_subjects, 
                                method, lambda2, coreg, plot, redo):
    """Reconstruct sources for all EEG subjects. 
        Automatically uses subject MRI or a template MRI.
    
        Parameters
        ----------
        liste_participants : list
            list of list of subjects path to eeg and path to mri.
        method : str
            method for source reconstruction
    """

    for (path_or_name_eeg, path_to_mri) in list_subjects:
        epochs_HDC = recover_epochs_sonetaa(path_or_name_eeg, 'HDC', path_to_mri, True)
        epochs_RS = recover_epochs_sonetaa(path_or_name_eeg, 'RS', path_to_mri, True)

        #set folders to save results of source reconstruction
        if os.path.isdir(path_or_name_eeg):
            #save results in subject folder
            folder_to_save_HDC = path_or_name_eeg + '/HDC'
            folder_to_save_RS = path_or_name_eeg + '/RS'
        else:
            #create a subject folder to save results
            path_results_subject = os.path.join(path_results_source_rec, path_or_name_eeg)
            if not os.path.exists(path_results_subject):
                os.mkdir(path_results_subject)
            folder_to_save_HDC = os.path.join(path_results_subject, 'HDC')
            folder_to_save_RS = os.path.join(path_results_subject, 'RS')
        #source rec for RS
        _, noise_cov = source.source_reconstruction(epochs_RS, method, lambda2, folder_to_save_RS, 
                                        path_to_mri = path_to_mri, coreg = coreg, 
                                        plot = plot, redo = redo)
        #source rec for HDC, using RS noise cov matrix
        _, _ = source.source_reconstruction(epochs_HDC, method, lambda2, folder_to_save_HDC,
                                        path_to_mri = path_to_mri, coreg = coreg, 
                                        plot = plot, redo = redo, noise_cov = noise_cov)
        
    print('Source reconstruction computed for all subjects with method ' + method + '!')

################################################################################################################
#### POWER ANALYSIS
################################################################################################################

def set_function_names(level):
    if level == "cortex":
        compute_morph, morph_source_space = compute_morph_surface, morph_surface_source_space
    elif level in ["cereb", "cerebellum"]:
        compute_morph, morph_source_space = compute_morph_volume, morph_volume_source_space 
    
    return compute_morph, morph_source_space

def power_analysis_sensor_groups_psds(liste_ASD, liste_TD, path_to_figures, 
                                    liste_cond, frequency):
    """PSD analyses at sensor level between TD and ASD, between 2 conditions
    Save only the PSDs for later statistical analyses
    """
    liste_cond, cond1 = check_list_cond(liste_cond)
    mean_psd_asd1, mean_psd_asd2 = None, None
    mean_psd_td1, mean_psd_td2 = None, None
    std_psd_asd1, std_psd_asd2 = None, None
    std_psd_td1, std_psd_td2 = None, None

    for i, liste in enumerate([liste_ASD, liste_TD]):
        for (path_or_name_eeg, path_to_mri) in liste:
            epochs_RS = recover_epochs_sonetaa(path_or_name_eeg, 'RS', path_to_mri)
            epochs_HDC = recover_epochs_sonetaa(path_or_name_eeg, 'HDC', path_to_mri)
            
            #compute PSD in cond 1 and cond 2
            for cond in liste_cond:
                epochs_condition = get_epochs_in_condition(cond, epochs_HDC, epochs_RS)
                psd, _, _ = power_spectral_density_welch(epochs_condition, frequency)
                
                #psd : variable nb of epochs (1) x n sensors x frequencies
                # average over frequencies n epochs x n sensors
                psd = np.mean(psd, axis = 2)
                moy_epochs = np.mean(psd, axis =0) #shape n sensors
                moy_epochs = np.reshape(moy_epochs, (1, moy_epochs.shape[0]))
                std_epochs = np.std(psd, axis = 0)
                std_epochs = np.reshape(std_epochs, (1, std_epochs.shape[0]))

                if cond is cond1 and i == 0:
                    mean_psd_asd1 = add_to_array(mean_psd_asd1, moy_epochs)
                    std_psd_asd1 = add_to_array(std_psd_asd1, std_epochs)
                elif cond is cond1 and i != 0:
                    mean_psd_td1 = add_to_array(mean_psd_td1, moy_epochs)
                    std_psd_td1 = add_to_array(std_psd_td1, std_epochs)
                elif cond is not cond1 and i == 0:
                    mean_psd_asd2 = add_to_array(mean_psd_asd2, moy_epochs)
                    std_psd_asd2 = add_to_array(std_psd_asd2, std_epochs)
                elif cond is not cond1 and i != 0:
                    mean_psd_td2 = add_to_array(mean_psd_td2, moy_epochs)
                    std_psd_td2 = add_to_array(std_psd_td2, std_epochs)

    save_results("PSD_sensors", liste_cond, frequency, liste_TD+liste_ASD, 
                    path_to_figures = path_to_figures, Mtd1 = mean_psd_td1, Mtd2 = mean_psd_td2, 
                    Masd1 = mean_psd_asd1, Masd2 = mean_psd_asd2, 
                    Std1 = std_psd_td1, Std2= std_psd_td2,
                    Sasd1 = std_psd_asd1, Sasd2 = std_psd_asd2)

def power_analysis_source_groups_psds(path_results_subjects, liste_ASD, liste_TD, path_to_figures, 
                                method, lambda2, subject_to, area, liste_cond,
                                frequency, tfce = True, n_epochs = None):
    """ 
        PSD analyses at source level between TD and ASD, between 2 conditions
        Write the Power Spectral densities for later statistical analyses

        path_results_subjects: path to where the subjects informations are stored
        liste_ASD: list of ASD subjects
        liste_TD: list of TD subjects
        path_to_figures: path to the folder where the figures and files are saved
        method: method for source reconstruction (MNE or eLORETA)
        lambda2: parameter to solve inverse problem
        subject_to: name of the subject to morph the subjects PSDs to a common source space (used fsaverage)
        area: cerebellum or cortex
        liste_cond: list of conditions of interest (HDC / RS)
        frequency: frequency of interest (alpha, theta...)
    """
    liste_cond, cond1 = check_list_cond(liste_cond)
    compute_morph, morph_source_space = set_function_names(area)

    #compute zscore for each participant between HDC and RS
    mean_psd_asd1, mean_psd_asd2 = None, None
    mean_psd_td1, mean_psd_td2 = None, None
    std_psd_asd1, std_psd_asd2 = None, None
    std_psd_td1, std_psd_td2 = None, None

    for i, liste in enumerate([liste_ASD, liste_TD]):
        for (path_or_name_eeg, path_to_mri) in liste:
            epochs_RS = recover_epochs_sonetaa(path_or_name_eeg, 'RS', path_to_mri)
            epochs_HDC = recover_epochs_sonetaa(path_or_name_eeg, 'HDC', path_to_mri)            
            
            #recover inverse operator  and compute morph matrix for later morphing
            inverse_op = op.recover_inverse_op(path_or_name_eeg, 'RS', method, path_results_subjects)
            if i == 0:
                morph, src = compute_morph(subjects_dir, inverse_op, subject_to)
            
            for cond in liste_cond:
                #recover inverse operator
                if cond in ["RS", "eo", "eyeo", "ec", "eyec"]:
                    inverse_op = op.recover_inverse_op(path_or_name_eeg, 'RS', method, path_results_subjects)
                else:
                    inverse_op = op.recover_inverse_op(path_or_name_eeg, 'HDC', method, path_results_subjects)
                epochs_condition = get_epochs_in_condition(cond, epochs_HDC, epochs_RS)
                #compute PSD : a generator of PSDs over epochs (psd = sources x freq)
                psd = power_spectral_density_multitapper(epochs_condition[:n_epochs], inverse_op, 
                                                        method, lambda2, frequency, return_type='stc')
                
                #compute the mean and std of the subject PSDs over epochs
                array_epochs = None
                for (_, stc) in enumerate(psd):
                    #transport PSD to common source space
                    psd_morphed = stc.mean()
                    psd_morphed = morph_source_space(morph, src, psd_morphed, subjects_dir, 
                                            inverse_op, subject_to, path_to_figures)
                    array_epochs = add_to_array(array_epochs, psd_morphed.data, axis = 1)

                moy_epochs = np.mean(array_epochs, axis = 1)
                moy_epochs = np.expand_dims(moy_epochs, axis = 0)
                std_epochs = np.std(array_epochs, axis = 1)
                std_epochs = np.expand_dims(std_epochs, axis = 0)
               
                #add the mean and std to the relevant list 
                if cond is cond1 and i == 0:
                    mean_psd_asd1 = add_to_array(mean_psd_asd1, moy_epochs)
                    std_psd_asd1 = add_to_array(std_psd_asd1, std_epochs)
                elif cond is cond1 and i != 0:
                    mean_psd_td1 = add_to_array(mean_psd_td1, moy_epochs)
                    std_psd_td1 = add_to_array(std_psd_td1, std_epochs)
                elif cond is not cond1 and i == 0:
                    mean_psd_asd2 = add_to_array(mean_psd_asd2, moy_epochs)
                    std_psd_asd2 = add_to_array(std_psd_asd2, std_epochs)
                elif cond is not cond1 and i != 0:
                    mean_psd_td2 = add_to_array(mean_psd_td2, moy_epochs)
                    std_psd_td2 = add_to_array(std_psd_td2, std_epochs)

    save_results("PSD_"+area, liste_cond, frequency, liste_TD + liste_ASD, tfce, path_to_figures, 
                    method, Mtd1 = mean_psd_td1, Mtd2 = mean_psd_td2, 
                    Masd1 = mean_psd_asd1, Masd2 = mean_psd_asd2, 
                    Std1 = std_psd_td1, Std2= std_psd_td2,
                    Sasd1 = std_psd_asd1, Sasd2 = std_psd_asd2)
    
def power_analysis_statistics(path_results_subjects, psd_file, path_to_figures, list_subjects,
                                liste_TD, liste_ASD, method, subject_to, level, liste_cond,
                                tfce = True, compare = "all", score = "zscore", frequency = "theta"):
    """
        Open an array of PSDs and compute statistics (inter and/or intra group)
    """
    #exclusions
    indices_td, indices_asd = sujets_exclusion_psds(path_to_figures)
    indices_asd, indices_td = np.asarray(indices_asd), np.asarray(indices_td)

    liste_cond, _ = check_list_cond(liste_cond)
    (path_or_name_eeg, path_to_mri) = list_subjects[0]  
    epochs_RS = recover_epochs_sonetaa(path_or_name_eeg, 'RS', path_to_mri)  

    if level not in ["sensor", "sensors"]:
        inverse_op = op.recover_inverse_op(path_or_name_eeg, 'RS', method, path_results_subjects)
        compute_morph, _ = set_function_names(level)
        _, src = compute_morph(subjects_dir, inverse_op, subject_to)

    parent_folder = os.path.dirname(psd_file)

    # Load dara
    npzfile = np.load(psd_file, allow_pickle = True)

    # Exclusions
    Mtd1 = np.delete(npzfile["Mtd1"], obj = indices_td, axis = 0)
    Mtd2 = np.delete(npzfile["Mtd2"], obj = indices_td, axis = 0)
    Masd1 = np.delete(npzfile["Masd1"], obj = indices_asd, axis = 0)
    Masd2 = np.delete(npzfile["Masd2"], obj = indices_asd, axis = 0)
    Std1 = np.delete(npzfile["Std1"], obj = indices_td, axis = 0)
    Sasd1 = np.delete(npzfile["Sasd1"], obj = indices_asd, axis = 0)

    #compute z score
    score_td = zscore(Mtd2, Mtd1, Std1)
    score_asd = zscore(Masd2, Masd1, Sasd1)
        
    if level in ["sensor", "sensors"]:
        connectivity, _ = find_ch_adjacency(epochs_RS.info, ch_type='eeg')
    else:
        connectivity = mne.spatial_src_connectivity(src)
    if compare == "all":
        clu = cluster_permutation_test_1sample(score_td, connectivity, tfce = tfce)
        save_results(level, liste_cond, frequency, liste_TD, tfce, parent_folder, 
                    method, score = score, Xtd=score_td, T_obs = clu[0], clusters = clu[1],
                    clusters_pvalues = clu[2], participants = "td")
        
        clu = cluster_permutation_test_1sample(score_asd, connectivity, tfce = tfce)
        save_results(level, liste_cond, frequency, liste_ASD, tfce, parent_folder, method, 
                    score = score, Xasd=score_asd, T_obs = clu[0], clusters = clu[1],
                    clusters_pvalues = clu[2], participants = "asd")

    clu = cluster_permutation_test_2sample(score_asd, score_td, connectivity, tfce = tfce)
    save_results(level, liste_cond, frequency, liste_ASD + liste_TD, tfce, parent_folder, method, 
                score = score, Xasd=score_asd, Xtd=score_td, T_obs = clu[0], clusters = clu[1],
                clusters_pvalues = clu[2])

