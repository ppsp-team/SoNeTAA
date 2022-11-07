"""
===========
Group file for connectivity
===========
Script for group level analyses.
Connectivity computation between ROIs and statistical tests
"""

import mne
import os
import csv
import numpy as np
from tools import (add_to_array, zscore, save_results, cohen_d, 
                    check_list_cond, get_epochs_in_condition, 
                    sujets_exclusion_psds)
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.graphics.regressionplots import *
from open_files import recover_epochs_sonetaa, recover_inverse_op
from connectivity import connectivity_one_epoch
from scipy.stats import ttest_ind, ttest_1samp
from parameters import (subjects_dir)

#global path_to_eeg_group


def get_labels():
    """
    Get cortical labels from subject sample (subject used for source reconstruction)
    """
    labels_surf = mne.read_labels_from_annot(subject = "new-sample", parc = 'aparc', 
                                        subjects_dir = subjects_dir) #tous les labels

    frontal_rh = ["frontalpole-rh", 'rostralmiddlefrontal-rh', 'parsorbitalis-rh', 
                    'medialorbitofrontal-rh', 'lateralorbitofrontal-rh']
    frontal_lh = ["frontalpole-lh",'rostralmiddlefrontal-lh', 'parsorbitalis-lh', 
        'medialorbitofrontal-lh', 'lateralorbitofrontal-lh']
    sensorimotor_rh = ['postcentral-rh', 'precentral-rh']
    sensorimotor_lh = ['postcentral-lh', 'precentral-lh']
    rtpj = ['bankssts-rh', "inferiorparietal-rh"]
    ltpj = ['bankssts-lh', "inferiorparietal-lh"]

    label_frontal_rh = [l for l in labels_surf if l.name in frontal_rh]
    label_frontal_lh = [l for l in labels_surf if l.name in frontal_lh]
    label_sm_rh = [l for l in labels_surf if l.name in sensorimotor_rh]
    label_sm_lh = [l for l in labels_surf if l.name in sensorimotor_lh]
    label_tpj_rh = [l for l in labels_surf if l.name in rtpj]
    label_tpj_lh = [l for l in labels_surf if l.name in ltpj]
    all = label_frontal_rh + label_frontal_lh + label_sm_rh + label_sm_lh + label_tpj_rh + label_tpj_lh
    
    return all

def save_conn_csv(path_to_figures, list_asd, list_td, path_results_source_rec, method):
    #exclusions
    indices_td, indices_asd = sujets_exclusion_psds(path_to_figures)
    indices_asd, indices_td = np.asarray(indices_asd), np.asarray(indices_td)

    #write csv
    f = open(path_to_figures+"/connectivity.csv", "w", newline = "")
    writer = csv.writer(f)
    path_or_name_eeg = 'REC-190515-A'
    inverse_op = recover_inverse_op(path_or_name_eeg, 'RS', method, path_results_source_rec)
    src = inverse_op['src']
    labels_cereb = mne.get_volume_labels_from_src(src, "new-sample", subjects_dir)   
    labels_names = ["frontal_rh", "frontal_lh", "sensorimotor_rh", "sensorimotor_lh", "rTPJ", "lTPJ"]
    labels_names += [l.name for l in labels_cereb]
    print(labels_names)
    labs = []
    for x in range(len(labels_names)):
        for j in range(len(labels_names)):
            if x != j:
                labs.append(labels_names[x] + "_"+ labels_names[j])

    writer.writerow(["id"] + ["groupe"] +labs)


def connectivity_computation(path_results_subjects, liste_ASD, liste_TD, path_to_figures, 
                            method, lambda2, subjects_dir, condition, frequency):
    """
    Computes connectivity for a condition (HDC, RS eo or RS ec)
    Save raw connectivity values in npz file
    """
    #remove exluded subjects
    indices_td, indices_asd = sujets_exclusion_psds(path_to_figures)
    indices_asd.sort(reverse = True)
    indices_td.sort(reverse = True)
    for k in indices_asd:
        liste_ASD.pop(k)    
    for k in indices_td:
        liste_TD.pop(k)

    labels = get_labels()
    array_asd, array_td = None, None
    array_asd_std, array_td_std = None, None

    for i, liste in enumerate([liste_ASD, liste_TD]):
        for (path_or_name_eeg, path_to_mri) in liste:
            #recover inverse operator
            if condition == "HDC":
                inverse_op = recover_inverse_op(path_or_name_eeg, 'HDC', method, path_results_subjects)
            else:
                inverse_op = recover_inverse_op(path_or_name_eeg, 'RS', method, path_results_subjects)
            #recover epochs 
            epochs_RS = recover_epochs_sonetaa(path_or_name_eeg, 'RS', path_to_mri)
            epochs_HDC = recover_epochs_sonetaa(path_or_name_eeg, 'HDC', path_to_mri)
            src = inverse_op['src']
            sfreq = epochs_RS.info['sfreq']
            epochs_condition = get_epochs_in_condition(condition, epochs_HDC, epochs_RS)
            #get stc for each epoch
            stcs = mne.minimum_norm.apply_inverse_epochs(epochs_condition, inverse_op, 
                                                        lambda2, method,return_generator=True)
            #get label stc for each epoch 
            label_ts = mne.extract_label_time_course(stcs, labels, src, return_generator = True)
            #calculate connectivity between labels for each epoch
            conn_epochs = None
            for (_, stc_label_epoch) in enumerate(label_ts): #n labels x n times
                #mean of stc for each region
                stc_frontal_rh = np.mean(stc_label_epoch[:5, :], axis = 0)
                stc_frontal_lh = np.mean(stc_label_epoch[5:10, :], axis = 0)
                stc_sm_rh = np.mean(stc_label_epoch[10:12, :], axis = 0)
                stc_sm_lh = np.mean(stc_label_epoch[12:14, :], axis = 0)
                stc_tpj_rh = np.mean(stc_label_epoch[14:16, :], axis = 0)
                stc_tpj_lh = np.mean(stc_label_epoch[16:18, :], axis = 0)
                cereb = stc_label_epoch[-2:, :]
                
                stc_label_epoch2 = np.stack([stc_frontal_rh, stc_frontal_lh, stc_sm_rh, 
                                            stc_sm_lh, stc_tpj_rh, stc_tpj_lh, cereb])
                conn, _ = connectivity_one_epoch(stc_label_epoch2, stc_label_epoch2, src, sfreq, 
                                                subjects_dir, frequency) 
                conn = np.expand_dims(conn, axis = 0)
                conn_epochs = add_to_array(conn_epochs, conn)

            mean_conn1 = np.mean(conn_epochs, axis = 0)
            mean_conn1_flat = np.reshape(mean_conn1, (1, mean_conn1.shape[0]*mean_conn1.shape[1]))
            std = np.std(conn_epochs, axis = 0)
            std_flat = np.reshape(std, (1, std.shape[0]*std.shape[1]))

            #add zscore to the list of subject zscore for ASD or TD condition
            if i == 0:
                array_asd = add_to_array(array_asd, mean_conn1_flat)
                array_asd_std = add_to_array(array_asd_std, std_flat)
            else:
                array_td = add_to_array(array_td, mean_conn1_flat)
                array_td_std = add_to_array(array_td_std, std_flat)
    
    #save raw connectivity values
    save_results("Raw_connectivity", [condition, ""], frequency, liste_ASD + liste_TD,
                path_to_figures = path_to_figures, method = method, Xtd=array_td, 
                Xasd = array_asd, Std = array_td_std, Sasd = array_asd_std)


def connectivity_effect_size(path_results_subjects, folder_name, connectivity_file,
                            liste_ASD, liste_TD, path_to_figures, 
                            method, lambda2, subjects_dir, liste_cond,
                            compare = "all", score = "cohen"):
    """
        Computes effect sizes for connectivity measures (for intra and/or inter group comparisons)
        folder_name = folder where the raw connectivity values are stored
        liste_cond = the two conditions to compare
    """
    liste_cond, _ = check_list_cond(liste_cond)

    # Search for the right files
    folder_to_look = os.path.join(path_to_figures, folder_name)
    files = [f for f in os.listdir(folder_to_look) if "npz" in f and "Raw" in f]
    f_cond1 = [f for f in files if liste_cond[0] in f][0]
    f_cond2 = [f for f in files if liste_cond[1] in f][0]
    for f in [f_cond1, f_cond2]:
        npzfile = np.load(folder_to_look + "/" + f, allow_pickle = True)
        if "Xtd" in npzfile.files and "Xasd" in npzfile.files:
            Xtd, Xasd = npzfile["Xtd"], npzfile["Xasd"]
        if f == f_cond1:
            Std, Sasd = npzfile["Std"], npzfile["Sasd"]
            Xtd_cond1, Xasd_cond1 = Xtd, Xasd
        else:
            Xtd_cond2, Xasd_cond2 = Xtd, Xasd
    
    #cohen d
    score_td = cohen_d(Xtd_cond1, Xtd_cond2) 
    score_asd = cohen_d(Xasd_cond1, Xasd_cond2)    
    score_td_asd = cohen_d(Xtd_cond2, Xasd_cond2)
    
    score_td = np.reshape(score_td, (8,8))
    score_asd = np.reshape(score_asd, (8,8))
    score_td_asd = np.reshape(score_td_asd, (8,8))

    score_td[:6,:6], score_td[-2:,-2:] = 0, 0
    score_asd[:6,:6], score_asd[-2:,-2:] = 0, 0
    score_td_asd[:6,:6], score_td_asd[-2:,-2:] = 0, 0

    #save results
    save_results("connectivity_cohen", liste_cond, "theta", liste_TD,
                path_to_figures = folder_to_look, method = method, Xtd=Xtd, T_obs = score_td,
                score = score, participants="td")
    save_results("connectivity_cohen", liste_cond, "theta", liste_ASD,
                path_to_figures = folder_to_look, method = method, Xasd = Xasd, T_obs = score_asd,
                score = score, participants="asd")
    save_results("connectivity_cohen", liste_cond, "theta", liste_ASD + liste_TD,
                path_to_figures = folder_to_look, method = method, Xtd=Xtd, Xasd = Xasd, 
                T_obs = score_td_asd, score = score)
    

def connectivity_statistics(path_results_subjects, folder_name, connectivity_file,
                            liste_ASD, liste_TD, path_to_figures, 
                            method, lambda2, subjects_dir, liste_cond,
                            compare = "all", score = "zscore"):
    """
        Computes student t tests for connectivity measures (for intra and/or inter group comparisons)
        folder_name = folder where the raw connectivity values are stored
        liste_cond = the two conditions to compare
    """
    liste_cond, _ = check_list_cond(liste_cond)

    # Search for the right files
    folder_to_look = os.path.join(path_to_figures, folder_name)
    files = [f for f in os.listdir(folder_to_look) if "npz" in f and "Raw" in f]
    f_cond1 = [f for f in files if liste_cond[0] in f][0]
    f_cond2 = [f for f in files if liste_cond[1] in f][0]

    for f in [f_cond1, f_cond2]:
        npzfile = np.load(folder_to_look + "/" + f, allow_pickle = True)
        if "Xtd" in npzfile.files and "Xasd" in npzfile.files:
            Xtd, Xasd = npzfile["Xtd"], npzfile["Xasd"]
        if f == f_cond1:
            Std, Sasd = npzfile["Std"], npzfile["Sasd"]
            Xtd_cond1, Xasd_cond1 = Xtd, Xasd
        else:
            Xtd_cond2, Xasd_cond2 = Xtd, Xasd
    
    # Zscore subjects data
    array_td = zscore(Xtd_cond2, Xtd_cond1, Std)
    array_asd = zscore(Xasd_cond1, Xasd_cond2, Sasd)
    
    # TD group
    t_obs_td, pvalues_td = ttest_1samp(array_td, popmean = 0, axis = 0)
    # ASD group
    t_obs_asd, pvalues_asd = ttest_1samp(array_asd, popmean = 0, axis = 0)
    # ASD vs TD groups
    t_obs_asd_td, pvalues_asd_td = ttest_ind(array_asd, array_td, axis = 0)

    for t_obs, pvalues in [[t_obs_td, pvalues_td], [t_obs_asd, pvalues_asd], [t_obs_asd_td, pvalues_asd_td]]:
        #reshape t_obs into a 2d matrix (n sub x n labels x m labels) for visualization
        t_obs, pvalues = np.reshape(t_obs, (8,8)), np.reshape(pvalues, (8,8))
        #keep only connectivity between cereb and cortex
        t_obs[:6,:6], t_obs[-2:,-2:] = 0, 0
        # FDR correction for multiple tests
        _, goodpvalues = fdrcorrection(pvalues[:6, -2:].flatten())
        goodpvalues = np.reshape(goodpvalues, (6,2))
        pvalues[:6, -2:], pvalues[-2:, :6] = goodpvalues, goodpvalues

    #save results
    save_results("connectivity", liste_cond, "theta", liste_TD,
            path_to_figures = folder_to_look, method = method, 
            Xtd=array_td, T_obs = t_obs, pvalues = pvalues, score = score, participants="td")

    save_results("connectivity", liste_cond, "theta", liste_ASD,
            path_to_figures = folder_to_look, method = method, Xasd=array_asd, 
            T_obs = t_obs, pvalues = pvalues, score = score, participants="asd")

    save_results("connectivity", liste_cond, "theta", liste_ASD + liste_TD,
            path_to_figures = folder_to_look, method = method, Xtd=array_td, Xasd = array_asd, 
            T_obs = t_obs, pvalues = pvalues, score = score)

    
