"""
===========
run_group file
===========
Script for calling group level analyses.
"""

import analyses_groupes as gr
from config import path_to_eeg_group 
from parameters import (method, coreg, redo, plot, 
                        lambda2, level, subject_to, path_to_results, tfce,
                        path_results_source_rec, n_epochs)

#fetch asd and td
list_asd = ["path/to/subject_1", "path/to/subject_n"]
list_td = ["path/to/subject_1", "path/to/subject_n"]
list_subjects = list_asd + list_td

#### LAUNCH SOURCE RECONSTRUCTION
gr.source_reconstruction_group(path_to_eeg_group, path_results_source_rec, list_subjects, 
                                method, lambda2, coreg, plot, redo)


#### SENSOR-LEVEL ANALYSES
# Save the Power Spectral Densities for each subject and condition (sensor level)
gr.power_analysis_sensor_groups_psds(list_td, list_asd, path_to_eeg_group, 
                                    liste_cond = ["HDC", "RS"], frequency = "theta")
psd_file = "path/to/psd_data_cortex.npz"
gr.power_analysis_statistics(path_results_source_rec, psd_file, path_to_eeg_group, list_subjects,
                                list_td, list_asd, method, subject_to, level = "sensor", liste_cond=["HDC", "RS"],
                                compare = "all", frequency = "theta")


#### SOURCE-LEVEL ANALYSES

# Save the Power Spectral Densities for each subject and condition (source level)
gr.power_analysis_source_groups_psds(path_results_source_rec, list_asd, list_td, path_to_eeg_group, 
                                    method, lambda2, subject_to, area = "cortex", liste_cond=["HDC", "RS"],
                                    frequency = "theta")
gr.power_analysis_source_groups_psds(path_results_source_rec, list_asd, list_td, path_to_eeg_group, 
                                    method, lambda2, subject_to, area = "cerebellum", liste_cond=["HDC", "RS"],
                                    frequency = "theta")

# Perform statistical tests
psd_file = "path/to/psd_data_cortex.npz"
gr.power_analysis_statistics(path_results_source_rec, psd_file, path_to_eeg_group, list_subjects,
                                list_td, list_asd, method, subject_to, level = "source", liste_cond=["HDC", "RS"],
                                compare = "all", frequency = "theta")
psd_file = "path/to/psd_data_cerebellum.npz"
gr.power_analysis_statistics(path_results_source_rec, psd_file, path_to_eeg_group, list_subjects,
                            list_td, list_asd, method, subject_to, level = "source", liste_cond=["HDC", "RS"],
                            compare = "all", frequency = "theta")


