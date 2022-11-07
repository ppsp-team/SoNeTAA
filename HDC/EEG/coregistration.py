"""
===========
coregistration file
===========
Script for coregistration functions (called by analyses_individuelles)
"""

import mne
import os
import numpy as np
from mne.coreg import get_mni_fiducials
from mne.io import write_fiducials
from mne.io.constants import FIFF
from mne.gui._file_traits import DigSource
from mne.gui._fiducials_gui import MRIHeadWithFiducialsModel
from mne.gui._coreg_gui import CoregModel


def coregistration_automated(files):
    """Generates (or open if already exists) a transformation matrix to align MRI and EEG
        Simplified version of automated coregistation
        To do: check if works for every subject 

        Parameters
        ----------
        files: File_paths object
            Object containing all the file paths for one subject

        Returns
        ----------
        trans : object of class mne.Transform
            transformation matrix
    """
    if not os.path.exists(files.matrix_auto):
        print('---- Automated coregistration begins ----')
        #estimate MRI fiducials if don't exist
        if not os.path.isfile(files.mri_fiducials):
            fids_mri = get_mni_fiducials(files.subject_mri, files.subjects_dir_mri)
            write_fiducials(files.mri_fiducials, fids_mri, coord_frame=FIFF.FIFFV_COORD_MRI)

        # set up HSP DigSource
        hsp = DigSource()
        hsp.file = files.f_epochs
        # Set up subject MRI source space with fiducials
        mri = MRIHeadWithFiducialsModel(subjects_dir = files.subjects_dir_mri, subject = files.subject_mri)
        # Set up coreg model
        model = CoregModel(mri=mri, hsp=hsp)

        #run coregistration
        model.reset()
        model.fit_fiducials(n_scale_params = 1)
        model.omit_hsp_points(distance=5. / 1000)        
        errs_icp = model._get_point_distance()
        print('Median distance from digitized points to head surface is %.3f mm'
            % np.median(errs_icp * 1000))
        
        #save rescaled mri
        job = model.get_scaling_job(subject_to = 'new-sample', skip_fiducials = False)
        scaling_factor = job[3][0]
        if round(scaling_factor,2) !=1:
            if not os.path.exists(files.subjects_dir_mri + "/" + job[2]):
                mne.scale_mri(subject_from = files.subject_mri, subject_to = job[2], 
                        scale = scaling_factor, subjects_dir = files.subjects_dir_mri,
                        labels = True, annot = True, overwrite = False)
            print('Subject ' + files.subject_mri + ' rescaled with factor '+ str(scaling_factor),
                    ', new MRI saved to ' + job[2])
            files.change_mri_subject(subject_to = job[2])

        #save trans matrix
        model.save_trans(fname = files.matrix_auto)
        print('Transformation matrix saved')

    #read the matrix
    trans = mne.read_trans(files.matrix_auto)

    #average distance
    """
    liste_dist = mne.dig_mri_distances(files.epochs.info, trans, 
                                        subject = files.subject_mri, 
                                        subjects_dir=subjects_dir_mri)
    print('average distance between head shape points and the scalp surface', 
                statistics.mean(liste_dist))"""

    return trans

def best_initial_coreg_fit(model):
    """Get best fit from initial coreg fit for outlier detection

    Parameters
    ----------
    model : CoregModel object

    Return
    ----------
    it_fid : int
        number of icp iterations minimizing the error
    """
    it_fids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30]
    errs_it = []
    for iterate in it_fids:
        model.reset()
        model.fit_fiducials()
        model.icp_iterations = int(iterate)
        model.fit_icp()
        model.omit_hsp_points(distance=5. / 1000)  # Distance is in meters
        errs_temp = model._get_point_distance()
        if len(errs_temp) > 50:
            errs_it.append(np.median(errs_temp))
        else:
            errs_it.append(999.)

    it_fid = it_fids[np.argmin(errs_it)]
    return it_fid

def best_final_coreg_fit(model, it_fid):
    """Test final coreg fit
    
    Parameters
    ----------
    model : CoregModel object
    it_fid : int
        number of icp iterations minimizing the error for initial coreg fit

    Return
    ----------
    it_fid : int
        number of icp iterations minimizing the error
    """
    wts = [5., 10., 15.] #test several nasion weights
    it_icp = [10, 20, 30, 40, 50] #test several numbers of icp iterations
    err_icp = np.ones([len(wts), len(it_icp)])
    pts_icp = np.ones([len(wts), len(it_icp)], dtype='int64')
    for j, wt in enumerate(wts):
        for k, iterate in enumerate(it_icp):
            # REpeat best-fitting steps from above
            model.reset
            model.fit_fiducials()
            model.icp_iterations = int(it_fid)
            model.fit_icp()
            model.omit_hsp_points(distance=5. / 1000)
            # Test new parms
            model.nasion_weight = wt
            model.icp_iterations = int(iterate)
            model.fit_icp()
            errs_temp = model._get_point_distance()
            if len(errs_temp) > 50:
                err_icp[j, k] = (np.median(errs_temp))
                pts_icp[j, k] = len(errs_temp)
            else:
                err_icp[j, k] = 999.
                pts_icp[j, k] = int(1)
            print(err_icp[j, k])

    idx_wt, idx_it = np.where(err_icp == np.min(err_icp))
    wt = wts[idx_wt[0]]
    iterate = it_icp[idx_it[0]]
    #errs_icp = np.min(err_icp)
    #num_pts_icp = pts_icp[idx_wt[0], idx_it[0]]
    
    return wt, iterate

