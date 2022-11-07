"""
===========
morphing file
===========
Script morphing functions for group analyses (called by analyses_groupes).
"""

import mne
import os
import numpy as np
import matplotlib.pyplot as plt
from traitement_sources import divide_stc, get_src_from_inverse_op
from mne.datasets import fetch_fsaverage
from mayavi import mlab

def compute_morph_surface(subjects_dir, inverse_op, subject_to):
    """Morph a subject surface source estimate to a common source space

    Parameters
    ----------
    stc_surf : SourceEstimate object
        surface source estimate
    subjects_dir : str
        path to subjects MRI directory
    subject_to : str
        Name of the subject we are morphing to
    path_to_figures : str
        path to the folder to save the figures
    plot : bool
        if True, save a figure in the path_to_figures folder
    
    Return
    ----------
    stc_surf : SourceEstimate object
        Morphed source estimate
    src_surf : SourceSpaces object
        Source space we morphed to
    """
    src = get_src_from_inverse_op(inverse_op, option = 'cortex')

    fetch_fsaverage(subjects_dir)  # ensure fsaverage src exists

    #fetch destination source spaces
    folder = os.path.join(subjects_dir, subject_to, 'bem')
    for f_name in os.listdir(folder):
        if 'ico-5-src' in f_name:
            src_fsaverage_surf = os.path.join(folder, f_name)
    
    # Read the source space we are morphing to
    src_surf = mne.read_source_spaces(src_fsaverage_surf)
    fsave_vertices = [s['vertno'] for s in src_surf]
    
    morph = mne.compute_source_morph(src, subjects_dir = subjects_dir,
                                    spacing = fsave_vertices, smooth=20)

    return morph, src_surf

def morph_surface_source_space(morph, src, stc, subjects_dir, inverse_op, 
                                subject_to, path_to_figures, 
                                cond = '', plot = False):
    """Morph a subject surface source estimate to a common source space

    Parameters
    ----------
    stc_surf : SourceEstimate object
        surface source estimate
    subjects_dir : str
        path to subjects MRI directory
    subject_to : str
        Name of the subject we are morphing to
    path_to_figures : str
        path to the folder to save the figures
    plot : bool
        if True, save a figure in the path_to_figures folder
    
    Return
    ----------
    stc_surf : SourceEstimate object
        Morphed source estimate
    src_surf : SourceSpaces object
        Source space we morphed to
    """
    #surface sources
    stc_surf = divide_stc(stc, option = 'cortex')

    #created morphed stc
    stc_surf_morphed = morph.apply(stc_surf)
    
    if plot:
        #before morphing
        brain = stc_surf.plot(subject = stc_surf.subject, hemi = 'both')
        title = 'Before_morphing_surface_psd'
        brain.save_image(filename = path_to_figures + '/' + title +'_' + cond + '.png')
        mlab.close()
        #after morphing
        brain = stc_surf_morphed.plot(subject = stc_surf_morphed.subject, hemi = 'both')
        title = 'Morphing_surface_psd_to_'+ subject_to
        brain.save_image(filename = path_to_figures + '/' + title +'_' + cond + '.png')
        mlab.close()

    return stc_surf_morphed

def compute_morph_volume(subjects_dir, inverse_op, subject_to):
    """Morph a subject volume source estimate to a common source space
    Morph separately the 2 cerebellar hemispheres, then reunite them

    Parameters
    ----------
    stc_vol : SourceEstimate object
        volume source estimate
    subjects_dir : str
        path to subjects MRI directory
    subject_to : str
        Name of the subject we are morphing to
    path_to_figures : str
        path to the folder to save the figures
    plot : bool
        if True, save a figure in the path_to_figures folder
    
    Return
    ----------
    stc_vol : VolSourceEstimate object
        Morphed volume source estimate
    src_vol : SourceSpaces object
        Source space we morphed to
    """
    #fetch destination source spaces
    fetch_fsaverage(subjects_dir)  # ensure fsaverage src exists
    folder = os.path.join(subjects_dir, subject_to, 'bem')
    for f_name in os.listdir(folder):
        if 'vol-5-src' in f_name:
            src_fsaverage_vol = os.path.join(folder, f_name)

    src_vol = mne.read_source_spaces(src_fsaverage_vol) #target source space

    cereb_used1, cereb_used2 = get_src_from_inverse_op(inverse_op, option = 'cereb_both_hemi') #source source space

    #compute the 2 morph (1 for each hemisphere)
    morph = mne.compute_source_morph(src = cereb_used1, 
                                    subject_to = subject_to, 
                                    subjects_dir=subjects_dir,
                                    niter_affine=[10, 10, 5], niter_sdr=[10, 10, 5],  # just for speed
                                    src_to=src_vol, verbose=True)
    morph2 = mne.compute_source_morph(src = cereb_used2, 
                                    subject_to = subject_to, 
                                    subjects_dir=subjects_dir,
                                    niter_affine=[10, 10, 5], niter_sdr=[10, 10, 5],  # just for speed
                                    src_to=src_vol, verbose=True)

    return [morph, morph2], src_vol


def morph_volume_source_space(morph, src_vol, stc, subjects_dir, 
                                inverse_op, subject_to, path_to_figures,
                                cond = '', plot = False):
    """Morph a subject volume source estimate to a common source space
    Morph separately the 2 cerebellar hemispheres, then reunite them

    Parameters
    ----------
    stc_vol : SourceEstimate object
        volume source estimate
    subjects_dir : str
        path to subjects MRI directory
    subject_to : str
        Name of the subject we are morphing to
    path_to_figures : str
        path to the folder to save the figures
    plot : bool
        if True, save a figure in the path_to_figures folder
    
    Return
    ----------
    stc_vol : VolSourceEstimate object
        Morphed volume source estimate
    src_vol : SourceSpaces object
        Source space we morphed to
    """
    [morph1, morph2] = morph
    #extract the 2 volume src    
    cereb_used1, cereb_used2 = get_src_from_inverse_op(inverse_op, option = 'cereb_both_hemi')

    #extract the 2 volume stc
    stc_volume1, stc_volume2 = divide_stc(stc, option = 'cereb_both_hemi')
    
    stc_volume1_morphed = morph1.apply(stc_volume1)
    stc_volume2_morphed = morph2.apply(stc_volume2)
    
    #reunite the two estimates
    new_stc = stc_volume1_morphed.__add__(stc_volume2_morphed)

    return new_stc

