"""
===========
traitement_sources file
===========
Script for source reconstruction functions (called by analyses_individuelles)
"""

import os
import mne
from mayavi import mlab
from mne import compute_covariance


def creation_source_space(source_space, subjects_dir_mri, 
                            subject_mri, bem, data_path_to_figures, 
                            plot = False):
    """Generates (or open if already exists) a mixed source space

        Parameters
        ----------
        source_space : str
            path to the mixed source space FIF file 
        subjects_dir_mri : str
            Path to the directory containing the subject MRI data (or the template MRI data)
        subject_mri : str
             Name of the sub directory containing the subject MRI data
        plot : bool
            If True, plot the mixed source space inside the subject's head. Default is False.

        Returns
        ----------
        src : a mne.SourceSpaces object
            mixed source space containing a volume and a surface source space
    """
    if os.path.exists(source_space):
        src = mne.read_source_spaces(source_space)
    
    else:
        aseg_fname = subjects_dir_mri + '/'+ subject_mri + '/mri/aseg.mgz'

        # setup a cortical surface source space
        surf = mne.setup_source_space(subject_mri, 
                                        subjects_dir=subjects_dir_mri, 
                                        add_dist=False)

        # setup a volume source space of the cerebellum cortex
        volume_label = ['Left-Cerebellum-Cortex','Right-Cerebellum-Cortex'] 
        cereb = mne.setup_volume_source_space(subject_mri, #sphere=sphere,
                                                bem = bem,
                                                volume_label = volume_label,
                                                subjects_dir = subjects_dir_mri,
                                                mri = aseg_fname)

        # Combine the source spaces and save them
        src = surf + cereb
        surf.save(source_space+"surf.fif")
        cereb.save(source_space+"vol.fif")
        src.save(source_space)

    if plot:
        fig = mne.viz.plot_alignment(subject = subject_mri, 
                                    subjects_dir = subjects_dir_mri,
                                    surfaces = ['head-dense','white'], 
                                    coord_frame = 'head',
                                    src=src)
        mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75,
                            distance=0.30, focalpoint=(-0.03, -0.01, 0.03))
        mlab.savefig(data_path_to_figures + 'source_space.png', figure = fig)
        mlab.close()

    return src

def boundary_element_model(epochs, bem_file, bem_surfaces, 
                            subject_mri, subjects_dir_mri, 
                            data_path_to_figures, plot = False):
    """Compute (or open if already exists) forward solution

        Parameters
        ----------
        epochs : object of class mne.Epochs
            EEG epoched data
        fwd_file : str
            Path to the forward solution FIF file
        bem_file : str
            Path to the BEM element FIF file
        subject_mri : str
             Name of the sub directory containing the subject MRI data
        subjects_dir_mri : str
            Path to the directory containing the subject MRI data (or the template MRI data)  
        plot : bool
            If True, plot BEM contours on anatomical slices. Default is False.

        Return
        ----------
        bem : instance of ConductorModel
            The BEM model
    """
    if os.path.exists(bem_file):
        bem = mne.read_bem_solution(bem_file)
    else:
        conductivity = (0.3, 0.006, 0.3)  # for three layers
        model = mne.make_bem_model(subject = subject_mri, ico=4,
                                    conductivity = conductivity,
                                    subjects_dir = subjects_dir_mri)                      
        bem = mne.make_bem_solution(model)

        mne.write_bem_surfaces(bem_surfaces, model)
        mne.write_bem_solution(bem_file, bem)

    if plot:
        fig = mne.viz.plot_bem(subject = subject_mri, subjects_dir = subjects_dir_mri,
                                brain_surfaces = 'white', orientation = 'coronal',
                                show = False)
        fig.savefig(data_path_to_figures + 'bem.png')
        
    return bem

def forward_solution(epochs, fwd_file, subject_mri, 
                        subjects_dir_mri, bem, src, trans):
    """Compute (or open if already exists) forward solution

        Parameters
        ----------
        epochs : object of class mne.Epochs
            EEG epoched data
        fwd_file : str
            Path to the forward solution FIF file
        subject_mri : str
             Name of the sub directory containing the subject MRI data
        subjects_dir_mri : str
            Path to the directory containing the subject MRI data (or the template MRI data)  
        src : a mne.SourceSpaces object
            mixed source space containing a volume and a surface source space
        trans : object of class mne.Transform
            transformation matrix

        Return
        ----------
        fwd : instance of Forward
            The Forward solution
    """        
    if os.path.exists(fwd_file):
        fwd = mne.read_forward_solution(fwd_file)
    else:
        fwd = mne.make_forward_solution(epochs.info, trans=trans, src = src, 
                                        bem = bem, meg=False, eeg=True, mindist=5.0)
        mne.write_forward_solution(fwd_file,fwd, overwrite = True)

    return fwd

def noise_covariance(f_name, epochs, path_to_figures, plot = False):
    """Compute (or open if already exists) noise covariance matrix of the epochs

        Parameters
        ----------
        f_name : str
            Path to the noise covariance matrix FIF file
        epochs : instance of mne.Epochs
            EEG epoched data
        plot : bool
            If True,plot the noise covariance matrix. Default is False.
        
        Return
        ----------
        noise_cov : instance of mne.Covariance
            the noise covariance matrix.
    """
    if os.path.exists(f_name):
        noise_cov = mne.read_cov(f_name)
    else:
        epochs_filtered = epochs.filter(l_freq = None, h_freq = 3.)

        noise_cov = compute_covariance(
                    epochs_filtered, method=['shrunk', 'empirical'], rank=None, verbose=True)
        mne.write_cov(f_name,noise_cov)
    
    if plot == True:
        fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, epochs.info, show = False)
        fig_spectra.savefig(fname = path_to_figures + 'noise_covariance_spectra.png')
        fig_cov.savefig(fname = path_to_figures + 'noise_covariance.png')

    return noise_cov

def inverse_operator(f_name, epochs = None, fwd = None, noise_cov = None):
    """Compute (or open if already exists) the inverse operator.

        Parameters
        ----------
        f_name : str
            Path to the inverse operator FIF file.
        epochs : instance of mne.Epochs
            EEG epoched data. Default to None when f_name already exists.
        fwd : instance of Forward
            The Forward solution. Default to None when f_name already exists.
        noise_cov : instance of mne.Covariance
            the noise covariance matrix. Default to None when f_name already exists.
        
        Return
        ----------
        inverse_op : instance of InverseOperator
            The inverse operator.
    """
    if os.path.exists(f_name):
        inverse_op = mne.minimum_norm.read_inverse_operator(f_name)
    else:
        inverse_op = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov,
                                                            loose='auto', depth=0.8)
        mne.minimum_norm.write_inverse_operator(f_name, inverse_op)
    
    return inverse_op

def inverse_solution_non_param(f_name, data, inverse_op, method, lambda2):
    """Compute the inverse solution using MNE or sLORETA method.

        Parameters
        ----------
        f_name : str
            Path to the inverse solution FIF file.
        data : instance of mne.Epochs, or mne.Evoked
            EEG data
            Default to None when f_name already exists.
        inverse_op : instance of InverseOperator
            The inverse operator.
            Default to None when f_name already exists.
        lambda2 : number.
            Parameter for inverse solution. Default to None when f_name already exists.
        method : str
            Method to resolve the inverse solution. 
            Can be 'MNE', 'dSPM', 'sLORETA', 'eLORETA'
            Default to None when f_name already exists.
        
        Return
        ----------
        stcs : MixedSourceEstimate
            The source estimates. 
    """
    if os.path.exists(f_name):
        return mne.read_source_estimate(f_name)

    if isinstance(data,mne.epochs.EpochsFIF):
            stcs = mne.minimum_norm.apply_inverse_epochs(data, inverse_op, 
                                                        lambda2, method,
                                                        pick_ori=None,
                                                        return_generator = True) 
    elif isinstance(data,mne.evoked.EvokedArray):
        stcs = mne.minimum_norm.apply_inverse(data, inverse_op, lambda2, 
                                                method, pick_ori=None)
    
    stcs.save(f_name) 

    return stcs


def divide_stc(stc, option = 'all'):
    """Divide a mixed source estimates object into surface source estimates and volume source estimates

        Parameters
        ----------
        stcs : MixedSourceEstimate
            The source estimates .
        subject_mri : str
             Name of the sub directory containing the subject MRI data
        option : str
            Name of the structure to extract STC from
        
        Return
        ----------
        stc_surface,  : SourceEstimate object
            The surface source estimate.
        stc_volume : VolumeSourceEstimate object
            The volume source estimate 
    """
    nb_sources_cortex = stc.vertices[0].shape[0] + stc.vertices[1].shape[0]
    nb_sources_cereb1 = nb_sources_cortex + stc.vertices[2].shape[0] 
    stc_surface = mne.SourceEstimate(stc.data[0:nb_sources_cortex], stc.vertices[0:2], 
                                tmin=stc.tmin, tstep=stc.tstep,
                                subject = stc.subject)

    #get stc of volume sources
    stc_volume = mne.VolSourceEstimate(stc.data[nb_sources_cortex:], stc.vertices[2:], 
                                    tmin=stc.tmin, tstep=stc.tstep,
                                    subject = stc.subject)
    
    if option == 'all':
        return stc_surface, stc_volume
    elif option == 'cortex':
        return stc_surface
    elif option == 'cereb':
        return stc_volume

    #get stc of volume sources, separating lh and rh
    stc_volume1 = mne.VolSourceEstimate(stc.data[nb_sources_cortex:nb_sources_cereb1], 
                                        [stc.vertices[2]], 
                                        tmin=stc.tmin, tstep=stc.tstep,
                                        subject = stc.subject)
    stc_volume2 = mne.VolSourceEstimate(stc.data[nb_sources_cereb1:], [stc.vertices[3]], 
                                        tmin=stc.tmin, tstep=stc.tstep,
                                        subject = stc.subject)
    if option == 'cereb_both_hemi':
        return stc_volume1, stc_volume2
    elif option == 'cereb_lh':
        return stc_volume1
    elif option == 'cereb_rh':
        return stc_volume2


def get_src_from_inverse_op(inverse_op, option = 'all'):
    """Extract source spaces from inverse operator

        Parameters
        -----------
        inverse_op : instance of InverseOperator
            The inverse operator.
        option : 'str'
            source space to extract
        
        Return :
        object mne.SourceSpaces, or tuple of mne.SourceSpaces
    """
    cortex = mne.SourceSpaces(inverse_op['src'][:2])
    cereb = mne.SourceSpaces(inverse_op['src'][2:])
    cereb_used_lh = mne.SourceSpaces(inverse_op['src'][2:3])
    cereb_used_rh = mne.SourceSpaces(inverse_op['src'][3:])

    if option == 'all':
        return cortex, cereb_used_lh, cereb_used_rh
    elif option == 'cortex':
        return cortex
    elif option == 'cereb':
        return cereb
    elif option == 'cereb_both_hemi':
        return cereb_used_lh, cereb_used_rh
    elif option == 'cereb_lh':
        return cereb_used_lh
    elif option == 'cereb_rh':
        return cereb_used_rh
    
