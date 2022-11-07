""" Visualization functions for one subject (called by analyses_individuelles)
"""

import mne
import matplotlib.pyplot as plt
from mayavi import mlab
import numpy as np
import os

def save_mayavi_plot(name, fig):
    if not os.path.exists(name+'.png'):
        mlab.savefig(name+'.png', figure=fig)

def visualization_epochs(files, epochs):
    """Function to visualize epoched data
        Plot activity of all EEG channels along epochs

        Parameters
        ----------
        epochs : object of class mne.Epochs
            EEG epoched data
    """
    fig1 = epochs.plot(block=True, show = False)
    fig1.savefig(files.data_path_to_figures + 'epochs.png')

    #visualize one channel
    fig2 = epochs.plot_image(1, cmap='interactive', sigma=1., vmin=-250, 
                                vmax=250, show = False)
    fig2[0].savefig(files.data_path_to_figures + 'epochs_one_channel.png')
    
    #overview of all channels
    fig3 = epochs.plot_image(combine='gfp', sigma=2., cmap="YlGnBu_r",
                                show = False)
    fig3[0].savefig(files.data_path_to_figures + 'epochs_all_channels.png')
       

def visualization_evoked(files, evoked):
    """Function to visualize evoked data

        Parameters
        ----------
        evoked : object mne.Evoked
            EEG epoched data
    """
    fig1 = evoked.plot(spatial_colors=True, gfp=True, picks='eeg', show = False)
    fig1.savefig(files.data_path_to_figures + 'evoked.png')

    fig2 = evoked.plot_topomap(time_unit='s', show = False)
    fig2.savefig(files.data_path_to_figures + 'evoked_topomap.png')

def check_alignment(info, trans, subject_mri, 
                    subjects_dir_mri, data_path_to_figures, 
                    title = '', src = None, bem = None):
    """Print distances between head shape points and the scalp surface
        Plot alignment between source space, EEG electrodes and MRI head shape

        Parameters
        ----------
        info : instance of mne.Info
            information about EEG epoched data
        trans : object of class mne.Transform
            transformation matrix
        src : a mne.SourceSpaces object
            mixed source space containing a volume and a surface source space
        subject_mri : str
             Name of the sub directory containing the subject MRI data
        subjects_dir_mri : str
            Path to the directory containing the subject MRI data (or the template MRI data)   
    """
    fig = mne.viz.plot_alignment(info, trans = trans, subject = subject_mri,
                             subjects_dir = subjects_dir_mri, surfaces = ['head-dense','white'],
                             show_axes = True, dig = True, meg = None, src = src,
                             coord_frame = 'head', eeg = ['original'], bem = bem)
    mne.viz.set_3d_view(fig, 45, 90, distance=0.6, focalpoint=(0., 0., 0.))
    mlab.savefig(data_path_to_figures + title + '.png', figure = fig)
    mlab.close()