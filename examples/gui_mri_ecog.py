from nilearn.image import crop_img
import numpy as np
from ecoggui import ElectrodeGUI

fname = 'T1_post_deface.nii.gz'
niimg = crop_img(fname)  # to automatically zoom on useful voxels

# We know we have a 4x4 grid of ecog channels, separated by 10 mm.
xy = np.meshgrid(np.linspace(0, 30, 4), np.linspace(0, 40, 5))
xy = np.transpose([ii.ravel() for ii in xy])

# Click on the grid to select the electrode and press spacebar to add it.
gui = ElectrodeGUI(niimg=niimg, xy=xy)

# Show electrode manually identified positions
print(gui.ch_user[['x', 'y', 'z']])

# Show electrodes predictions
print(gui.ch_pred[['x', 'y', 'z']])
