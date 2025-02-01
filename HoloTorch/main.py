"""
File:           main.py
Author:         Avion Lowery
Date:           1/31/25
Email:          alowery1@umbc.edu or loweryavion@gmail.com
Description:    This will create a hologram pattern to be sent to the matlab to convert to a proper
                CGH.
"""

import os, sys, torch, glob, urllib, zipfile, pathlib
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display_functions import display

import holotorch.CGH_Datasets.Single_Image_Dataset as Single_Image_Dataset
import holotorch.CGH_Datasets.Factory_Dataset as Factory_Dataset
from holotorch.CGH_Datatypes.IntensityField import IntensityField
from holotorch.ComponentWrapper.PARAM_DATASET import PARAM_DATASET
from holotorch.utils.Enumerators import *

import holotorch
from holotorch.utils.units import * # E.g. to get nm, um, mm etc.

from holotorch.CGH_Datatypes.ElectricField import ElectricField
# ElectricFields are 6D objects:
# B x T x P x C x H x W
# BATCH x TIME x PUPIL (lightfields) x Channel (Wavelength) x Height x Width

if __name__ == '__main__':
    #  Basics
    print("10mm : ", 10 * mm, "in m")
    print(" 1nm : ", 1 * nm, "in m")

    N = 1024
    field_data = torch.zeros(1, 1, 1, 1, N, N) + 0j  # 0j to make it complex

    # Set ones to the field
    field_data[..., N // 4: 3 * N // 4, N // 4: 3 * N // 4] = 1

    # Cast into our Holotorch Datatype
    field_input = ElectricField(
        data=field_data,
        wavelengths=532 * nm,
        spacing=8 * um,
    )

    # Supposed to output a plot
    field_input.visualize(flag_axis=True)

    print(field_input.spacing)
    print(field_input.shape)
    downsampled_field = field_input.rescale(0.25)
    print(downsampled_field.spacing)
    print(downsampled_field.shape)

    # Supposed to output a plot
    plt.figure(figsize=(10, 10))
    downsampled_field.visualize(flag_axis=True)
    plt.tight_layout()
    plt.show()