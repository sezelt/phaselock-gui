import py4DSTEM
from PyQt5.QtWidgets import QFileDialog
import h5py
import numpy as np
import os
import ncempy.io as ncemio
from PIL import Image


def load_data_auto(self):
    filename = self.show_file_dialog()
    self.load_file(filename)


def load_file(self, filepath):
    print(f"Loading file {filepath}")

    ncempy_types = ["ser", "mrc", "emd", "dm3", "dm4"]
    pil_types = ["png", "tif", "tiff"]

    extension = os.path.splitext(filepath)[1][1:]

    if extension in ncempy_types:
        self.image = ncemio.read(filepath)["data"]
    elif extension in pil_types:
        self.image = np.array(Image.open(filepath)).astype(np.float32).sum(axis=-1)
    else:
        raise ValueError(f"Unrecognized filetype {extension}")

    # move the selector to the center of the image
    self.virtual_detector_roi.setPos(self.image.shape[0]//2, self.image.shape[1]//2)

    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)

    self.setWindowTitle(filepath)


def show_file_dialog(self):
    filename = QFileDialog.getOpenFileName(
        self,
        "Open Image Data",
        "",
        "(S)TEM Data (*.dm3 *.dm4 *.emd *.ser *.mrc *.png *.tif *.tiff);;Any file (*)",
    )
    if filename is not None and len(filename[0]) > 0:
        return filename[0]
    else:
        print("File was invalid, or something?")
        raise ValueError("Could not read file")
