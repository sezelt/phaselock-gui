import py4DSTEM
from PyQt5.QtWidgets import QFileDialog
import h5py
import numpy as np
import os
import ncempy.io as ncemio


def load_data_auto(self):
    filename = self.show_file_dialog()
    self.load_file(filename)


def load_file(self, filepath):
    print(f"Loading file {filepath}")

    self.image = ncemio.read(filepath)['data']

    self.update_diffraction_space_view(reset=True)
    self.update_real_space_view(reset=True)

    self.setWindowTitle(filepath)


def show_file_dialog(self):
    filename = QFileDialog.getOpenFileName(
        self,
        "Open Image Data",
        "",
        "(S)TEM Data (*.dm3 *.dm4 *.emd *.ser *.mrc);;Any file (*)",
    )
    if filename is not None and len(filename[0]) > 0:
        return filename[0]
    else:
        print("File was invalid, or something?")
        raise ValueError("Could not read file")

