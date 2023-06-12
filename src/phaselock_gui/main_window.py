from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QWidget,
    QMenu,
    QAction,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QPushButton,
    QScrollArea,
    QCheckBox,
    QLineEdit,
    QRadioButton,
    QButtonGroup,
    QDesktopWidget,
    QMessageBox,
    QActionGroup,
)
from PyQt5 import QtGui

import pyqtgraph as pg
import numpy as np

from functools import partial
from pathlib import Path

from phaselock_gui.utils import pg_point_roi


class DataViewer(QMainWindow):
    """
    The class is used by instantiating and then entering the main Qt loop with, e.g.:
        app = DataViewer(sys.argv)
        app.exec_()
    """

    LOG_SCALE_MIN_VALUE = 1

    from phaselock_gui.menu_actions import (
        load_data_auto,
        load_file,
        show_file_dialog,
    )

    from phaselock_gui.update_views import (
        update_diffraction_space_view,
        update_real_space_view,
    )

    def __init__(self, argv):
        super().__init__()
        # Define this as the QApplication object
        self.qtapp = QApplication.instance()
        if not self.qtapp:
            self.qtapp = QApplication(argv)

        self.setWindowTitle("Clever Title Goes Here")

        # icon = QtGui.QIcon(str(Path(__file__).parent.absolute() / "logo.png"))
        # self.setWindowIcon(icon)
        # self.qtapp.setWindowIcon(icon)

        self.setWindowTitle("Clever Title Goes Here")
        self.setAcceptDrops(True)

        self.image = None

        self.setup_menus()
        self.setup_views()

        self.resize(800, 400)

        self.show()

        # If a file was passed on the command line, open it
        if len(argv) > 1:
            self.load_file(argv[1])

    def setup_menus(self):
        self.menu_bar = self.menuBar()

        # File menu
        self.file_menu = QMenu("&File", self)
        self.menu_bar.addMenu(self.file_menu)

        self.load_auto_action = QAction("&Load Data...", self)
        self.load_auto_action.triggered.connect(self.load_data_auto)
        self.file_menu.addAction(self.load_auto_action)

        # Snapping menu
        self.snap_menu = QMenu("&Snapping",self)
        self.menu_bar.addMenu(self.snap_menu)
        snapping_group = QActionGroup(self)
        snapping_group.setExclusive(True)
        self.snapping_group = snapping_group

        no_snap = QAction("None")
        no_snap.setCheckable(True)
        snapping_group.addAction(no_snap)
        self.snap_menu.addAction(no_snap)
        no_snap.triggered.connect(
            partial(self.update_real_space_view, False)
        )

        CoM_snap = QAction("Center of Mass")
        CoM_snap.setCheckable(True)
        snapping_group.addAction(CoM_snap)
        self.snap_menu.addAction(CoM_snap)
        CoM_snap.triggered.connect(
            partial(self.update_real_space_view, False)
        )

        max_snap = QAction("Maximum")
        max_snap.setCheckable(True)
        snapping_group.addAction(max_snap)
        self.snap_menu.addAction(max_snap)
        max_snap.setChecked(True)
        max_snap.triggered.connect(
            partial(self.update_real_space_view, False)
        )

        # Scaling Menu
        self.scaling_menu = QMenu("&Scaling", self)
        self.menu_bar.addMenu(self.scaling_menu)

        # Diffraction scaling
        diff_scaling_group = QActionGroup(self)
        diff_scaling_group.setExclusive(True)
        self.diff_scaling_group = diff_scaling_group
        diff_menu_separator = QAction("Diffraction", self)
        diff_menu_separator.setDisabled(True)
        self.scaling_menu.addAction(diff_menu_separator)

        diff_scale_linear_action = QAction("Linear", self)
        diff_scale_linear_action.setCheckable(True)
        diff_scale_linear_action.setChecked(True)
        diff_scale_linear_action.triggered.connect(
            partial(self.update_diffraction_space_view, True)
        )
        diff_scaling_group.addAction(diff_scale_linear_action)
        self.scaling_menu.addAction(diff_scale_linear_action)

        diff_scale_log_action = QAction("Log", self)
        diff_scale_log_action.setCheckable(True)
        diff_scale_log_action.triggered.connect(
            partial(self.update_diffraction_space_view, True)
        )
        diff_scaling_group.addAction(diff_scale_log_action)
        self.scaling_menu.addAction(diff_scale_log_action)

        diff_scale_sqrt_action = QAction("Square Root", self)
        diff_scale_sqrt_action.setCheckable(True)
        diff_scale_sqrt_action.triggered.connect(
            partial(self.update_diffraction_space_view, True)
        )
        diff_scaling_group.addAction(diff_scale_sqrt_action)
        self.scaling_menu.addAction(diff_scale_sqrt_action)

        self.scaling_menu.addSeparator()

        # Real space scaling
        vimg_scaling_group = QActionGroup(self)
        vimg_scaling_group.setExclusive(True)
        self.vimg_scaling_group = vimg_scaling_group

        vimg_menu_separator = QAction("Virtual Image", self)
        vimg_menu_separator.setDisabled(True)
        self.scaling_menu.addAction(vimg_menu_separator)

        vimg_scale_linear_action = QAction("Linear", self)
        self.vimg_scale_linear_action = vimg_scale_linear_action  # Save this one!
        vimg_scale_linear_action.setCheckable(True)
        vimg_scale_linear_action.triggered.connect(
            partial(self.update_real_space_view, True)
        )
        vimg_scaling_group.addAction(vimg_scale_linear_action)
        self.scaling_menu.addAction(vimg_scale_linear_action)

        vimg_scale_log_action = QAction("Log", self)
        vimg_scale_log_action.setCheckable(True)
        vimg_scale_log_action.triggered.connect(
            partial(self.update_real_space_view, True)
        )
        vimg_scaling_group.addAction(vimg_scale_log_action)
        self.scaling_menu.addAction(vimg_scale_log_action)

        vimg_scale_sqrt_action = QAction("Square Root", self)
        vimg_scale_sqrt_action.setCheckable(True)
        vimg_scale_linear_action.setChecked(True)
        vimg_scale_sqrt_action.triggered.connect(
            partial(self.update_real_space_view, True)
        )
        vimg_scaling_group.addAction(vimg_scale_sqrt_action)
        self.scaling_menu.addAction(vimg_scale_sqrt_action)


    def setup_views(self):
        # Set up the diffraction space window.
        self.diffraction_space_widget = pg.ImageView()
        self.diffraction_space_widget.setImage(np.zeros((512, 512)))
        self.diffraction_space_widget.getImageItem().setOpts(axisOrder="row-major")
        self.diffraction_space_view_text = pg.TextItem(
            "Slice", (200, 200, 200), None, (0, 1)
        )
        self.diffraction_space_widget.addItem(self.diffraction_space_view_text)

        # Create virtual detector ROI selector
        # x0, y0 = 512, 512
        x0,y0 = 0,0
        xr, yr = 25, 25
        self.virtual_detector_roi = pg.CircleROI(
            [int(x0 - xr / 2), int(y0 - yr / 2)], [int(xr), int(yr)], pen=(3, 9)
        )
        self.diffraction_space_widget.getView().addItem(self.virtual_detector_roi)
        self.virtual_detector_roi.sigRegionChangeFinished.connect(
            self.update_real_space_view
        )

        # Name and return
        self.diffraction_space_widget.setWindowTitle("Reciprocal Space")

        # Set up the real space window.
        self.real_space_widget = pg.ImageView()
        self.real_space_widget.getImageItem().setOpts(axisOrder="row-major")
        self.real_space_widget.setImage(np.zeros((512, 512)))

        # Name and return
        self.real_space_widget.setWindowTitle("Real Space")

        self.diffraction_space_widget.setAcceptDrops(True)
        self.real_space_widget.setAcceptDrops(True)
        self.diffraction_space_widget.dragEnterEvent = self.dragEnterEvent
        self.real_space_widget.dragEnterEvent = self.dragEnterEvent
        self.diffraction_space_widget.dropEvent = self.dropEvent
        self.real_space_widget.dropEvent = self.dropEvent

        layout = QHBoxLayout()
        layout.addWidget(self.diffraction_space_widget, 1)
        layout.addWidget(self.real_space_widget, 1)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    # Handle dragging and dropping a file on the window
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if len(files) == 1:
            print(f"Reieving dropped file: {files[0]}")
            self.load_file(files[0])

