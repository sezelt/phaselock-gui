#!/usr/bin/env python

import phaselock_gui
import sys
from PyQt5.QtWidgets import QApplication


def launch():
    app = QApplication(sys.argv)
    win = phaselock_gui.DataViewer(sys.argv)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch()
