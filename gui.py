#!/usr/bin/env python

import sys
import os.path as op

from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import h5py
import pylab as P
from matplotlib import colors
import pyqtgraph as pg

class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self._initUI()

    def _initUI(self):
        self.setWindowTitle('GUI')
        self.resize(1200,800)

        self.main_widget = QtWidgets.QWidget(self)
        self.main_layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.setCentralWidget(self.main_widget)

        line = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(line)
        self.main_imview = pg.ImageView()
        line.addWidget(self.main_imview, stretch=2)

        vbox = QtWidgets.QVBoxLayout()
        line.addLayout(vbox, stretch=1)
        self.plot_widget = pg.PlotWidget()
        vbox.addWidget(self.plot_widget, stretch=1)
        self.param_type = QtWidgets.QComboBox()
        self.param_type.addItems(['angles', 'shifts', 'diameters'])
        self.param_type.currentIndexChanged.connect(self._update)
        vbox.addWidget(self.param_type)

        line = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(line)
        label = QtWidgets.QLabel('File:')
        line.addWidget(label)
        self.filename = QtWidgets.QLabel()
        line.addWidget(self.filename)
        line.addStretch(1)
        self.view_mag = QtWidgets.QCheckBox('View magnitude')
        self.view_mag.stateChanged.connect(self._update)
        line.addWidget(self.view_mag)
        self.prev_button = QtWidgets.QPushButton('-')
        self.prev_button.clicked.connect(self._prev_file)
        self.prev_button.setFixedWidth(20)
        self.prev_button.setEnabled(False)
        line.addWidget(self.prev_button)
        self.next_button = QtWidgets.QPushButton('+')
        self.next_button.clicked.connect(self._next_file)
        self.next_button.setFixedWidth(20)
        self.next_button.setEnabled(False)
        line.addWidget(self.next_button)
        self.load_button = QtWidgets.QPushButton('Load')
        line.addWidget(self.load_button)
        self.load_button.clicked.connect(self._load_dialog)

        line = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(line)
        label = QtWidgets.QLabel('Reference File:')
        line.addWidget(label)
        self.ref_filename = QtWidgets.QLabel('data/norot/photons_norot_meta.h5')
        line.addWidget(self.ref_filename)
        line.addStretch(1)
        self.refload_button = QtWidgets.QPushButton('Load')
        line.addWidget(self.refload_button)
        self.refload_button.clicked.connect(lambda x: self._refload(self.ref_filename.text()))
        self.refload_button.click()
        button = QtWidgets.QPushButton('Quit')
        button.clicked.connect(self.close)
        line.addWidget(button)

        self.show()

    def _update(self, index=None, data=None):
        if data is None:
            data = self.model
        curr_range = self.main_imview.getHistogramWidget().getLevels()
        if self.view_mag.isChecked():
            rgbdata = np.abs(data)
        else:
            hsvdata = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.float64)
            hsvdata[..., 0] = np.clip(np.angle(data), -np.pi, np.pi) / (2*np.pi) + 0.5
            hsvdata[..., 1] = 1
            hsvdata[..., 2] = np.clip(np.abs(data)**0.4, 0, curr_range[1])
            rgbdata = colors.hsv_to_rgb(hsvdata)
        self.main_imview.setImage(rgbdata, levels=curr_range)

        self.plot_widget.clear()
        key = self.param_type.currentText()
        if key in ['angles', 'diameters']:
            self.plot_widget.plot(self.refparams[key], self.params[key],
                                  pen=None, symbol='o',
                                  symbolPen=None, symbolSize=5,
                                  symbolBrush=(255, 255, 255, 32))
            cc = np.corrcoef(self.refparams[key], self.params[key])[0,1]
            rms = np.sqrt(((self.refparams[key]-self.params[key])**2).mean())
            self.plot_widget.getPlotItem().setTitle('%s (RMS=%.4f, CC=%.6f)' % (key, rms, cc))
        else:
            self.plot_widget.plot(self.refparams[key][:,0], self.params[key][:,0],
                                  pen=None, symbol='o',
                                  symbolPen=None, symbolSize=5,
                                  symbolBrush=(0, 255, 255, 32))
            self.plot_widget.plot(self.refparams[key][:,1], self.params[key][:,1],
                                  pen=None, symbol='o',
                                  symbolPen=None, symbolSize=5,
                                  symbolBrush=(255, 255, 0, 32))
            self.plot_widget.getPlotItem().setTitle('%s (CC = %.3f, %3f)' % (key,
                                                                             np.corrcoef(self.refparams[key][:,0], self.params[key][:,0])[0,1],
                                                                             np.corrcoef(self.refparams[key][:,1], self.params[key][:,1])[0,1]))

    def _load_dialog(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', './data/norot/')[0]
        if fname == '':
            return
        self._load(fname)

    def _load(self, fname):
        self.filename.setText(fname)
        with h5py.File(fname, 'r') as f:
            if 'true_model' in f:
                names = ['true_model', 'true_angles', 'true_shifts', 'true_diameters']
            else:
                names = ['model', 'angles', 'shifts', 'diameters']
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)
            self.model = f[names[0]][...]
            if names[1] in f:
                self.params = {}
                self.params['angles'] = f[names[1]][:]
                self.params['shifts'] = f[names[2]][:]
                self.params['diameters'] = f[names[3]][:]
            
        size = int(np.rint(self.model.size**0.5))
        self.model = self.model.reshape((size, size))
        self._update()

    def _refload(self, fname):
        self.ref_filename.setText(fname)
        with h5py.File(fname, 'r') as f:
            self.refmodel = f['true_model'][...]
            self.refparams = {}
            self.refparams['angles'] = f['true_angles'][:]
            self.refparams['shifts'] = f['true_shifts'][:]
            self.refparams['diameters'] = f['true_diameters'][:]

    def _next_file(self):
        curr_fname = self.filename.text()
        iternum = int(curr_fname.split('_')[-1].split('.')[0])
        new_fname = curr_fname.split('_')[0] + '_%.3d.h5'%(iternum+1)
        if not op.isfile(new_fname):
            return
        self._load(new_fname)

    def _prev_file(self):
        curr_fname = self.filename.text()
        iternum = int(curr_fname.split('_')[-1].split('.')[0])
        if iternum == 0:
            return
        new_fname = curr_fname.split('_')[0] + '_%.3d.h5'%(iternum-1)
        if not op.isfile(new_fname):
            return
        self._load(new_fname)

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = GUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
