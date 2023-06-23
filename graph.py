import matplotlib.pyplot as plt

class Graph():
    def __init__(self): 
        self.xpoints = []
        self.ypoints = [] 
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Generation')
        self.ax.set_ylabel('Fitness')
        self.ax.set_title('Fitness over time')
        #self.ax.set_yscale('log')
        plt.show(block=False)
        self.line, = self.ax.plot(self.xpoints, self.ypoints)
        
    def update(self, x, y):
        self.xpoints.append(x)
        self.ypoints.append(y)
        self.line.set_xdata(self.xpoints)
        self.line.set_ydata(self.ypoints)
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        plt.draw()
        plt.pause(0.01)


'''
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication
import sys

class Graph():
    def __init__(self):
        self.xpoints = []
        self.ypoints = []
        self.app = QApplication(sys.argv)
        self.win = pg.PlotWidget(title='Fitness over time')
        self.win.resize(800, 600)
        self.win.setLabel('bottom', 'Generation')
        self.win.setLabel('left', 'Fitness')
        self.win.setLogMode(y=True)
        self.curve = self.win.plot(self.xpoints, self.ypoints)
        self.win.show()

    def update(self, x, y):
        self.xpoints.append(x)
        self.ypoints.append(y)
        self.curve.setData(self.xpoints, self.ypoints)
        self.app.processEvents()
'''