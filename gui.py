import Tkinter as tk
import tkFileDialog as tkfile
import sys, os
from uuid import uuid1 as uuid
import forcebalance

class ForceBalanceObject(object):
    def __init__(self, name='', type='object'):
        self.properties = dict()
        self.properties['name']=name
        self.properties['type'] = type
        self.properties['id'] = str(uuid())

    def __getitem__(self, item):
        return self.properties[item]

class ForceBalanceCalculation(ForceBalanceObject):
    def __init__(self, inputFile):
        super(ForceBalanceOptions, self).__init__(type='calculation')
        self.properties['inputfile'] = inputFile
        self.options, self.targets = forcebalance.parse_inputs(inputFile)

class ForceBalanceTestGUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.menubar = tk.Menu(self)
        filemenu = tk.Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="Open")
        filemenu.add_command(label="Exit", command=sys.exit)
        self.menubar.add_cascade(label="File", menu=filemenu)

        self['menu']=self.menubar

### program

window = ForceBalanceTestGUI()
window.mainloop()
