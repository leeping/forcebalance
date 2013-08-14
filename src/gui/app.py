import Tkinter as tk
import tkFileDialog as tkfile
import sys, os
import elements

class MainWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.title("ForceBalance 1.1")

        ## Window Components
        # general view of selectable objects in calculation
        self.objectsPane = elements.ObjectViewer(self)
        # more detailed properties relating to selected object(s)
        self.detailsPane = elements.DetailViewer(self)
        # forcebalance output is routed to the console pane
        self.consolePane = elements.ConsoleViewer(self)
        
        # Arrange components using grid manager
        self.objectsPane.grid(row=0, column=0)
        self.detailsPane.grid(row=0, column=1)
        self.consolePane.grid(row=1, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S)
        
        # Allow components to be resized
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)
        
        self._initialize_menu()

        # Application-wide event bindings
        self.objectsPane.selectionchanged.trace('r', self.updateDetailView)

    def _initialize_menu(self):
        self.menubar = tk.Menu(self)
        filemenu = tk.Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="New", state="disabled")

        filemenu.add_command(label="Open", command=self.open)
        filemenu.add_command(label="Save", state="disabled")
        filemenu.add_command(label="Save As...",state="disabled")
        filemenu.add_command(label="Close", command=self.close)
        filemenu.add_command(label="Exit", command=sys.exit)
        self.menubar.add_cascade(label="File", menu=filemenu)
        self.menubar.add_command(label="Run", command=self.objectsPane.run, state="disabled")

        self['menu']=self.menubar

        #calculationmenu = tk.Menu(self.menubar, tearoff=0)
        #calculationmenu.add_command(label="Check", state="disabled")
        #calculationmenu.add_command(label="Run", command=self.objectsPane.run)
        #self.menubar.add_cascade(label="Calculation", menu=calculationmenu)

    def open(self):
        filters = [('Forcebalance input files', '*.in'),('Show all', '*')]
        inputfile = tkfile.askopenfilename(title="Open ForceBalance input file...", filetypes=filters)
        if inputfile:
            self.consolePane.clear()
            self.objectsPane.open(inputfile)
            self.menubar.entryconfig(2, state=tk.NORMAL)
            print dir(self.menubar)

    def close(self):
        self.objectsPane.clear()
        self.detailsPane.clear()
        self.menubar.entryconfig(2, state=tk.DISABLED)

    def updateDetailView(self, *args):
        self.detailsPane.load(self.objectsPane.activeselection)
