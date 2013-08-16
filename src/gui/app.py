import Tkinter as tk
import tkFileDialog as tkfile
import sys, os
import elements

class MainWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.title("ForceBalance 1.1")
        self.geometry("800x600")

        ## Window Components
        # general view of selectable objects in calculation
        self.objectsPane = elements.ObjectViewer(self)
        # more detailed properties relating to selected object(s)
        self.detailsPane = elements.DetailViewer(self)
        # forcebalance output is routed to the console pane
        self.consolePane = elements.ConsoleViewer(self, height = 500)
        
        self.showObjects = tk.IntVar()
        self.showObjects.set(True)
        
        self.arrangeWindow()
        
        self._initialize_menu()

        # Application-wide event bindings
        self.objectsPane.selectionchanged.trace('r', self.updateDetailView)
        self.showObjects.trace('w', self.arrangeWindow)

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
        
        optionsmenu = tk.Menu(self.menubar, tearoff=0)
        optionsmenu.add_checkbutton(label="show object viewer", variable=self.showObjects)
        optionsmenu.add_checkbutton(label="show default options", variable=self.detailsPane.printAll)
        self.menubar.add_cascade(label="Options", menu=optionsmenu)
        
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
            self.menubar.entryconfig(3, state=tk.NORMAL)

    def close(self):
        self.objectsPane.clear()
        self.detailsPane.clear()
        self.menubar.entryconfig(3, state=tk.DISABLED)
        
    def arrangeWindow(self, *args):
        self.objectsPane.grid_forget()
        self.detailsPane.grid_forget()
        self.consolePane.grid_forget()
    
        if self.showObjects.get():
            self.consolePane.grid(row=1, columnspan=2, sticky=tk.W+tk.E+tk.N+tk.S)
            self.objectsPane.grid(row=0, column=0, sticky=tk.W+tk.E+tk.N+tk.S)
            self.detailsPane.grid(row=0, column=1, sticky=tk.W+tk.E+tk.N+tk.S)
            self.rowconfigure(0, weight=0)
            self.rowconfigure(1, weight=1)
            self.columnconfigure(0, weight=1)
            self.columnconfigure(1, weight=1)            
            
        else:
            self.consolePane.grid(sticky=tk.W+tk.E+tk.N+tk.S)
            self.rowconfigure(0, weight=1)
            self.rowconfigure(1, weight=0)
            self.columnconfigure(0, weight=1)
            self.columnconfigure(1, weight=0)  

    def updateDetailView(self, *args):
        self.detailsPane.load(self.objectsPane.activeselection)
