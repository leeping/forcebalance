import Tkinter as tk
import tkFileDialog as tkfile
import sys, os
import elements

class MainWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.title("ForceBalance 1.1")
        self._initialize_menu()

        ## Window Components
        self.objectsPane = elements.ObjectViewer(self)
        self.detailsPane = elements.DetailViewer(self)
        #self.consolePane = elements.ConsoleViewer(self)
        self.objectsPane.pack(side=tk.LEFT, fill=tk.Y)
        self.detailsPane.pack(side=tk.LEFT, fill=tk.Y)
        #self.consolePane.pack(side=tk.LEFT, fill=tk.Y)

        # Application-wide event bindings
        self.bind('<Button-4>', self.scrollWidgetUp)
        self.bind('<Button-5>', self.scrollWidgetDown)
        self.objectsPane.selectionchanged.trace('r', self.updateDetailView)

    def _initialize_menu(self):
        self.menubar = tk.Menu(self)
        filemenu = tk.Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="New", state="disabled")

        filemenu.add_command(label="Open", command=self.open)
        filemenu.add_command(label="Save", state="disabled")
        filemenu.add_command(label="Close", command=self.close)
        filemenu.add_command(label="Exit", command=sys.exit)
        self.menubar.add_cascade(label="File", menu=filemenu)

        self['menu']=self.menubar

        calculationmenu = tk.Menu(self.menubar, tearoff=0)
        calculationmenu.add_command(label="Check", state="disabled")
        calculationmenu.add_command(label="Run", state="disabled")
        self.menubar.add_cascade(label="Calculation", menu=calculationmenu, state="disabled")

    def open(self):
        filters = [('Forcebalance input files', '*.in'),('Show all', '*')]
        inputfile = tkfile.askopenfilename(title="Open ForceBalance input file...", filetypes=filters)
        self.objectsPane.open(inputfile)

    def close(self):
        self.objectsPane.clear()
        self.detailsPane.clear()

    def updateDetailView(self, *args):
        self.detailsPane.load(self.objectsPane.activeselection)

    def scrollWidgetDown(self, event):
        p = {'x': self.winfo_pointerx(), 'y':self.winfo_pointery()}

        for widget in self.winfo_children():
            widgetTop={'x':widget.winfo_rootx(), 'y':widget.winfo_rooty()}
            widgetBottom={'x': widget.winfo_rootx() + widget.winfo_width(),\
                          'y': widget.winfo_rooty() + widget.winfo_height()}

            if p[tk.X] > widgetTop[tk.X] and\
               p[tk.Y] > widgetTop[tk.Y] and\
               p[tk.X] < widgetBottom[tk.X] and\
               p[tk.Y] < widgetBottom[tk.Y]:
                try: 
                    widget.scrollDown(event)
                    break
                except: pass

    def scrollWidgetUp(self, event):
        p = {'x': self.winfo_pointerx(), 'y':self.winfo_pointery()}

        for widget in self.winfo_children():
            widgetTop={'x':widget.winfo_rootx(), 'y':widget.winfo_rooty()}
            widgetBottom={'x': widget.winfo_rootx() + widget.winfo_width(),\
                          'y': widget.winfo_rooty() + widget.winfo_height()}

            if p[tk.X] > widgetTop[tk.X] and\
               p[tk.Y] > widgetTop[tk.Y] and\
               p[tk.X] < widgetBottom[tk.X] and\
               p[tk.Y] < widgetBottom[tk.Y]:
                try: 
                    widget.scrollUp(event)
                    break
                except: pass