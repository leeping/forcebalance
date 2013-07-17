import Tkinter as tk
import tkFileDialog as tkfile
import sys, os
from uuid import uuid1 as uuid
import forcebalance
from collections import OrderedDict

class CalculationObject(object):
    def __init__(self, name='', type='object'):
        self.properties = dict()
        self.properties['active'] = False
        self.properties['name']=name
        self.properties['type'] = type
        self.properties['id'] = str(uuid())

    def __getitem__(self, item):
        return self.properties[item]

    def display(self, verbose = False):
        s = ''
        for key in self.properties.iterkeys():
            s+= "%s : %s\n" % (key, str(self.properties[key]))
        return s

class TargetObject(CalculationObject):
    def __init__(self, tgt_opt):
        super(TargetObject,self).__init__()
        self.opts = tgt_opt
        self.properties.update(self.opts)

    def display(self, verbose=0):
        s=''
        default_keys=[]
        for key in self.opts.iterkeys():
            if key not in forcebalance.parser.tgt_opts_defaults.keys() or self.opts[key] != forcebalance.parser.tgt_opts_defaults[key]:
                s+= "%s : %s\n" % (key, str(self.opts[key]))
            else: default_keys.append(key)
        if verbose:
            s+= "\n--- options remaining at default ---\n"
            for default_key in default_keys:
                s+= "%s : %s\n" % (default_key, str(self.opts[default_key]))
        
        return s
        

class OptionObject(CalculationObject):
    def __init__(self, opts=None, name="unknown options file"):
        super(OptionObject,self).__init__()

        if not opts: self.opts = forcebalance.parser.gen_opts_defaults
        else: self.opts = opts

        self.properties['name'] = name

    def display(self, verbose = False):
        s=''
        default_keys=[]
        for key in self.opts.iterkeys():
            if key not in forcebalance.parser.gen_opts_defaults.keys() or self.opts[key] != forcebalance.parser.gen_opts_defaults[key]:
                s+= "%s : %s\n" % (key, str(self.opts[key]))
            else: default_keys.append(key)
        if verbose:
            s+= "\n--- options remaining at default ---\n"
            for default_key in default_keys:
                s+= "%s : %s\n" % (default_key, str(self.opts[default_key]))
        return s
        

class ForcefieldObject(CalculationObject):
    def __init__(self, opts):
        super(ForcefieldObject, self).__init__()
        self.opts = opts
        self.forcefield = forcebalance.forcefield.FF(self.opts)
        self.properties["name"] = self.forcefield.fnms[0]

    def display(self, verbose = False):
        s=''
        s+= "name : " + self.properties["name"] + '\n'
        s+= "files : " + str(self.forcefield.fnms) + '\n'
        s+= "number of parameters : " + str(self.forcefield.np) + '\n'
        s+= "map : \n"
        for value in self.forcefield.map:
            s+= '\t' + str(value) + '\n'
        s+= "pvals : " + str(self.forcefield.pvals0) + '\n'
        return s
            

class ObjectViewer(tk.LabelFrame):
    def __init__(self,root):
        tk.LabelFrame.__init__(self, root, text="Loaded Objects")
        self.root = root

        self.calculations=[] # list of loaded option input files
        self.activeselection=None
        self.selectionchanged=tk.BooleanVar()
        self.selectionchanged.set(True)

        self.content = tk.Text(self, cursor="arrow", state="disabled", width="30")
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL)

        # bind scrollbar actions
        self.scrollbar.config(command = self.content.yview)
        self.content['yscrollcommand']=self.scrollbar.set

        # arrange and display list elements
        self.content.pack(side=tk.LEFT, fill=tk.Y)
        self.content.update()
        self.scrollbar.pack(side=tk.RIGHT,fill=tk.Y)

    def _bindSelection(self, widget):
        def select(e):
            self.select(e,widget)

        return select

    def add(self, obj):
        objtype = type(obj)

        if objtype==OptionObject:
            newcalc = {'options':obj,'targets':[],'forcefields':[]}
            self.calculations.append(newcalc)

        elif objtype==TargetObject: 
            self.calculations[-1]['targets'].append(obj)
        elif objtype==ForcefieldObject:
            self.calculations[-1]['forcefields'].append(obj)
        else: raise TypeError("ObjectViewer can only handle option, target, and forcefield objects")

    def update(self):
        self.content["state"]= "normal"
        self.content.delete("1.0","end")
        for calculation in self.calculations:
            self.content.window_create("end",window = tk.Label(self.content,text=calculation['options']['name'], bg="#FFFFFF"))
            self.content.insert("end",'\n')
            
            l = tk.Label(self.content,text="General Options", bg="#DEE4FA")
            self.content.window_create("end",window = l)
            l.bind('<Button-1>', self._bindSelection(calculation['options']))
            self.content.insert("end",'\n')

            self.content.window_create("end", window = tk.Label(self.content,text="Targets", bg="#FFFFFF"))
            self.content.insert("end",'\n')
            for target in calculation['targets']:
                l=tk.Label(self.content, text=target['name'], bg="#DEE4FA")
                self.content.window_create("end", window = l)
                self.content.insert("end",'\n')
                l.bind('<Button-1>', self._bindSelection(target))

            self.content.window_create("end", window = tk.Label(self.content,text="Forcefields", bg="#FFFFFF"))
            self.content.insert("end",'\n')
            for forcefield in calculation['forcefields']:
                l=tk.Label(self.content, text=forcefield['name'], bg="#DEE4FA")
                self.content.window_create("end", window = l)
                l.bind('<Button-1>', self._bindSelection(forcefield))
            self.content.insert("end",'\n\n')
        self.content["state"]="disabled"

    def select(self, e, o):
        for widget in self.content.winfo_children():
            if not widget['bg']=="#FFFFFF":
                widget['bg']='#DEE4FA'
        e.widget['bg']='#4986D6'
        self.activeselection=o
        self.selectionchanged.get() # reading this variable triggers a refresh

    def scrollUp(self, e):
        self.content.yview('scroll', -1, 'units')

    def scrollDown(self, e):
        self.content.yview('scroll', 1, 'units')

class DetailViewer(tk.LabelFrame):
    def __init__(self, root, opts=''):
        # initialize variables
        self.root = root
        self.printAll = tk.IntVar()
        self.printAll.set(False)
        self.currentObject = None # keep current object in case view needs refreshing

        # Viewer GUI elements
        tk.LabelFrame.__init__(self, root, text="Details Viewer")
        self.content = tk.Text(self,cursor="arrow",state="disabled")
        self.content.tag_config("error", foreground="red")
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL)

        # bind scrollbar actions
        self.scrollbar.config(command = self.content.yview)
        self.content['yscrollcommand']=self.scrollbar.set
        self.scrollbar.pack(side=tk.RIGHT,fill=tk.Y)

        # arrange and display list elements
        self.content.pack(side=tk.LEFT, fill=tk.Y)
        self.content.update()

        # right click context menu
        self.contextmenu = tk.Menu(self, tearoff=0)
        self.contextmenu.add_checkbutton(label="show all", variable=self.printAll)
        self.content.bind("<Button-3>", lambda e : self.contextmenu.post(e.x_root, e.y_root))
        self.content.bind("<Button-1>", lambda e : self.contextmenu.unpost())
        self.printAll.trace('w', lambda *x : self.load())

    def load(self,newObject=None):
        if newObject:
            self.currentObject = newObject
            self.printAll.set(0)
        if not self.currentObject:
            return   # no object being viewed, nothing to do!

        self.content["state"]="normal"
        self.content.delete("1.0","end")

        try:
            self.content.insert("end", self.currentObject.display(self.printAll.get()))
        except:
            #self.content.insert("end", "Error trying to display <%s %s>\n" % (self.currentObject['type'], self.currentObject['name']), "error")
            self.content.insert("end", "%s\n" % self.currentObject)
            from traceback import format_exc
            self.content.insert("end", format_exc(), "error")

        self.content["state"]="disabled"

    def scrollUp(self, e):
        self.content.yview('scroll', -1, 'units')

    def scrollDown(self, e):
        self.content.yview('scroll', 1, 'units')

class ConsoleViewer(tk.LabelFrame):
    def __init__(self, root):
        tk.LabelFrame.__init__(self, root, text="Console")
        
        self.console = tk.Text(self, state="disabled",cursor="arrow")
        self.console.pack(fill=tk.Y)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, input):
        self.console['state']="normal"
        self.console.insert(tk.END, input)
        self.stdout.write(input)
        self.scrollDown()
        self.console['state']="disabled"

    def flush(self):
        self.stdout.flush()

    def scrollUp(self, *args):
        self.console.yview('scroll', -1, 'units')

    def scrollDown(self, *args):
        self.console.yview('scroll', 1, 'units')
            

class MainWindow(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.title("ForceBalance 1.1")
        self._initialize_menu()

        ## Window Components
        self.objectsPane = ObjectViewer(self)
        self.detailsPane = DetailViewer(self)
        #self.consolePane = ConsoleViewer(self)
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
        filemenu.add_command(label="New")
        filemenu.add_command(label="Open", command=self.open)
        filemenu.add_command(label="Save")
        filemenu.add_command(label="Close")
        filemenu.add_command(label="Exit", command=sys.exit)
        self.menubar.add_cascade(label="File", menu=filemenu)

        self['menu']=self.menubar

        calculationmenu = tk.Menu(self.menubar, tearoff=0)
        calculationmenu.add_command(label="Check", state="disabled")
        calculationmenu.add_command(label="Run")
        self.menubar.add_cascade(label="Calculation", menu=calculationmenu)

    def open(self):
        filters = [('Forcebalance input files', '*.in'),('Show all', '*')]
        inputfile = tkfile.askopenfilename(title="Open ForceBalance input file...", filetypes=filters)

        if inputfile=='': return

        cwd=os.getcwd()
        os.chdir(os.path.dirname(inputfile))
        opts, tgt_opts = forcebalance.parser.parse_inputs(inputfile)

        self.objectsPane.add(OptionObject(opts, os.path.split(inputfile)[1]))
        for target in tgt_opts:
            self.objectsPane.add(TargetObject(target))
        self.objectsPane.add(ForcefieldObject(opts))
        self.objectsPane.update()
        
        os.chdir(cwd)

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

        

x = MainWindow()
x.mainloop()