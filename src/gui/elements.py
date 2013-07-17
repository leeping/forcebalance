import Tkinter as tk
import tkFileDialog as tkfile
import sys, os

import objects
import forcebalance

class ObjectViewer(tk.LabelFrame):
    def __init__(self,root):
        tk.LabelFrame.__init__(self, root, text="Loaded Objects")
        self.root = root

        self.calculations=[] # list of loaded option input files
        self.activeselection=None
        self.selectionchanged=tk.BooleanVar()
        self.selectionchanged.set(True)
        self.needUpdate=tk.BooleanVar()
        self.needUpdate.trace('r',self.update)

        self.content = tk.Text(self, cursor="arrow", state="disabled", width="30")
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL)

        # bind scrollbar actions
        self.scrollbar.config(command = self.content.yview)
        self.content['yscrollcommand']=self.scrollbar.set

        # arrange and display list elements
        self.content.pack(side=tk.LEFT, fill=tk.Y)
        self.content.update()
        self.scrollbar.pack(side=tk.RIGHT,fill=tk.Y)

    def _bindEventHandler(self, handler, object):
        def f(e):
            return handler(e, object)
        return f

    def open(self, filename):
        """Parse forcebalance input file and add referenced objects"""
        if filename=='': return

        self.calculations.append(objects.CalculationObject(filename))
        self.update()

    def update(self, *args):
        self.content["state"]= "normal"
        self.content.delete("1.0","end")
        for calculation in self.calculations:
            self.content.window_create("end",window = tk.Label(self.content,text=calculation['options']['name'], bg="#FFFFFF"))
            self.content.insert("end",'\n')
            
            l = tk.Label(self.content,text="General Options", bg="#DEE4FA")
            self.content.window_create("end",window = l)
            l.bind('<Button-1>', self._bindEventHandler(self.select, calculation['options']))
            self.content.insert("end",'\n')

            def toggle_targets(e):
                if calculation['_expand_targets']:
                    calculation['_expand_targets']=False
                else: calculation['_expand_targets']=True
                self.needUpdate.get()
            
            targetLabel = tk.Label(self.content,text="Targets", bg="#FFFFFF")
            self.content.window_create("end", window = targetLabel)
            targetLabel.bind("<Button-1>", self._bindEventHandler(self.toggle, calculation))
            self.content.insert("end",'\n')

            if calculation['_expand_targets']:
                for target in calculation['targets']:
                    self.content.insert("end",'  ')
                    l=tk.Label(self.content, text=target['name'], bg="#DEE4FA")
                    self.content.window_create("end", window = l)
                    self.content.insert("end",'\n')
                    l.bind('<Button-1>', self._bindEventHandler(self.select, target))

            l=tk.Label(self.content, text="Forcefield", bg="#DEE4FA")
            self.content.window_create("end", window = l)
            l.bind('<Button-1>', self._bindEventHandler(self.select, calculation['forcefield']))
            self.content.insert("end",'\n\n')
        self.content["state"]="disabled"

    def select(self, e, o):
        for widget in self.content.winfo_children():
            if not widget['bg']=="#FFFFFF":
                widget['bg']='#DEE4FA'
        e.widget['bg']='#4986D6'
        self.activeselection=o
        self.selectionchanged.get() # reading this variable triggers a refresh

    def toggle(self, e, calculation):
        if calculation['_expand_targets']: calculation['_expand_targets'] = False
        else: calculation['_expand_targets'] = True
        self.needUpdate.get()

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
        tk.LabelFrame.__init__(self, root, text="Details")
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
        self.content["state"]="normal"
        self.content.delete("1.0","end")
        if newObject:
            self.currentObject = newObject
            self.printAll.set(0)
        else:
            self.content["state"]="disabled"
            return   # no object being viewed, nothing else to do!

        self['text']="Details - %s" % self.currentObject['name']

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