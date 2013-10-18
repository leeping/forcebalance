import Tkinter as tk
import tkFileDialog as tkfile
import sys, os, re
import threading

import objects
from forcebalance.output import getLogger, StreamHandler, INFO

class ObjectViewer(tk.LabelFrame):
    """Provides a general overview of the loaded calculation objects"""
    def __init__(self,root, **kwargs):
        tk.LabelFrame.__init__(self, root, text="Loaded Objects", **kwargs)
        self.root = root

        self.calculation=None
        self.activeselection=None
        
        # needUpdate signals when the list of objects needs to be refreshed
        self.needUpdate=tk.BooleanVar()
        self.needUpdate.trace('r',self.update)
        
        # selectionchanged can be used by the root window to determine when
        # the DetailViewer needs to be loaded with a new object
        self.selectionchanged=tk.BooleanVar()
        self.selectionchanged.set(True)

        self.content = tk.Text(self, cursor="arrow", state="disabled", width=30, height=20)
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL)

        # bind scrollbar actions
        self.scrollbar.config(command = self.content.yview)
        self.content['yscrollcommand']=self.scrollbar.set

        # arrange and display list elements
        self.content.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.content.update()
        self.scrollbar.pack(side=tk.RIGHT,fill=tk.Y)

    def open(self, filename):
        """Parse forcebalance input file and add referenced objects"""
        if filename=='': return

        self.calculation = objects.CalculationObject(filename)
        self.update()

    def clear(self):
        self.calculation = None
        self.update()

    def run(self):
        def runthread():
            cwd = os.getcwd()
            os.chdir(self.calculation['options'].opts['root'])
            try: self.calculation.run()
            except:
                print "An Error occurred"
            self.update()
        
        if threading.active_count() < 2:
            calculation_thread = threading.Thread(target=runthread)
            calculation_thread.start()
        else:
            print "Calculation already running"

    def update(self, *args):
        """Update the list of objects being displayed, based on the contents of self.calculation"""
        self.content["state"]= "normal"
        self.content.delete("1.0","end")
        self['text']="Objects"

        if self.calculation:
            self['text'] += " - " + self.calculation['options']['name']
            self.content.bind('<Button-1>', _bindEventHandler(self.select, object = [ self.calculation ]))
            
            self.content.insert("end",' ')
            l = tk.Label(self.content,text="General Options", bg="#FFFFFF")
            self.content.window_create("end",window = l)
            l.bind('<Button-1>', _bindEventHandler(self.select, object = [ self.calculation['options'] ]))
            self.content.insert("end",'\n')
                

            # Event handler to toggle whether targets list should be expanded
            def toggle(e):
                self.calculation['_expand_targets'] = not self.calculation['_expand_targets']
                self.needUpdate.get()

            targetLabel = tk.Label(self.content,text="Targets", bg="#FFFFFF")
            targetLabel.bind("<Button-3>", toggle)
            targetLabel.bind('<Button-1>', _bindEventHandler(self.select, object = self.calculation['targets']))

            if self.calculation['_expand_targets']:
                self.content.insert("end",' ')
                self.content.window_create("end", window = targetLabel)
                self.content.insert("end",'\n')
                for target in self.calculation['targets']:
                    self.content.insert("end",'   ')
                    l=tk.Label(self.content, text=target['name'], bg="#FFFFFF")
                    self.content.window_create("end", window = l)
                    self.content.insert("end",'\n')
                    l.bind('<Button-1>', _bindEventHandler(self.select, object=[ target ]))
            else:
                self.content.insert("end",'+')
                self.content.window_create("end", window = targetLabel)
                self.content.insert("end",'\n')

            self.content.insert("end",' ')
            l=tk.Label(self.content, text="Forcefield", bg="#FFFFFF")
            self.content.window_create("end", window = l)
            l.bind('<Button-1>', _bindEventHandler(self.select, object=[ self.calculation['forcefield'] ]))
            self.content.insert("end",'\n')
            
            if self.calculation["result"]:
                self.content.insert("end",' ')
                l=tk.Label(self.content, text="Result", bg="#FFFFFF")
                self.content.window_create("end", window = l)
                l.bind('<Button-1>', _bindEventHandler(self.select, object=[ self.calculation["result"] ]))
                self.content.insert("end",'\n\n')

        self.content["state"]="disabled"

    def select(self, e, object):
        for widget in self.content.winfo_children():
            widget["relief"]=tk.FLAT
        e.widget["relief"]="solid"

        if type(object) is not list: self.activeselection=[ object ]
        else: self.activeselection=object

        self.selectionchanged.get() # reading this variable triggers a refresh
        
    def scrollUp(self, e):
        self.content.yview('scroll', -2, 'units')

    def scrollDown(self, e):
        self.content.yview('scroll', 2, 'units')

class DetailViewer(tk.LabelFrame):
    """Shows detailed properties of the currently selected object (as defined in
    and ObjectViewer)"""
    def __init__(self, root, opts='', **kwargs):
        # initialize variables
        self.root = root
        self.printAll = tk.IntVar()
        self.printAll.set(False)
        self.currentObject = None # keep current object in case view needs refreshing
        self.currentElement= None # currently selected element within current object

        # Viewer GUI elements
        tk.LabelFrame.__init__(self, root, text="Details", **kwargs)
        self.content = tk.Text(self,cursor="arrow",state="disabled", height=20)
        self.content.tag_config("error", foreground="red")
        self.scrollbar = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.helptext = tk.Text(self.root, width=70, state="disabled", bg="#F0F0F0", wrap=tk.WORD)

        # bind scrollbar actions
        self.scrollbar.config(command = self.content.yview)
        self.content['yscrollcommand']=self.scrollbar.set
        self.scrollbar.pack(side=tk.RIGHT,fill=tk.Y)
        self.root.bind_class("scrollable", "<Button-4>", self.scrollUp)
        self.root.bind_class("scrollable", "<Button-5>", self.scrollDown)

        # arrange and display list elements
        self.content.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        self.content.update()

        # update when printAll variable is changed
        self.printAll.trace('w', lambda *x : self.load())

    def load(self,newObject=None):
        """update current object if a new one is passed in. Then clear existing text and
        replace it with whatever the current object says to write"""
        if newObject:
            self.currentObject = newObject

        self['text']="Details"

        self.content["state"]="normal"
        self.content.delete("1.0","end")
        if self.currentObject:
            if len(self.currentObject) ==1:   # if there is an object to display and it is not a collection
                self['text']+=" - %s" % self.currentObject[0]['name']
            else:
                self['text']+=" - %d Configured Targets" % len(self.currentObject)
                
            try:
                for object in self.currentObject:
                    self.populate(object)
            except:
                self.content.insert("end", "Error trying to display <%s %s>\n" % (self.currentObject[0]['type'], self.currentObject[0]['name']), "error")
                from traceback import format_exc
                self.content.insert("end", format_exc(), "error")
        
        self.content["state"]="disabled"

    def populate(self, object):
        """Populate the view with information in displayText argument"""
        # ask object to tell us how to display it
        displayText = object.display(self.printAll.get())

        # if object provides string back, just print it to the content area
        if type(displayText)==str:
            self.content.insert("end", displayText)

        # if object provides dictionary, iterate through key value pairs and organize using tk.Label objects
        if type(displayText)==dict:
            for key in displayText.keys():
                frame = tk.Frame(self.content)
                frame.bindtags((key, "scrollable"))
                keylabel = tk.Label(frame, text=key, bg="#FFFFFF", padx=0, pady=0)
                keylabel.bindtags((key, "scrollable"))
                separator = tk.Label(frame, text=" : ", bg="#FFFFFF", padx=0, pady=0)
                separator.bindtags((key, "scrollable"))
                valuelabel = tk.Label(frame, text= str(displayText[key]), bg="#FFFFFF", padx=0, pady=0)
                valuelabel.bindtags((key, "scrollable"))

                keylabel.pack(side=tk.LEFT)
                separator.pack(side=tk.LEFT)
                valuelabel.pack(side=tk.LEFT)

                self.content.window_create("end", window = frame)
                self.content.insert("end", '\n')

        # tuple is used to separate out options set at default values
        # (<options dictionary>, <default options dictionary>)
        if type(displayText)==tuple:
            for key in displayText[0].keys():
                frame = tk.Frame(self.content)
                frame.bindtags((key, "scrollable"))
                keylabel = tk.Label(frame, text=key, bg="#FFFFFF", padx=0, pady=0)
                keylabel.bindtags((key, "scrollable"))
                separator = tk.Label(frame, text=" : ", bg="#FFFFFF", padx=0, pady=0)
                separator.bindtags((key, "scrollable"))
                valuelabel = tk.Label(frame, text= str(displayText[0][key]), bg="#FFFFFF", padx=0, pady=0)
                valuelabel.bindtags((key, "scrollable"))

                keylabel.pack(side=tk.LEFT)
                separator.pack(side=tk.LEFT)
                valuelabel.pack(side=tk.LEFT)

                self.content.window_create("end", window = frame)
                self.content.insert("end", '\n')

                # right click help popup
                self.root.bind_class(key, "<Button-3>", _bindEventHandler(self.showHelp, object = object, option=key))

            if self.printAll.get():
                self.content.insert("end", "\n--- Default Values ---\n")
                for key in displayText[1].keys():
                    frame = tk.Frame(self.content)
                    frame.bindtags((key, "scrollable"))
                    keylabel = tk.Label(frame, text=key, bg="#FFFFFF", padx=0, pady=0)
                    keylabel.bindtags((key, "scrollable"))
                    separator = tk.Label(frame, text=" : ", bg="#FFFFFF", padx=0, pady=0)
                    separator.bindtags((key, "scrollable"))
                    valuelabel = tk.Label(frame, text= str(displayText[1][key]), bg="#FFFFFF", padx=0, pady=0)
                    valuelabel.bindtags((key, "scrollable"))

                    keylabel.pack(side=tk.LEFT)
                    separator.pack(side=tk.LEFT)
                    valuelabel.pack(side=tk.LEFT)

                    self.content.window_create("end", window = frame)
                    self.content.insert("end", '\n')

                    self.root.bind_class(key, "<Button-3>", _bindEventHandler(self.showHelp, object = object, option=key))

        self.content.insert("end",'\n')

    def clear(self):
        """Clear the current object and reload a blank page"""
        self.currentObject=None
        self.load()

    def showHelp(self, e, object, option):
        """Update and display help window showing option documentation string"""
        self.helptext["state"]="normal"
        self.helptext.delete("1.0","end")

        # get message and calculate how high window should be 
        helpmessage = object.getOptionHelp(option)
        height=0
        for line in object.getOptionHelp(option).splitlines():
            height += 1 + len(line)/70

        self.helptext.insert("end", helpmessage)
        self.helptext['height']=height
        self.helptext.place(x=e.x_root-self.root.winfo_x(), y=e.y_root-self.root.winfo_y())
        self.root.bind("<Motion>", lambda e : self.helptext.place_forget())
        self.root.bind("<Button>", lambda e : self.helptext.place_forget())
        
    def scrollUp(self, e):
        self.content.yview('scroll', -2, 'units')

    def scrollDown(self, e):
        self.content.yview('scroll', 2, 'units')

class ConsoleViewer(tk.LabelFrame):
    """Tries to emulate a terminal by displaying standard output"""
    def __init__(self, root, **kwargs):
        tk.LabelFrame.__init__(self, root, text="Console", **kwargs)
        
        self.console = tk.Text(self,
                            state=tk.DISABLED,
                            cursor="arrow",
                            fg="#FFFFFF",
                            bg="#000000")
        self.console.pack(fill=tk.BOTH, expand=1)
        
        # console colors corresponding to ANSI escape sequences
        self.console.tag_config("0", foreground="white", background="black")
        #self.console.tag_config("1")  # make text bold
        self.console.tag_config("44", background="blue")
        self.console.tag_config("91", foreground="red")
        self.console.tag_config("92", foreground="green")
        self.console.tag_config("93", foreground="yellow")
        self.console.tag_config("94", foreground="blue")
        self.console.tag_config("95", foreground="purple")

        getLogger("forcebalance").addHandler(ConsolePaneHandler(self))
        getLogger("forcebalance").setLevel(INFO)
        
        # scroll to bottom of text when widget is resized
        self.bind("<Configure>", lambda e: self.console.yview(tk.END))

    ## we implement write and flush so the console viewer
    #  can serve as a drop in replacement for sys.stdout
    def write(self, input, tags="0"):
        self.console['state']=tk.NORMAL
        
        # processing of input
        input = re.sub("\r","\n", input)
        
        self.console.insert(tk.END, input, tags)
        self.console.yview(tk.END)
        self.console['state']=tk.DISABLED

    def flush(self):
        pass    # does nothing since messages are sent to the console immediately on write
        
    def clear(self):
        self.console['state']=tk.NORMAL
        self.console.delete(1.0, tk.END)
        self.console['state']=tk.DISABLED
        
class ConsolePaneHandler(StreamHandler):
    def __init__(self, console):
        super(ConsolePaneHandler, self).__init__()
        self.console = console
        self.color = ["0"]
    
    def emit(self, record):
        # split messages by looking for terminal escape sequences
        message = re.split("(\x1b\[[01]?;?[0-9]{1,2};?[0-9]{,2}m)", record.getMessage())
        
        for section in message:
            # if we get a new escape sequence, assign self.color to new color codes
            # else write text using the current color codes
            if section[0] == "\x1b":
                self.color = tuple(section[2:-1].split(';'))
            else:
                self.console.write(section, tags=self.color)
                
        self.flush()
        
def _bindEventHandler(handler, **kwargs):
    """Creates an event handler by taking a function that takes many arguments
    and using kwargs to create a function that only takes in one argument"""
    def f(e):
        return handler(e, **kwargs)
    return f

