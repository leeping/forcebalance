import forcebalance
from uuid import uuid1 as uuid
import os

"""
This file contains classes that interface with forcebalance and act as an intermediary
between the GUI frontend and lower level calculation elements.

Objects that need to be displayed should implement a display() function that returns
a type that can be recognized by the widget. Generic display function that returns a
string filled with an object's properties is provided by ForceBalanceObject, but
more complex data structures like dictionaries provide information on structure that
can be used by the widget in creating more complicated views.
"""

class ForceBalanceObject(object):
    """Generic forcebalance object with some identifying information"""
    def __init__(self, name='', type='object'):
        self.properties = dict()
        self.properties['active'] = False
        self.properties['name']=name
        self.properties['type'] = type
        self.properties['id'] = str(uuid())

    def __getitem__(self, item):
        return self.properties[item]

    def __setitem__(self, key, value):
        self.properties[key]=value

    def display(self, verbose = False):
        s = ''
        for key in self.properties.iterkeys():
            s+= "%s : %s\n" % (key, str(self.properties[key]))
        return s

class CalculationObject(ForceBalanceObject):
    """Represents a set of forcebalance objects that together make up a calculation"""
    def __init__(self, filename=None):
        super(CalculationObject,self).__init__(type='calculation')
        self.properties['options']=[]
        self.properties['targets']=[]
        self.properties['forcefield']=None
        self.properties['result']=None

        self.properties['_expand']=True
        self.properties['_expand_targets']=True

        cwd=os.getcwd()
        os.chdir(os.path.dirname(filename))

        self.opts, self.tgt_opts = forcebalance.parser.parse_inputs(filename)
        self.properties['options'] = OptionObject(self.opts, os.path.basename(filename))

        self.properties['name'] = self.properties['options']['name']

        for target in self.tgt_opts:
            self.properties['targets'].append(TargetObject(target))

        if filename:
            self.properties['forcefield'] = ForcefieldObject(self.opts)

        os.chdir(cwd)

    def addTarget(self, opts = forcebalance.parser.tgt_opts_defaults.copy()):
        self.properties['targets'].append(TargetObject(opts))

    def writeOptions(self, filename):
        with open('~' + filename,'w') as f:
            f.write("$options\n")
            for option in self.properties['options'].opts.iterkeys():
                f.write("%s %s\n" % (option, self.properties['options'].opts[option]))
            f.write("$end\n")
            
            for target in self.properties['targets']:
                f.write("\n$target\n")
                for option in target.opts:
                    f.write("%s %s\n" % (option, target.opts[option]))
                f.write("$end\n")

    def display(self, verbose = False):
        properties = {
            'forcefield' : self.properties['forcefield'],
            'input file' : self.properties['options']['name']}

        for n, target in enumerate(self.properties['targets']):
            properties.update({"target%d" % n : target['name']})

        return properties

    def run(self):
        try:
            #options, tgt_opts = parse_inputs(input_file)
            ## The force field component of the project
            #forcefield  = FF(options)
            ## The objective function
            objective   = forcebalance.objective.Objective(self.opts, self.tgt_opts, self.properties['forcefield'].forcefield)
            ## The optimizer component of the project
            optimizer   = forcebalance.optimizer.Optimizer(self.opts, objective, self.properties['forcefield'].forcefield)
            ## Actually run the optimizer.
            optimizer.Run()
            
            
            resultopts = self.opts.copy()
            resultopts.update({"ffdir" : "result"})
            
            # temporarily silence nifty and forcefield while reading the results forcefield
            forcebalance.output.getLogger("forcebalance.forcefield").disabled = True
            forcebalance.output.getLogger("forcebalance.nifty").propagate = True
            self.properties['result'] = ForcefieldObject(resultopts)
            forcebalance.output.getLogger("forcebalance.forcefield").disabled = False
            forcebalance.output.getLogger("forcebalance.nifty").disabled = False
        except:
            import traceback
            logger = forcebalance.output.getLogger("forcebalance")
            logger.critical("Calculation failed\n%s" % traceback.format_exc())
        
# maybe the current implementation of TargetObject should be merged here
# to keep all options in the same place?
class OptionObject(ForceBalanceObject):
    """Represents general options parsed from an input file"""
    def __init__(self, opts=None, name="unknown options file"):
        super(OptionObject,self).__init__(type = 'options')

        if not opts: self.opts = forcebalance.parser.gen_opts_defaults
        else: self.opts = opts

        self.properties['name'] = name

    def display(self, verbose = False):
        options = dict()
        default_options = dict()
        for key in self.opts.iterkeys():
            if not self.isDefault(key):
                options[key]=self.opts[key]
            else: default_options[key]=self.opts[key]
        return (options, default_options)

    def resetOption(self, option):
        try:
            self.opts[option] = forcebalance.parser.gen_opts_defaults[option]
        except:
            self.opts[option] = None        

    def setOption(self, option, value):
        self.opts[option] = value

    def getOptionHelp(self, option):
        message = "No help available for option '%s'" % option

        for opt_type in forcebalance.parser.gen_opts_types:
            if option in forcebalance.parser.gen_opts_types[opt_type].keys():
                message = option + ": "
                message += forcebalance.parser.gen_opts_types[opt_type][option][2]
                message +="\nApplicable to: "
                message += forcebalance.parser.gen_opts_types[opt_type][option][3]
        return message

    def isDefault(self, option):
        return option in forcebalance.parser.gen_opts_defaults.keys() and self.opts[option] == forcebalance.parser.gen_opts_defaults[option]

class TargetObject(ForceBalanceObject):
    """Currently holds target option information. In the future, might also hold actual target objects"""
    def __init__(self, tgt_opt):
        super(TargetObject,self).__init__(type='target')
        self.opts = tgt_opt
        self.properties.update(self.opts)

    def display(self, verbose=0):
        options = dict()
        default_options = dict()
        for key in self.opts.iterkeys():
            if not self.isDefault(key):
                options[key]=self.opts[key]
            else: default_options[key]=self.opts[key]
        return (options, default_options)

    def resetOption(self, option):
        try:
            self.opts[option] = forcebalance.parser.tgt_opts_defaults[option]
        except:
            self.opts[option] = None

    def setOption(self, option, value):
        self.opts[option] = value

    def getOptionHelp(self, option):
        message = "No help available for option '%s'" % option

        for opt_type in forcebalance.parser.tgt_opts_types:
            if option in forcebalance.parser.tgt_opts_types[opt_type].keys():
                message = option + ": "
                message += forcebalance.parser.tgt_opts_types[opt_type][option][2]
                message +="\nApplicable to: "
                message += forcebalance.parser.tgt_opts_types[opt_type][option][3]
        return message

    def isDefault(self, option):
        return option in forcebalance.parser.tgt_opts_defaults.keys() and self.opts[option] == forcebalance.parser.tgt_opts_defaults[option]

class ForcefieldObject(ForceBalanceObject):
    """Represents a forcebalance forcefield object"""
    def __init__(self, opts):
        super(ForcefieldObject, self).__init__(type = 'forcefield')
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
