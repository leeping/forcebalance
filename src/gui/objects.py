import forcebalance
from uuid import uuid1 as uuid
import os

class ForceBalanceObject(object):
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
    def __init__(self, filename=None):
        super(CalculationObject,self).__init__()
        self.properties['options']=[]
        self.properties['targets']=[]
        self.properties['forcefield']=None

        self.properties['_expand']=True
        self.properties['_expand_targets']=True

        cwd=os.getcwd()
        os.chdir(os.path.dirname(filename))

        opts, tgt_opts = forcebalance.parser.parse_inputs(filename)
        self.properties['options'] = OptionObject(opts, os.path.basename(filename))

        for target in tgt_opts:
            self.properties['targets'].append(TargetObject(target))

        if filename:
            self.properties['forcefield'] = ForcefieldObject(opts)

        os.chdir(cwd)

        

#        objtype = type(obj)

#        if objtype==objects.OptionObject:
#            newcalc = {'options':obj,'targets':[],'forcefields':[]}
#            self.calculations.append(newcalc)

#        elif objtype==objects.TargetObject: 
#            self.calculations[-1]['targets'].append(obj)
#        elif objtype==objects.ForcefieldObject:
#            self.calculations[-1]['forcefields'].append(obj)
#        else: raise TypeError("ObjectViewer can only handle option, target, and forcefield objects")
########################################3
#        if filename=='': return
#
#        cwd=os.getcwd()
#        os.chdir(os.path.dirname(filename))
#
#        self.add(objects.OptionObject(opts, os.path.split(filename)[1]))
#        for target in tgt_opts:
#            self.add(objects.TargetObject(target))
#        self.add(objects.ForcefieldObject(opts))
#        self.update()
#        
#        os.chdir(cwd)

class TargetObject(ForceBalanceObject):
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
            s+= "\n--- default options ---\n"
            for default_key in default_keys:
                s+= "%s : %s\n" % (default_key, str(self.opts[default_key]))
        
        return s
        

class OptionObject(ForceBalanceObject):
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
            s+= "\n--- default options ---\n"
            for default_key in default_keys:
                s+= "%s : %s\n" % (default_key, str(self.opts[default_key]))
        return s
        

class ForcefieldObject(ForceBalanceObject):
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