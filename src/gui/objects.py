import forcebalance
from uuid import uuid1 as uuid

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