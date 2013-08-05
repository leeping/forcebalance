from collections import OrderedDict
from forcebalance.parser import tgt_opts_defaults, gen_opts_defaults

class ForceBalanceBaseClass(object):
    """ Provides some nifty functions that are common to all ForceBalance classes. """

    def __init__(self, options):
        self.PrintOptionDict  = OrderedDict()
        self.verbose_options  = options['verbose_options']
        
    def set_option(self, in_dict, src_key, dest_key = None, val = None, default = None, forceprint=False):
        if dest_key == None:
            dest_key = src_key
        if val == None:
            val = in_dict[src_key]
        if default == None:
            if src_key in gen_opts_defaults: 
                default = gen_opts_defaults[src_key]
            elif src_key in tgt_opts_defaults:
                default = tgt_opts_defaults[src_key]
            else: default = None
        if ((val != default or self.verbose_options) and dest_key != 'root') or forceprint:
            self.PrintOptionDict[dest_key] = val
        return super(ForceBalanceBaseClass,self).__setattr__(dest_key, val)
