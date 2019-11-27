from __future__ import print_function
from argparse import ArgumentParser
#from text_utils import print_dict

def print_dict(dic, sorted_keys=True, end='\n'):
    if sorted_keys:
        keys = sorted(dic.keys())
    else:
        keys = dic.keys()
    for k in keys:
        print(k, dic[k], end=end)

class ArgParser(object):
    def __init__(self, ps=None, name=None):
        if ps is None:
            self._ps = ArgumentParser()
        elif isinstance(ps, ArgumentParser):
            self._ps = ps
        elif isinstance(ps, ArgParser):
            self._ps = ps._ps
        else:
            raise TypeError("Invalid type of ps: %s" % str(type(ps)))
        if isinstance(name, str):
            self._g = self._ps.add_argument_group(name)
        else:
            self._g = self._ps
        self._args = None

    def add(self, *args, **kwargs):
        assert self._args == None
        self._g.add_argument(*args, **kwargs)

    def add_flag(self, name, *args, **kwargs):
        assert self._args == None
        self.add(name, *args, action="store_true", default=False, **kwargs)

    def parse_args(self, *args, **kwargs):
        if self._args is None:
           self._args = self._ps.parse_args(*args, **kwargs)
        return vars(self._args)

    def print_args(self):
        if self._args is None:
            print('Use option --help')
        else:
            print_dict(vars(self._args))
