from typing import Any, Callable, List, Optional, Set, Tuple

import importlib
import inspect
import os
import sys
import types
from collections import namedtuple
from operator import itemgetter, attrgetter
from enum import Enum


def import_dual(m: str, stub_path: str) -> Tuple:
    """ 
    Import both a stub package and a real package with the same name.
    
    Parameters:
    
    m (str): module name
    stub_path (str): location of type stubs
    
    Returns:
    Tuple - the tuple of (real module, stub module)
    
    """
    
    def _clean(m):
        to_del = [k for k in sys.modules.keys() if k == m or k.startswith(m + '.')]
        for k in to_del:
            del sys.modules[k]
        importlib.invalidate_caches()
    
    _clean(m)
        
    m1 = importlib.import_module(m)
    
    _clean(m)
    
    sys.path_hooks.insert(0,
        importlib.machinery.FileFinder.path_hook(
            (importlib.machinery.SourceFileLoader, ['.pyi']))
    )
    sys.path.insert(0, stub_path)
    
    try:
        m2 = importlib.import_module(m)
        return m1, m2
    finally:
        sys.path.pop(0)
        sys.path_hooks.pop(0)
        _clean(m)


class Item:
    
    class ItemType(Enum):
        MODULE = 1
        CLASS = 2
        FUNCTION = 3
        PROPERTY = 4

    def __init__(self, file: str, module: str, name: str, object_: object, type_: ItemType, children: dict=None):
        self.file = file
        self.module = module
        self.name = name
        self.object_ = object_
        self.type_ = type_
        self.children = children
        self.done = False
        self.analog = None

    def ismodule(self):
        return self.type_ == Item.ItemType.MODULE

    def isclass(self):
        return self.type_ == Item.ItemType.CLASS

    def isfunction(self):
        return self.type_ == Item.ItemType.FUNCTION

    @staticmethod
    def make_function(file: str, module: str, name: str, object_: object):
        return Item(file, module, name, object_, Item.ItemType.FUNCTION)

    @staticmethod
    def make_class(file: str, module: str, name: str, object_: object, children:dict):
        return Item(file, module, name, object_, Item.ItemType.CLASS, children)

    @staticmethod
    def make_module(file: str, module: str, name: str, object_: object, children:dict):
        return Item(file, module, name, object_, Item.ItemType.MODULE, children)


def isfrompackage(v: object, path: str) -> bool:
    # Try to ensure the object lives below the root path and is not 
    # imported from elsewhere.
    try:
        f = inspect.getfile(v)
        return f.startswith(path)
    except TypeError:  # builtins or non-modules; for the latter we return True for now
        return not inspect.ismodule(v)


def isfrommodule(v: object, module: str, default: bool=True) -> bool:
    try:
        # Make sure it came from this module
        return v.__dict__['__module__'] == module
    except:
        return default


def gather(name: str, m: object) -> Item:
    """
    Parameters:
    name: module name
    m: module object
    root: package path
    completed: a set of modules already traversed
    items: the list of discovered items
    """
    
    def _gather(mpath: str, name: str, m: object, root: str, fpath: str, completed: set, items: dict):
        """
        Parameters:
        mpath: module path (e.g. pandas.core)
        name: module name (e.g. core)
        m: module object
        root: package path
        fpath: module path relative to root
        completed: a set of modules already traversed
        items: the dict of discovered items
        """

        for k, v in m.__dict__.items():

            if not (inspect.isclass(v) or inspect.isfunction(v) or inspect.ismodule(v)):
                continue
            if inspect.isbuiltin(v) or k[0] == '_' or not isfrompackage(v, root) or not isfrommodule(v, mpath):
                continue

            if inspect.ismodule(v):
                if v not in completed:
                    completed.add(v)
                    mpath = inspect.getfile(v)
                    if mpath.startswith(root):
                        mpath = mpath[len(root)+1:]
                        members = dict()
                        items[k] = Item.make_module(mpath, name, k, v, members) 
                        _gather(name + '.' + k, k, v, root, mpath, completed, members)
            elif inspect.isfunction(v):
                items[k] = Item.make_function(fpath, name, k, v)
            elif inspect.isclass(v):
                members = dict()
                items[k] = Item.make_class(fpath, name, k, v, members)
                for kc, vc in inspect.getmembers(v):
                    if kc[0] != '_' and (inspect.isfunction(vc) or str(type(vc)) == "<class 'property'>"):
                        members[kc] = Item.make_function(fpath, name, kc, vc)
            else:
                pass

    fpath = m.__dict__['__file__']
    root = fpath[:fpath.rfind('/')]  # fix for windows
    members = dict()
    package = Item.make_module(fpath, '', name, m, members)
    _gather(name, name, m, root, fpath, set(), members)
    return package


def walk(tree: dict, fn: Callable, *args, postproc: Callable=None, path=None):
    """
    Walk the object tree and apply a function.
    If the function returns True, do not walk its children,
    but add the object to a postproc list and if a postproc function
    is provided, call that at the end for those objects. This gives
    us some flexibility in both traversing the tree and collecting
    and processing certain nodes.
    """
    if path is None:
        path = ''
    to_postproc = []
    for k, v in tree.items():
        if fn(path, k, v, *args):
            to_postproc.append(k)
        elif v.children:
            walk(v.children, fn, *args, postproc=postproc, path=path + '/' + k)
    if postproc:
        postproc(tree, to_postproc)
        

def collect_items(root: Item) -> Tuple[List[Item], List[Item]]:

    def _collect(path, name, node, functions, classes):
        if node.isclass():
            classes.append(node)
            return True  # Don't recurse
        elif node.isfunction():
            functions.append(node)

    functions = []
    classes = []
    walk(root.children, _collect, functions, classes)
    functions = sorted(functions, key=attrgetter('name'))
    classes = sorted(classes, key=attrgetter('name'))
    return functions, classes


def match_pairs(real: List[Item], stub: List[Item], label: str, owner: str=''):
    i_r = 0
    i_s = 0
    while i_r < len(real) or i_s < len(stub):
        if i_r == len(real) or (i_s < len(stub) and real[i_r].name > stub[i_s].name):
            fn = stub[i_s]
            print(f"No match for stub {label} {fn.module}.{owner}{fn.name}")
            i_s += 1
        elif i_s == len(stub) or real[i_r].name < stub[i_s].name:
            fn = real[i_r]
            print(f"No stub for {label} {fn.module}.{owner}{fn.name}")
            i_r += 1
        else:
            # TODO: Check for uniqueness
            stub[i_s].analog = real[i_r]            
            real[i_r].analog = stub[i_s]
            i_s += 1
            i_r += 1


def compare_functions(real: List[Item], stub: List[Item], owner: Optional[str]=None):
    if owner is None:
        owner = ''
    else:
        owner += '.'
    match_pairs(real, stub, 'function', owner)

    # For the functions that do have analogs, compare the 
    # signatures.
    i_s = 0
    while i_s < len(stub):
        s = stub[i_s]
        a = s.analog
        if a:
            try:
                sc = s.object_.__code__.co_argcount
                ac = a.object_.__code__.co_argcount
                if sc != ac:
                    print(f"Mismatched argument count for {s.module}.{owner}{s.name}: stub has {sc} but real has {ac}")
                else:
                    sa = s.object_.__code__.co_varnames
                    aa = a.object_.__code__.co_varnames
                    if sa != aa:
                        print(f"Mismatched argument names for {s.module}.{owner}{s.name}: stub has {sa} but real has {aa}")
                    else:
                         print(f"{s.module}.{owner}{s.name} passes argument checks")
            except Exception as e:
                print(f"Failed to validate {s.module}.{owner}{s.name}: {e}")

        i_s += 1


def compare_classes(real, stub, owner=None):
    match_pairs(real, stub, 'class')
    # For the classes that do have analogs, compare the 
    # methods.
    i_s = 0
    while i_s < len(stub):
        s = stub[i_s]
        a = s.analog
        if a:
            real_functions, _ = collect_items(a)
            stub_functions, _ = collect_items(s)
            compare_functions(real_functions, stub_functions, s.name)
        i_s += 1


def compare(name: str, stubpath: str):
    real, stub = import_dual(name, stubpath)
    real = gather(name, real)
    stub = gather(name, stub)

    # First print out all the modules in real package where
    # we don't have a matching module in the stub.

    def has_module(path, name, node, stubs):
        if not node.ismodule():
            return
        components = path.split('/')[1:]
        components.append(name)
        for c in components:
            if c in stubs.children:
                stubs = stubs.children[c]
            else:
                modname = '.'.join(components)
                print(f"No module {modname} in stubs")
                break

    walk(real.children, has_module, stub)

    # Collect all top-level functions and then print out the
    # ones that don't have analogs in the stubs, and vice-versa.

    real_functions, real_classes = collect_items(real)
    stub_functions, stub_classes = collect_items(stub)
    compare_functions(real_functions, stub_functions)
    compare_classes(real_classes, stub_classes)

    # TODO: if real code has type hints should compare with stubs

    # Get the docstrings and report mismatches
    # TODO
        

if __name__ == "__main__":
    compare('pandas', '/Users/grwheele/repos/typings')

    