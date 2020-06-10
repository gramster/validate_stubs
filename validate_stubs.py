from typing import Any, List, Optional, Set, Tuple

import importlib
import inspect
import os
import sys
import types
from collections import namedtuple
from operator import itemgetter, attrgetter


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
    
    def __init__(self, file, module, name, object_):
        self.file = file
        self.module = module
        self.name = name
        self.object_ = object_
        self.done = False
        self.analog = None


def gather(name: str, m: Any):
    """
    Parameters:
    name: module name
    m: module object
    root: package path
    completed: a set of modules already traversed
    items: the list of discovered items
    """
    
    def _gather(mpath, name, m, root, fpath, completed, items):
        """
        Parameters:
        mpath: module path (e.g. pandas.core)
        name: module name (e.g. core)
        m: module object
        root: package path
        fpath: module path relative to root
        completed: a set of modules already traversed
        items: the list of discovered items
        """

        submodules = dict()
        for k, v in m.__dict__.items():

            if inspect.isbuiltin(v):
                continue
            try:
                # Make sure it came from this module
                if v.__dict__['__module__'] != mpath:
                    continue
            except:
                pass

            # Try to ensure the object lives below the root path and is not 
            # imported from elsewhere.
            try:
                f = inspect.getfile(v)
                if not f.startswith(root):
                    continue
            except:
                #print(f"No file for {k}")
                pass

            t = type(v)

            if k[0] == '_':
                continue
            elif inspect.ismodule(v):
                if v not in completed:
                    completed.add(v)
                    submodules[k] = v
            elif inspect.isfunction(v):
                items['f:' + k] = Item(fpath, name, k, v)
            elif inspect.isclass(v):
                members = dict()
                items['c:' + k] = members
                for kc, vc in inspect.getmembers(v):
                    if kc[0] != '_' and (inspect.isfunction(vc) or str(type(v)) == "<class 'property'>"):
                        members[kc] = Item(fpath, name, kc, vc)
            else:
                pass

        for k, v in submodules.items():
            try:        
                fpath = inspect.getfile(v)  # v.__dict__['__file__']
                if not fpath.startswith(root):
                    continue
                fpath = fpath[len(root)+1:]
                members = dict()
                items['m:' + k] = members 
                _gather(name + '.' + k, k, v, root, fpath, completed, members)
            except:
                pass

    fpath = m.__dict__['__file__']
    root = fpath[:fpath.rfind('/')]  # fix for windows
    members = dict()
    items = {'m:' + name: members}
    _gather(name, name, m, root, fpath, set(), members)
    return items


def walk(tree: dict, fn, *args, postproc=None, path=None):
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
        elif isinstance(v, dict):
            walk(v, fn, *args, postproc=postproc, path=path + '/' + k)
    if postproc:
        postproc(tree, to_postproc)
        

def collect_items(root):

    def _collect(path, name, node, functions, classes):
        if name.startswith('c:'):
            classes.append(node)
            return True  # Don't recurse
        elif name.startswith('f:'):
            functions.append(node)

    functions = []
    classes = []
    walk(root, _collect, functions, classes)
    functions = sorted(functions, key=attrgetter('name'))
    #classes = sorted(classes, key=attrgetter('name'))
    return functions, classes


def match_pairs(real, stub, label, owner=''):
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


def compare_functions(real, stub, owner=None):
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
        real_functions, _ = collect_items(a)
        stub_functions, _ = collect_items(s)
        compare_functions(real_functions, stub_functions, s.name)


def compare(name: str, stubpath: str):
    real, stub = import_dual(name, stubpath)
    real = gather(name, real)
    stub = gather(name, stub)

    # First print out all the modules in real package where
    # we don't have a matching module in the stub.

    def has_module(path, name, node, stubs):
        if not name.startswith('m:'):
            return
        components = path.split('/')[1:]
        components.append(name)
        for c in components:
            if c in stubs:
                stubs = stubs[c]
            else:
                modname = '.'.join([c[2:] for c in components])
                print(f"No module {modname} in stubs")

    walk(real, has_module, stub)

    # Collect all top-level functions and then print out the
    # ones that don't have analogs in the stubs, and vice-versa.

    real_functions, real_classes = collect_items(real)
    stub_functions, stub_classes = collect_items(stub)
    compare_functions(real_functions, stub_functions)
    #compare_classes(real_classes, stub_classes)

    # TODO: if real code has type hints should compare with stubs

    # Get the docstrings and report mismatches
    # TODO
        

if __name__ == "__main__":
    compare('pandas', '/Users/grwheele/repos/typings')

    