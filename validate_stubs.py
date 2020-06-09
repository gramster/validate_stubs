from typing import Any, List, Optional, Set, Tuple

import importlib
import inspect
import os
import sys
import types
from collections import namedtuple


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


Item = namedtuple('Item', 'file module name object_ done')


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
                items['f:' + k] = Item(fpath, name, k, v, False)
            elif inspect.isclass(v):
                members = dict()
                items['c:' + k] = members
                for kc, vc in inspect.getmembers(v):
                    if kc[0] != '_' and (inspect.isfunction(vc) or str(type(v)) == "<class 'property'>"):
                        members[kc] = Item(fpath, name, kc, vc, False)
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


def walk(tree: dict, fn, *args, delete_on_fail=False, name=None):
    if name is None:
        name=''
    to_clean = []
    for k, v in tree.items():
        if fn(name, k, v, *args) and delete_on_fail:
            to_clean.append(k)
        elif isinstance(v, dict):
            walk(v, fn, *args, delete_on_fail=delete_on_fail, name=name + '/' + k)
    for k in to_clean:
        print(f'Removing {k} from {name}')
        tree.pop(k)
        

def compare(name: str, stubpath: str):
    real, stub = import_dual(name, stubpath)
    real = gather(name, real)
    stub = gather(name, stub)
    
    # Find modules missing from stubs
    
    def has_module(path, name, node, stubs):
        if not name.startswith('m:'):
            return False
        components = path.split('/')[1:]
        components.append(name)
        for c in components:
            if c in stubs:
                stubs = stubs[c]
            else:
                modname = '.'.join([c[2:] for c in components])
                print(f"No module {modname} in stubs")
                # TODO: we could generate a skeleton stub file here
                # Remove this module from further consideration
                return True
        return False


    walk(real, has_module, stub, delete_on_fail=True)
    print(real.keys())
        

if __name__ == "__main__":
    compare('pandas', '/Users/grwheele/repos/typings')

    