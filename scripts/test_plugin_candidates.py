"""Run bounded offline candidate tests with isolated fake data."""
import ctypes
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / 'data' / 'plugin-evaluation'
if os.name == 'nt':
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '4'
temp_root = STAGING / 'fixture-tmp'
temp_root.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(temp_root)
os.environ['WORKBUDDY_TEST_TMP'] = str(temp_root)
real_rmtree = shutil.rmtree
def checked_rmtree(path, *args, **kwargs):
    target = Path(path).resolve()
    if not target.is_relative_to(temp_root.resolve()) or target == temp_root.resolve():
        raise RuntimeError('Refusing to remove path outside isolated fixture directory')
    return real_rmtree(path, *args, **kwargs)
shutil.rmtree = checked_rmtree

def module(name, path, package=False):
    kw = {'submodule_search_locations':[str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **kw)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

kind=sys.argv[1]
if kind == 'workbuddy':
    home=STAGING/'workbuddy'
    module('astrbot_plugin_workbuddy',home/'__init__.py',True)
    names=['test_storage','test_scheduler']
elif kind == 'lingxi':
    home=STAGING/'lingxi'
    sys.path.insert(0,str(home))
    names=['test_prompt_guards','test_prompt_assembler','test_gates_default']
else:
    raise SystemExit('unknown candidate')
suite=unittest.TestSuite()
for name in names:
    test=module('isolated_'+name,home/'tests'/f'{name}.py')
    suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(test))
result=unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
