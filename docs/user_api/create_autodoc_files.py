# coding: utf-8
from pathlib import Path

files = '''    zfit.minimize.WrapOptimizer
    zfit.minimize.Adam
    zfit.minimize.Minuit
    zfit.minimize.Scipy
    zfit.minimize.BFGS
    zfit.minimize.DefaultStrategy
    zfit.minimize.DefaultToyStrategy
    zfit.minimize.FitResult'''
files = [file.strip() for file in files.split('\n')]
cwd = Path.cwd()
file_template = '''{}
{}

.. autoclass:: {}
    :members:
'''
for file in files:
    p = cwd / '{}.rst'.format(file)
    class_name = file.split('.')[-1]
    underline = '-'*len(class_name)
    content = file_template.format(class_name, underline, file)
    with p.open('w') as f:
        f.write(content)
