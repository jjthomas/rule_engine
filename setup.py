import subprocess
import os
import shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class LibExtension(Extension):
  def __init__(self, name, sourcedir=''):
    Extension.__init__(self, name, sources=[])
    self.sourcedir = os.path.abspath(sourcedir)

class LibBuild(build_ext):
  def run(self):
    assert len(self.extensions) == 1
    ext = self.extensions[0]
    extdir = \
        os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
    extdir = os.path.join(extdir, "rule_engine")
    if not os.path.isdir(extdir):
      os.mkdir(extdir)
    subprocess.check_call(["./build.sh"])
    shutil.copy("rule_engine/librule", extdir)

setup(
  name='rule_engine',
  version='0.1',
  url='https://github.com/jjthomas/rule_engine',
  license='BSD',
  maintainer='James Thomas',
  description='Anomaly classification with decision rules',
  packages=[
    "rule_engine"
  ],
  install_requires=[
    "pandas",
    "pyarrow",
  ],
  ext_modules=[LibExtension('librule')],
  cmdclass=dict(build_ext=LibBuild),
  python_requires='>=3.6'
)
