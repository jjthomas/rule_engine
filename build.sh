#!/bin/bash
python3 -c 'import pyarrow; pyarrow.create_library_symlinks()'
INC=$(python3 -c 'import pyarrow; print(pyarrow.get_include())')
LIB=$(python3 -c 'import pyarrow; print(pyarrow.get_library_dirs()[0])')
PY_INC=$(python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")
if [[ "$OSTYPE" == "darwin"* ]]; then
  PY_LIB_NAME=$(basename $PY_INC)
  PY_LIB_NAME="-l$PY_LIB_NAME"
  PY_LIB=$(dirname $(python3 -c "from sysconfig import get_paths as gp; print(gp()['stdlib'])"))
  PY_LIB="-L$PY_LIB"
fi
g++ -std=c++11 -I$INC -I$PY_INC -fPIC cube.cpp -shared -o libcube -L$LIB $PY_LIB -larrow -larrow_python $PY_LIB_NAME

