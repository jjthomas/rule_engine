#!/bin/bash
python3 -c 'import pyarrow; pyarrow.create_library_symlinks()'
INC=$(python3 -c 'import pyarrow; print(pyarrow.get_include())')
LIB=$(python3 -c 'import pyarrow; print(pyarrow.get_library_dirs()[0])')
PY_INC=$(python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")
PY_LIB_NAME=$(basename $PY_INC)
PY_LIB=$(dirname $(python3 -c "from sysconfig import get_paths as gp; print(gp()['stdlib'])"))
g++ -std=c++11 -I$INC -I$PY_INC -fPIC cube.cpp -shared -o libcube -L$LIB -L$PY_LIB -larrow -larrow_python -l$PY_LIB_NAME

