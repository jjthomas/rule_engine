#!/bin/bash
python3 -c 'import pyarrow; pyarrow.create_library_symlinks()'
INC=$(python3 -c 'import pyarrow; print(pyarrow.get_include())')
LIB=$(python3 -c 'import pyarrow; print(pyarrow.get_library_dirs()[0])')
g++ -I$INC -I/usr/include/python3.6m -fPIC cube.cpp -shared -o libcube.so -L$LIB -larrow -larrow_python

