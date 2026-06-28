#!/bin/bash
# Build and run jasna-mojo
# Usage: ./run.sh --input input.mp4 --output output.mp4 --device mps

export MOJO_PYTHON_LIBRARY="/opt/homebrew/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/lib/libpython3.13.dylib"
export PATH="/Users/elife/py313/bin:$PATH"

# Build if binary doesn't exist
if [ ! -f jasna_bin ]; then
    echo "Building jasna_bin..."
    ~/py313/bin/mojo build jasna/__main__.mojo -o jasna_bin
fi

./jasna_bin "$@"
