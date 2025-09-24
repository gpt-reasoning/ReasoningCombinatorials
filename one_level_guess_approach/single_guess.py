"""
START = 100
GUESS = 101
FAIL = 102
END = 0
"""

import os
import subprocess
from cffi import FFI
import numpy as np
import torch

# Determine shared library extension based on OS
lib_ext = ".dll" if os.name == "nt" else ".so"
lib_name = f"sudoku_guess{lib_ext}"

# Compile C source if shared object is not already built
if not os.path.exists(lib_name):
    compile_command = [
        "gcc", "-O3", "-shared", "-o", lib_name, "-fPIC", "sudoku1.c"
    ]
    subprocess.run(compile_command, check=True)

# Setup FFI
ffi = FFI()
ffi.cdef("int* solve_path(char *puzzle, char *labels);")
solver = ffi.dlopen(f"./{lib_name}")

# Solver wrapper
def solve_path(sudoku):
    l = len([p for p in sudoku if '1' <= p <= '9'])
    puzzle = ffi.new("char[]", (sudoku + "0").encode('utf-8'))
    labels = ffi.new("char[]", ("0" * 8300).encode('utf-8'))
    path_pointer = solver.solve_path(puzzle, labels)

    # Read path until 0
    path = []
    for i in range(85):
        x = path_pointer[i]
        if x == 0:
            break
        path.append(x)

    # Parse labels and build one-hot tensor
    s = ffi.string(labels)
    y = np.frombuffer(s, dtype=np.uint8) - ord('0')
    y = torch.tensor(y).long()
    Y = torch.nn.functional.one_hot(y, num_classes=10)
    Y[..., 0] = 0
    Y[(len(path) - 1) * 100, 0] = 1

    return l, torch.tensor(path + [0] * (83 - len(path))), Y.view(-1, 1000)
