import os
import subprocess
from cffi import FFI

# Determine shared library extension based on OS
lib_ext = "dll" if os.name == "nt" else "so"
lib_name = f"libjcz_solver.{lib_ext}"

# Compile the C file into a shared library if not already compiled
if not os.path.exists(lib_name):
    compile_command = [
        "gcc", "-O3", "-shared", "-o", lib_name, "-fPIC", "JCZSolve.c"
    ]
    subprocess.run(compile_command, check=True)

# Set up FFI
ffi1 = FFI()
ffi1.cdef("""
    int JCZSolver(const char *puzzle, char *solution, int limit);
    void JCZGenerate(char *puzzle, int *perm);
""")
lib = ffi1.dlopen(f"./{lib_name}")

# Puzzle generation function
def puzzle_generate(inp, p):
    if not isinstance(inp, str):
        inp = "".join(map(str, inp.tolist()))
    puzzle = ffi1.new("char[]", inp.encode('utf-8'))
    perm = ffi1.new("int[]", list(p))
    lib.JCZGenerate(puzzle, perm)
    return ffi1.string(puzzle).decode('utf-8')
