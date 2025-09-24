#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include "src/jcz.h"
#include "src/sudoku_path.h"

#define SUDOKU_TOTAL 18383222420692992ULL

void unrank_permutation(long long rank, int *out) {
    int digits[9] = {0,1,2,3,4,5,6,7,8};
    int used[9] = {0};
    long long f = 1;
    for (int i = 1; i < 9; i++) f *= i;  // f = 8! = 40320

    for (int i = 0; i < 9; i++) {
        f /= (9 - i);
        int index = rank / f;
        rank %= f;

        // Find the index-th unused digit
        int count = -1;
        for (int j = 0; j < 9; j++) {
            if (!used[j]) count++;
            if (count == index) {
                out[i] = digits[j];
                used[j] = 1;
                break;
            }
        }
    }
}

static PyObject* solve_sudoku(PyObject* self, PyObject* args) {
    PyArrayObject *input_array;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_array)) {
        return NULL;
    }
    if (PyArray_TYPE(input_array) != NPY_INT) {
        return NULL;  // handle error
    }

    npy_intp size = PyArray_SIZE(input_array);

    PyArrayObject *output_array = (PyArrayObject*)PyArray_SimpleNew(1, &size, NPY_INT);

    npy_int *in = (npy_int*)PyArray_DATA(input_array);
    npy_int *out = (npy_int*)PyArray_DATA(output_array);


    char sudoku[82];
    char solved_sudoku[82];
    for(int k=0;k<size;k+=81) {
        for(int i=0;i<81;i++) sudoku[i] = '0'+in[k+i];
        
        JCZSolver(sudoku, solved_sudoku, 1);

        for (npy_intp i = 0; i < 81; ++i) {
            out[k+i] = solved_sudoku[i] - '0';
        }
    }

    PyArray_Dims new_shape;
    new_shape.ptr = PyArray_SHAPE(input_array);  // Pointer to the shape of the input array
    new_shape.len = PyArray_NDIM(input_array);         // Number of dimensions (same as input)


    PyArrayObject *reshaped_array = (PyArrayObject *)PyArray_Newshape(output_array,
        &new_shape,
        NPY_ANYORDER);
    
    return (PyObject*)reshaped_array;
}

static PyObject* generate_sudoku(PyObject* self, PyObject* args) {
    PyArrayObject *input_array;

    
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_array)) {
        return NULL;
    }
    if (PyArray_TYPE(input_array) != NPY_INT) {
        return NULL;  // handle error
    }

    npy_intp size = PyArray_SIZE(input_array);

    PyArrayObject *output_array = (PyArrayObject*)PyArray_SimpleNew(1, &size, NPY_INT);

    npy_int *in = (npy_int*)PyArray_DATA(input_array);
    npy_int *out = (npy_int*)PyArray_DATA(output_array);


    char sudoku[82];
    int perm[81] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
        50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
        60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
        70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
        80
    };
    for(int k=0;k<size;k+=81) {
        for(int i=0;i<81;i++) sudoku[i] = '0'+in[k+i];
        shuffle_array(perm, 81);
        JCZGenerate(sudoku, perm);
        
        for (npy_intp i = 0; i < 81; ++i) {
            out[k+i] = sudoku[i] >= '1' && sudoku[i] <= '9' ? sudoku[i] - '0' : 0;
        }
    }

    PyArray_Dims new_shape;
    new_shape.ptr = PyArray_SHAPE(input_array);  // Pointer to the shape of the input array
    new_shape.len = PyArray_NDIM(input_array);         // Number of dimensions (same as input)


    PyArrayObject *reshaped_array = (PyArrayObject *)PyArray_Newshape(output_array,
        &new_shape,
        NPY_ANYORDER);
    
    return (PyObject*)reshaped_array;
}

static PyObject* sudoku_path(PyObject* self, PyObject* args) {
    PyArrayObject *input_array;
    int path_length = MAX_PATH;
    
    NEGATIVETOKENS = DISPLAYLEVELS = FIRSTTOKEN = DEBUGGING = 0;
    if (!PyArg_ParseTuple(args, "O!|iiiii", &PyArray_Type, &input_array, &path_length, &NEGATIVETOKENS, &DISPLAYLEVELS, &FIRSTTOKEN, &DEBUGGING)) {
        return NULL;  // Raise TypeError automatically
    }
    if (PyArray_TYPE(input_array) != NPY_INT) {
        return NULL;  // handle error
    }
    
    npy_intp size = PyArray_SIZE(input_array);
    
    int *in = (int*)PyArray_DATA(input_array);
    int B = size / 81;

    npy_intp ntdims[3] = { B, path_length, VOCAB_SIZE };
    PyObject *next_token_data = PyArray_SimpleNew(3, ntdims, NPY_BOOL);

    npy_bool *nt_data = (npy_bool *) PyArray_DATA((PyArrayObject *)next_token_data);
    npy_intp nt_size = PyArray_SIZE(next_token_data);

    npy_intp pdims[2] = { B, path_length };
    PyObject *path_data = PyArray_SimpleNew(2, pdims, NPY_INT);


    npy_int *p_data = (npy_int *) PyArray_DATA((PyArrayObject *)path_data);
    npy_intp p_size = PyArray_SIZE(path_data);


    memset(p_data, 0, PyArray_NBYTES((PyArrayObject *)path_data));
    memset(nt_data, 0, PyArray_NBYTES((PyArrayObject *)next_token_data));

    
    char sudoku[82];
    for(int b=0;b<B;b++) {
        for(int i=0;i<81;i++) sudoku[i] = '0'+in[b*81+i];
        solve_path(sudoku);
        for(int i=0;i<path_length && i < MAX_PATH;i++) {
            for(int j=0;j<VOCAB_SIZE;j++) {
                nt_data[b*path_length*VOCAB_SIZE+i*VOCAB_SIZE+j] = next_tokens[i][j];
            }
        }
        for(int i=0;i < path_length && i < pcnt && i < MAX_PATH;i++) {
            p_data[b*path_length+i] = path[i];
        }
    }


    return Py_BuildValue("NN", path_data, next_token_data);  // No INCREF needed
}

static PyMethodDef MyMethods[] = {
    {"solve_sudoku", solve_sudoku, METH_VARARGS, "Solve a given Sudoku given as a NumPy array"},
    {"generate_sudoku", generate_sudoku, METH_VARARGS, "Generate a Sudoku Puzzle from a Board given as a NumPy array"},
    {"sudoku_path", sudoku_path, METH_VARARGS, "sudoku_path(puzzles, path_length=MAX, negative=0, levels=0, debug=0): Calculate a solution path and next labels for a given Sudoku"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mylibmodule = {
    PyModuleDef_HEAD_INIT,
    "sudoku_tools",
    "A list of utils for Solving Sudoku",
    -1,
    MyMethods
};

PyMODINIT_FUNC PyInit_sudoku_tools(void) {
    import_array();
    return PyModule_Create(&mylibmodule);
}