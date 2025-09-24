#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "src/sat_path.h"


static PyObject* sat_path(PyObject* self, PyObject* args) {
    int B;
    GeneratorParameters params;
    PathParameters path_params;

    DEBUGGING = 0;
    if (!PyArg_ParseTuple(args, "iii|iiiii", 
            &B, &params.N, &params.M, 
            &path_params.length, 
            &params.planted, 
            &path_params.displaylevels, 
            &path_params.firsttoken, 
            &DEBUGGING)) {
        return NULL;  // Raise TypeError automatically
    }
    
    //print_instance(&p);

    npy_intp ntdims[3] = { B, path_params.length, VOCAB_SIZE };
    PyObject *next_token_data = PyArray_SimpleNew(3, ntdims, NPY_BOOL);

    npy_bool *nt_data = (npy_bool *) PyArray_DATA((PyArrayObject *)next_token_data);
    //npy_intp nt_size = PyArray_SIZE(next_token_data);

    npy_intp pdims[2] = { B, path_params.length };
    PyObject *path_data = PyArray_SimpleNew(2, pdims, NPY_INT);


    npy_int *p_data = (npy_int *) PyArray_DATA((PyArrayObject *)path_data);
    //npy_intp p_size = PyArray_SIZE(path_data);


    memset(p_data, 0, PyArray_NBYTES((PyArrayObject *)path_data));
    memset(nt_data, 0, PyArray_NBYTES((PyArrayObject *)next_token_data));

    
    for(int b=0;b<B;b++) {
        Problem p = generate_instance(&params);
        solve_path(&p, &path_params);
        for(int i=0;i<path_params.length && i < MAX_PATH;i++) {
            for(int j=0;j<VOCAB_SIZE;j++) {
                nt_data[b*path_params.length*VOCAB_SIZE+i*VOCAB_SIZE+j] = next_tokens[i][j];
            }
        }
        for(int i=0;i < path_params.length && i < pcnt && i < MAX_PATH;i++) {
            p_data[b*path_params.length+i] = path[i];
        }
    }

    return Py_BuildValue("NN", path_data, next_token_data);  // No INCREF needed
}

static PyMethodDef MyMethods[] = {
    {"sat_path", sat_path, METH_VARARGS, "sat_path(Batch, N, M, path_length=MAX, planted=0, levels=0, firsttoken=0, debug=0): Calculate a solution path and next labels for a given sat"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mylibmodule = {
    PyModuleDef_HEAD_INIT,
    "sat_tools",
    "A list of utils for Solving sat",
    -1,
    MyMethods
};

PyMODINIT_FUNC PyInit_sat_tools(void) {
    import_array();
    return PyModule_Create(&mylibmodule);
}
