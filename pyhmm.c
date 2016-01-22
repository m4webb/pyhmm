#include <python2.7/Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "hmm.h"

int hmm_model_from_args(hmm_model_t *model, PyObject *args)
{
    PyObject *a_obj;
    PyObject *b_obj;
    PyObject *o_obj;
    npy_intp *dims;

    PyArg_UnpackTuple(args, "hmm_unpack", 3, 3, &a_obj, &b_obj, &o_obj);
    dims = PyArray_DIMS((PyArrayObject *)b_obj);
    model->n = dims[0];
    model->m = dims[1];
    dims = PyArray_DIMS((PyArrayObject *)o_obj);
    model->t = dims[0];
    model->a = PyArray_DATA((PyArrayObject *)a_obj);
    model->b = PyArray_DATA((PyArrayObject *)b_obj);
    model->o = PyArray_DATA((PyArrayObject *)o_obj);
    return 0;
}

static PyObject *hmm_train(PyObject *self, PyObject *args)
{
    hmm_model_t *model = malloc(sizeof(hmm_model_t));
    hmm_model_from_args(model, args);
    _hmm_train(model);
    free(model);
    Py_RETURN_NONE;
}

static PyObject *hmm_logprob(PyObject *self, PyObject *args)
{
    double logprob;
    hmm_model_t *model = malloc(sizeof(hmm_model_t));
    hmm_model_from_args(model, args);
    _hmm_logprob(model, &logprob);
    free(model);
    return PyFloat_FromDouble(logprob);
}

static PyMethodDef pyhmm_methods[] = {
    {"hmm_train", hmm_train, METH_VARARGS, "Train hmm in place. Args are a, b, o"},
    {"hmm_logprob", hmm_logprob, METH_VARARGS, "Return prob of o given a,b. Args are a, b, o"},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initpyhmm(void)
{
    Py_InitModule("pyhmm", pyhmm_methods);
    import_array();
}
