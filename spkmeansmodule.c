#define PY_SSIZE_T_CLEAN
#include <Python.h>       
#include <math.h>         
#include "spkmeans.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

typedef struct eigenVector {
    double eigenValue;
    int index;
} eigenVector; 

/* Global variables */
int clusters_num; /* make sure clusters_num == K*/
int k;
int vector_num;
int vector_len;
int max_iter;
double **vector_list;
double** weightedAdjMatrix;
double** diagDegMatrix;
double** LnormMatrix;
double **VMat;
double **UMat;
double **TMat;
double* eigenValues;
int vector_num;
int vector_len;
eigenVector *eigenVectors;
double **centroids;
double** clusters;
int *clustersindexes;




static PyObject* fit(PyObject *self, PyObject *args) 
{
    PyObject * centroids_list;
    PyObject * origin_vector_list;
    Py_ssize_t m;
    Py_ssize_t n;
    int max_iter;
    Py_ssize_t i,j;

    if (!PyArg_ParseTuple(args, "O!O!ii", &PyList_Type, &centroids_list, &PyList_Type, &origin_vector_list, &max_iter, &clusters_num)) {
        return NULL;
    }

    n = PyList_Size(PyList_GetItem(centroids_list, 0)); /* vector_len*/
    vector_len = (int) n;
    
    centroids = (double**) calloc(clusters_num, n*sizeof(double));
    assertmsg(centroids != NULL);
    for ( i=0; i<clusters_num; i++) {
        centroids[i] = (double*) calloc(n,sizeof(double));
        assertmsg(centroids[i] != NULL);
    }
    

    for (i = 0; i < clusters_num; i++){
        for (j=0; j < n; j++) {
            centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(centroids_list, i), j)); /*CONVERSION*/
        }
    }

    m = PyList_Size(origin_vector_list); /* vector_num */
    vector_num = (int) m;
    

    vector_list = (double**) calloc(m, n*sizeof(double));
    assertmsg(vector_list != NULL);
    for ( i=0; i<m; i++) {
        vector_list[i] = (double*) calloc(n,sizeof(double));
        assertmsg(vector_list[i] != NULL);
    }

    for (i = 0; i < m; i++){
        for (j=0; j < n; j++) {
            vector_list[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(origin_vector_list, i), j)); /*CONVERSION*/
        }
    } /* Now global vector_list is up to date */

    calccentroids(max_iter);

    PyObject *output_centroids = PyList_New(0);
    for (i = 0; i<clusters_num; i++) {
        PyObject * centroid = PyList_New(0);

        for(j=0; j<n;j++) {
            PyList_Append(centroid, Py_BuildValue("d",centroids[i][j]));
        }

        PyList_Append(output_centroids, centroid);
    }
    freearray(vector_list,vector_num);
    freearray(centroids,k);
    return output_centroids;
}








static PyObject* WeightedAdjacencyMatrix(PyObject *self, PyObject *args)
{
    PyObject * origin_vector_list;
    Py_ssize_t m;
    Py_ssize_t n;
    Py_ssize_t i,j;
    if(!PyArg_ParseTuple(args, "O!ii", &PyList_Type, &origin_vector_list, &vector_num, &vector_len)) {
        return NULL;
    }

    n = PyList_Size(PyList_GetItem(origin_vector_list, 0));
    m = PyList_Size(origin_vector_list);
    vector_list = (double**) calloc(m, n*sizeof(double));
    assertmsg(vector_list != NULL);
    for ( i=0; i<m; i++) {
        vector_list[i] = (double*) calloc(n,sizeof(double));
        assertmsg(vector_list[i] != NULL);
    }

    for (i = 0; i < m; i++){
        for (j=0; j < n; j++) {
            vector_list[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(origin_vector_list, i), j)); /*CONVERSION*/
        }
    } /* Now global vector_list is up to date */


    weightedAdjMat();

    PyObject *output_matrix = PyList_New(0);
    for (i = 0; i<vector_num; i++) {
        PyObject * centroid = PyList_New(0);

        for(j=0; j<vector_num;j++) {
            PyList_Append(centroid, Py_BuildValue("d",weightedAdjMatrix[i][j]));
        }

        PyList_Append(output_matrix, centroid);
    }
    freearray(vector_list,vector_num);
    freearray(weightedAdjMatrix,vector_num);
    
    return output_matrix;

}




static PyObject* DiagonalDegreeMatrix(PyObject *self, PyObject *args)
{
    PyObject * origin_matrix;
    Py_ssize_t m;
    Py_ssize_t n;
    Py_ssize_t i,j;
    if(!PyArg_ParseTuple(args, "O!ii", &PyList_Type, &origin_matrix, &vector_num, &vector_len)) {
        return NULL;
    }

    n = PyList_Size(PyList_GetItem(origin_matrix, 0));
    m = PyList_Size(origin_matrix);
    weightedAdjMatrix = (double**) calloc(m, n*sizeof(double));
    assertmsg(weightedAdjMatrix != NULL);
    for ( i=0; i<m; i++) {
        weightedAdjMatrix[i] = (double*) calloc(n,sizeof(double));
        assert(weightedAdjMarixt[i] != NULL); /* notice assert not assertmsg because of compiler */
    }

    for (i = 0; i < m; i++){
        for (j=0; j < n; j++) {
            weightedAdjMatrix[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(origin_matrix, i), j)); /*CONVERSION*/
        }
    } /* Now global weightedAdjMat is up to date */

    diagDegMat();

    PyObject *output_matrix = PyList_New(0);
    for (i = 0; i<vector_num; i++) {
        PyObject * centroid = PyList_New(0);

        for(j=0; j<vector_num;j++) {
            PyList_Append(centroid, Py_BuildValue("d",diagDegMatrix[i][j]));
        }

        PyList_Append(output_matrix, centroid);
    }
    freearray(weightedAdjMatrix,vector_num);
    freearray(diagDegMatrix,vector_num);

    return output_matrix;

}






static PyObject* NormalizedGraphLaplacian(PyObject *self, PyObject *args)
{
    PyObject * origin_vector_list;
     Py_ssize_t m;
    Py_ssize_t n;
    Py_ssize_t i,j;
    if(!PyArg_ParseTuple(args, "O!ii", &PyList_Type, &origin_vector_list, &vector_num, &vector_len)) {
        return NULL;
    }

    n = PyList_Size(PyList_GetItem(origin_vector_list, 0));
    m = PyList_Size(origin_vector_list);
    vector_list = (double**) calloc(m, n*sizeof(double));
    assertmsg(vector_list != NULL);
    for (i = 0; i < m; i++) {
        vector_list[i] = (double*) calloc(n,sizeof(double));
        assertmsg(vector_list[i] != NULL);
    }

    for (i = 0; i < m; i++){
        for (j=0; j < n; j++) {
            vector_list[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(origin_vector_list, i), j)); /*CONVERSION*/
        }
    }

    Lnorm();

    PyObject *output_matrix = PyList_New(0);
    for (i = 0; i<vector_num; i++) {
        PyObject * centroid = PyList_New(0);

        for(j=0; j<vector_num;j++) {
            PyList_Append(centroid, Py_BuildValue("d",LnormMatrix[i][j]));
        }

        PyList_Append(output_matrix, centroid);
    }
    freearray(vector_list,vector_num);
    freearray(weightedAdjMatrix,vector_num);
    freearray(diagDegMatrix,vector_num);
    freearray(LnormMatrix,vector_num);
    
    return output_matrix;

}


static PyObject* Jacobi(PyObject *self, PyObject *args)
{
    PyObject * origin_vector_list;
    Py_ssize_t m;
    Py_ssize_t n;
    Py_ssize_t i,j;
    double** A;
    if(!PyArg_ParseTuple(args, "O!ii", &PyList_Type, &origin_vector_list, &vector_num, &vector_len)) {
        return NULL;
    }

    n = PyList_Size(PyList_GetItem(origin_vector_list, 0));
    m = PyList_Size(origin_vector_list);
    A = (double**) calloc(m, n*sizeof(double));
    assertmsg(A != NULL);
    for ( i=0; i<m; i++) {
        A[i] = (double*) calloc(n,sizeof(double));
        assertmsg(A[i] != NULL);
    }

    for (i = 0; i < m; i++){
        for (j=0; j < n; j++) {
            A[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(origin_vector_list, i), j)); /*CONVERSION*/
        }
    }
    /*A is transfered correctly as vector_list */

    calcJacobi(A);
    
    PyObject *output_values = PyList_New(0);

    for(j=0; j<vector_num;j++) {
        PyList_Append(output_values, Py_BuildValue("d",eigenValues[j]));
        }

    transpMat(VMat);
    PyObject *output_vectors = PyList_New(0);
    for (i = 0; i<vector_num; i++) {
        PyObject * centroid = PyList_New(0);

        for(j=0; j<vector_num;j++) {
            PyList_Append(centroid, Py_BuildValue("d",VMat[i][j]));
        }

        PyList_Append(output_vectors, centroid);
    }

    freearray(A,(int)m);
    freearray(VMat,vector_num);
    free(eigenValues);

    return Py_BuildValue("OO", output_values, output_vectors);

}

static PyObject* heuristic(PyObject *self, PyObject *args)
{
    int newK;
    Py_ssize_t m;
    Py_ssize_t n;
    Py_ssize_t i,j;
    PyObject * origin_vector_list;
    if(!PyArg_ParseTuple(args, "O!ii", &PyList_Type, &origin_vector_list, &vector_num, &vector_len)) {
        return NULL;
    }

    n = PyList_Size(PyList_GetItem(origin_vector_list, 0));
    m = PyList_Size(origin_vector_list);
    vector_list = (double**) calloc(m, n*sizeof(double));
    assertmsg(vector_list != NULL);
    for ( i=0; i<m; i++) {
        vector_list[i] = (double*) calloc(n,sizeof(double));
        assertmsg(vector_list[i] != NULL);
    }

    for (i = 0; i < m; i++){
        for (j=0; j < n; j++) {
            vector_list[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(origin_vector_list, i), j)); /*CONVERSION*/
        }
    }

    newK = eigengapHeuristic();
    freearray(vector_list,(int)m);
    freearray(weightedAdjMatrix,vector_num);
    freearray(diagDegMatrix,vector_num);
    freearray(LnormMatrix,vector_num);
    free(eigenVectors);
    return Py_BuildValue("i",newK); /*K in python*/

}



static PyObject* fullSpectralPy(PyObject *self, PyObject *args)
{
    PyObject * origin_vector_list;
    Py_ssize_t m;
    Py_ssize_t n;
    Py_ssize_t i,j;
    if(!PyArg_ParseTuple(args, "iiO!", &k, &vector_num, &PyList_Type, &origin_vector_list)) {
        return NULL;
    }

    clusters_num = k;
    n = PyList_Size(PyList_GetItem(origin_vector_list, 0));
    m = PyList_Size(origin_vector_list);
    vector_len = (int) n;
    vector_list = (double**) calloc(m, n*sizeof(double));
    assertmsg(vector_list != NULL);
    for ( i=0; i<m; i++) {
        vector_list[i] = (double*) calloc(n,sizeof(double));
        assertmsg(vector_list[i] != NULL);
    }

    for (i = 0; i < m; i++){
        for (j=0; j < n; j++) {
            vector_list[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(origin_vector_list, i), j)); /*CONVERSION*/
        }
    } 

    eigengapHeuristic();
    createTMat();
    freearray(vector_list,vector_num);
    vector_list = UMat;
    
    PyObject* output_matrix = PyList_New(0);
    for (i = 0; i<vector_num; i++) {
        PyObject * centroid = PyList_New(0);

        for(j=0; j<k;j++) {
            PyList_Append(centroid, Py_BuildValue("d",vector_list[i][j]));
        }

        PyList_Append(output_matrix, centroid);
    }
    freearray(vector_list,vector_num);
    freearray(weightedAdjMatrix,vector_num);
    freearray(diagDegMatrix,vector_num);
    freearray(LnormMatrix,vector_num);
    freearray(VMat,vector_num);
    free(eigenValues);
    free(eigenVectors);
    
    return output_matrix;

}


static PyObject* kmeans(PyObject *self, PyObject *args)
{
    PyObject * origin_vector_list;
    PyObject * origin_final_centroids;
    Py_ssize_t m;
    Py_ssize_t n;
    Py_ssize_t i,j;
    int counter;
    int isequal;
    if(!PyArg_ParseTuple(args, "iiiO!O!",&clusters_num, &vector_num, &vector_len ,&PyList_Type, &origin_vector_list, &PyList_Type, &origin_final_centroids)) {
        return NULL;
    }
    k = clusters_num;
    n = PyList_Size(PyList_GetItem(origin_vector_list, 0));
    m = PyList_Size(origin_vector_list);
    vector_len = (int) n;
    vector_list = (double**) calloc(m, n*sizeof(double));
    assertmsg(vector_list != NULL);
    for ( i=0; i<m; i++) {
        vector_list[i] = (double*) calloc(n,sizeof(double));
        assertmsg(vector_list[i] != NULL);
    }
    for (i = 0; i < m; i++){
        for (j=0; j < n; j++) {
            vector_list[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(origin_vector_list, i), j)); /*CONVERSION*/
        }
    }
    n = PyList_Size(PyList_GetItem(origin_final_centroids, 0));
    m = PyList_Size(origin_final_centroids);
    centroids = (double**) calloc(k, vector_len*sizeof(double));
    assertmsg(centroids != NULL);
    for ( i=0; i<k; i++) {
        centroids[i] = (double*) calloc(vector_len,sizeof(double));
        assertmsg(centroids[i] != NULL);
    }
    for ( i=0; i<k; i++) {
        for (j=0; j<vector_len; j++) {
            centroids[i][j] = PyFloat_AsDouble(PyList_GetItem(PyList_GetItem(origin_final_centroids, i), j));
        }
    }
    clusters = (double**) calloc(k, sizeof(double*));
    /*kmeans */
    max_iter = 300;
    counter = 0;
    isequal = 1;
    while(counter<max_iter && isequal==1) {
        vector_to_cluster(k);
        isequal = update_centroids();
        counter++;
    }
    PyObject* output_matrix = PyList_New(0);
    for (i = 0; i<k; i++) {
        PyObject * centroid = PyList_New(0);

        for(j=0; j<vector_len;j++) {
            PyList_Append(centroid, Py_BuildValue("d",centroids[i][j]));
        }
        PyList_Append(output_matrix, centroid);
    }
    freearray(vector_list,(int)m);
    freearray(centroids,k);
    freearray(clusters,k);
    free(clustersindexes);
    return output_matrix;
}



static PyMethodDef spkmeansMethods[] = {
        {"fit",
     (PyCFunction) fit,
     METH_VARARGS,
     PyDoc_STR("The final centroids produced by the Kmeans algorithm")},
    {"WeightedAdjacencyMatrix",                  
      (PyCFunction) WeightedAdjacencyMatrix,
      METH_VARARGS,           
      PyDoc_STR("A Weighted Adjacency Matrix (1.1.1)")},
      {"DiagonalDegreeMatrix",                   
      (PyCFunction) DiagonalDegreeMatrix, 
      METH_VARARGS,           
      PyDoc_STR("A Diagonal Degree Matrix (1.1.2)")},
      {"NormalizedGraphLaplacian",                  
      (PyCFunction) NormalizedGraphLaplacian, 
      METH_VARARGS,           
      PyDoc_STR("A Normalized Graph Laplacian Matrix (1.1.3)")},
      {"Jacobi",                   
      (PyCFunction) Jacobi, 
      METH_VARARGS,           
      PyDoc_STR("Calculation of eigenvalues and eigenvectors with the Jacobi algorithm (1.2.1)")},
      {"heuristic",                   
      (PyCFunction) heuristic,
      METH_VARARGS,           
      PyDoc_STR("Calculation of K value with the eigengap heuristic method (1.3)")},
      {"fullSpectralPy",                   
      (PyCFunction) fullSpectralPy, 
      METH_VARARGS,           
      PyDoc_STR("Calculation of the first part of the whole process/algorithm")},
      {"kmeans",                   
      (PyCFunction) kmeans, 
      METH_VARARGS,           
      PyDoc_STR("Calculation of the second part of the whole process/algorithm")},
    {NULL, NULL, 0, NULL}   
};





static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "myspkmeans", 
    NULL, 
    -1,  
    spkmeansMethods
};


PyMODINIT_FUNC PyInit_myspkmeans(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }
    return m;
}


