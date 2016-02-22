#include <stdio.h>

typedef struct hmm_model_t {
    int n;
    int m;
    int t;
    double *a;
    double *b;
    long *o;
    double *alpha;
    double *beta;
    double *gamma;
    double *c;
    double *numer_nn;
    double *denom_nn;
    double *numer_nm;
    double *denom_nm;
    int mpi_rank;
    int mpi_size;
    MPI_Status status;
    FILE *a_file;
    FILE *b_file;
    FILE *o_file;
} hmm_model_t;

int _hmm_climb(hmm_model_t *model, double *logprob);
int _hmm_logprob(hmm_model_t *model, double *logprob);
int _hmm_write(hmm_model_t *model);
