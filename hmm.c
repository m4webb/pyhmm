#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include "hmm.h"

int _hmm_climb(hmm_model_t *model, double *logprob)
{
    int n = model->n;
    int m = model->m;
    int t = model->t;
    double *a = model->a;
    double *b = model->b;
    long *o = model->o;
    double *alpha = model->alpha;
    double *beta = model->beta;
    double *gamma = model->gamma;
    double *digamma = model->digamma;
    double *c = model->c;
    double numer;
    double denom;
    int i, j, k;

    // forward
    c[0] = 0.;
    for (i=0; i<n; i++)
    {
        alpha[i] = (1./n)*b[i*m + o[0]]; // uniform pi
        c[0] += alpha[i];
    }
    c[0] = 1./c[0];
    for (i=0; i<n; i++)
    {
        alpha[i] *= c[0];
    }
    for (k=1; k<t; k++)
    {
        c[k] = 0.;
        for (i=0; i<n; i++)
        {
            alpha[k*n + i] = 0.;
            for (j=0; j<n; j++)
            {
                alpha[k*n + i] += alpha[(k-1)*n + j]*a[j*n + i];
            }
            alpha[k*n + i] *= b[i*m + o[k]];
            c[k] += alpha[k*n + i];
        }
        c[k] = 1./c[k];
        for (i=0; i<n; i++)
        {
            alpha[k*n + i] *= c[k];
        }
    }

    // backward
    for (i=0; i<n; i++)
    {
        beta[(t-1)*n + i] = c[t-1];
    }
    for (k=(t-2); k>=0; k--)
    {
        for (i=0; i<n; i++)
        {
            beta[k*n + i] = 0.;
            for (j=0; j<n; j++)
            {
                beta[k*n + i] += a[i*n + j]*b[j*m + o[k+1]]
                        *beta[(k+1)*n + j];
            }
            beta[k*n + i] *= c[k];
        }
    }
    
    // gammas
    for (k=0; k<(t-1); k++)
    {
        denom = 0.;
        for (i=0; i<n; i++)
        {
            for (j=0; j<n; j++)
            {
                denom += alpha[k*n + i]*a[i*n + j]*b[j*m + o[k+1]]
                        *beta[(k+1)*n + j];
            }
        }
        for (i=0; i<n; i++)
        {
            gamma[k*n + i] = 0.;
            for (j=0; j<n; j++)
            {
                digamma[k*n*n + i*n + j] = alpha[k*n + i]*a[i*n + j] *b[j*m +
                        o[k+1]]*beta[(k+1)*n + j]/denom;
                gamma[k*n + i] += digamma[k*n*n + i*n + j];
            }
        }
    }
    denom = 0.;
    for (i=0; i<n; i++)
    {
        denom += alpha[(t-1)*n + i];
    }
    for (i=0; i<n; i++)
    {
        gamma[(t-1)*n + i] = alpha[(t-1)*n + i]/denom;
    }

    // climb
    for (i=0; i<n; i++)
    {
        for (j=0; j<n; j++)
        {
            numer = 0.;
            denom = 0.;
            for (k=0; k<(t-1); k++)
            {
                numer += digamma[k*n*n + i*n + j];
                denom += gamma[k*n + i];
            }
            MPI_Allreduce(MPI_IN_PLACE, &numer, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &denom, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
            a[i*n + j] = numer/denom;
        }
    }
    for (i=0; i<n; i++)
    {
        for (j=0; j<m; j++)
        {
            numer = 0.;
            denom = 0.;
            for (k=0; k<t; k++)
            {
                if (o[k] == j)
                {
                    numer += gamma[k*n + i];
                }
                denom += gamma[k*n + i];
            }
            MPI_Allreduce(MPI_IN_PLACE, &numer, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &denom, 1, MPI_DOUBLE, MPI_SUM,
                    MPI_COMM_WORLD);
            b[i*m + j] = numer/denom;
        }
    }

    *logprob = 0.;
    for (k=0; k<t; k++)
    {
        *logprob -= log(c[k]);
    }
    MPI_Allreduce(MPI_IN_PLACE, logprob, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    return 0;
}

int _hmm_logprob(hmm_model_t *model, double *logprob)
{
    int n = model->n;
    int m = model->m;
    int t = model->t;
    double *a = model->a;
    double *b = model->b;
    long *o = model->o;
    double *alpha = malloc(t*n*sizeof(double));
    double *c = malloc(t*sizeof(double));
    int i, j, k;

    // forward
    c[0] = 0.;
    for (i=0; i<n; i++)
    {
        alpha[i] = (1./n)*b[i*m + o[0]]; // uniform pi
        c[0] += alpha[i];
    }
    c[0] = 1./c[0];
    for (i=0; i<n; i++)
    {
        alpha[i] *= c[0];
    }
    for (k=1; k<t; k++)
    {
        c[k] = 0.;
        for (i=0; i<n; i++)
        {
            alpha[k*n + i] = 0.;
            for (j=0; j<n; j++)
            {
                alpha[k*n + i] += alpha[(k-1)*n + j]*a[j*n + i];
            }
            alpha[k*n + i] *= b[i*m + o[k]];
            c[k] += alpha[k*n + i];
        }
        c[k] = 1./c[k];
        for (i=0; i<n; i++)
        {
            alpha[k*n + i] *= c[k];
        }
    }

    // logprob
    *logprob = 0.;
    for (k=0; k<t; k++)
    {
        *logprob -= log(c[k]);
    }
    MPI_Allreduce(MPI_IN_PLACE, logprob, 1, MPI_DOUBLE, MPI_SUM,
            MPI_COMM_WORLD);
    return 0;
}

int _hmm_write(hmm_model_t *model)
{
    fwrite(model->a, sizeof(double), model->n*model->n, model->a_file);
    fwrite(model->b, sizeof(double), model->n*model->m, model->b_file);
    fflush(model->a_file);
    fflush(model->b_file);
    return 0;
}
