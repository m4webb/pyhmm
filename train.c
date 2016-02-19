#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "hmm.h"

#define PRIOR 5.

int main(int argc, char **argv)
{
    hmm_model_t *model = malloc(sizeof(hmm_model_t));
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &model->mpi_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &model->mpi_rank);

    model->m = atoi(argv[1]);
    model->n = atoi(argv[2]);
    long lag = atol(argv[3]);
    double logprob = -INFINITY;
    double oldlogprob;
    long iter = 0;
    
    if (model->mpi_rank == 0)
    {
        model->o_file = fopen(argv[4], "rb");
        model->a_file = fopen(argv[5], "wb");
        model->b_file = fopen(argv[6], "wb");
        long T;
        int i;
        fseek(model->o_file, 0, SEEK_END);
        T = ftell(model->o_file) / sizeof(long);
        model->t = T / model->mpi_size;
        MPI_Bcast(&model->t, 1, MPI_LONG, 0, MPI_COMM_WORLD);
        fseek(model->o_file, 0, SEEK_SET);

        model->o = malloc(sizeof(long)*model->t);
        fread(model->o, sizeof(long), model->t, model->o_file);

        long *send = malloc(sizeof(long) * model->t);
        for (i=1; i<model->mpi_size; i++)
        {
            fread(send, sizeof(long), model->t, model->o_file);
            MPI_Send(send, model->t, MPI_LONG, i, 0, MPI_COMM_WORLD);
        }
        free(send);
        fclose(model->o_file);

        gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);
        gsl_rng_set(rng, time(NULL));
        model->a = malloc(sizeof(double) * model->n*model->n);
        model->b = malloc(sizeof(double) * model->n*model->m);
        double *alpha_n = malloc(sizeof(double) * model->n);
        double *alpha_m = malloc(sizeof(double) * model->m);
        for (i=0; i<model->n; i++)
        {
            alpha_n[i] = PRIOR;
        }
        for (i=0; i<model->m; i++)
        {
            alpha_m[i] = PRIOR;
        }
        for (i=0; i<model->n; i++)
        {
            gsl_ran_dirichlet(rng, model->n, alpha_n, &model->a[i*model->n]);
            gsl_ran_dirichlet(rng, model->m, alpha_m, &model->b[i*model->m]);
        }
        free(alpha_n);
        free(alpha_m);
        gsl_rng_free(rng);
        MPI_Bcast(model->a, model->n*model->n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(model->b, model->n*model->m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Bcast(&model->t, 1, MPI_LONG, 0, MPI_COMM_WORLD);
        model->o = malloc(sizeof(long)*model->t);
        MPI_Recv(model->o, model->t, MPI_LONG, 0, 0, MPI_COMM_WORLD,
            &model->status);
        model->a = malloc(sizeof(double) * model->n*model->n);
        model->b = malloc(sizeof(double) * model->n*model->m);
        MPI_Bcast(model->a, model->n*model->n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(model->b, model->n*model->m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    model->alpha = malloc(model->t*model->n*sizeof(double));
    model->beta = malloc(model->t*model->n*sizeof(double));
    model->gamma = malloc(model->t*model->n*sizeof(double));
    model->digamma = malloc(model->t*model->n*model->n*sizeof(double));
    model->c = malloc(model->t*sizeof(double));

    do 
    {
        oldlogprob = logprob;
        _hmm_climb(model, &logprob);
        if (iter++ % lag == 0 && model->mpi_rank == 0)
        {
            _hmm_write(model);
        }
        if (model->mpi_rank == 0)
        {
            printf("%ld %f\n", iter, logprob);
        }
    }
    while (logprob - oldlogprob > 1e-8);

    if (model->mpi_rank == 0)
    {
        fclose(model->a_file);
        fclose(model->b_file);
    }
    free(model->a);
    free(model->b);
    free(model->o);
    free(model->alpha);
    free(model->beta);
    free(model->gamma);
    free(model->digamma);
    free(model->c);
    free(model);
    MPI_Finalize();
    return 0;
}
