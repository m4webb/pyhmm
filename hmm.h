typedef struct hmm_model_t {
    int n;
    int m;
    int t;
    double *a;
    double *b;
    long *o;
} hmm_model_t;

int _hmm_train(hmm_model_t *model);
int _hmm_logprob(hmm_model_t *model, double *logprob);
