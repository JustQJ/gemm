

void cublas_mm(float *d_A, float *d_B, float *d_C, int M, int N, int K);

void naive_mm(float *A, float *B, float *C, int N, int M, int K);

void shared_mm0(float *A, float *B, float *C, int N, int M, int K);

void shared_mm1(float *A, float *B, float *C, int N, int M, int K);

void shared_mm2(float *A, float *B, float *C, int N, int M, int K);

