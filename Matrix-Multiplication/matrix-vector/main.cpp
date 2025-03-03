#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernel.h"


void cpu_compute(float *A, float *x, float *y, int m, int n, float alpha, float beta){
    for(int i=0; i<m; i++){
        float val = 0;
        for(int j=0; j<n; j++){
            val+=A[i*n+j]*x[j];
        }
        y[i] = alpha*val+beta*y[i];
    }
}

float check_correct(float *y, float *y1, int  m){
    float max_error = 0;
    for(int i=0; i<m; i++){
        max_error = fmax(abs(y[i]-y1[i]), max_error);
    }
    return max_error;
}

int get_us(){

    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec*1000000+ t.tv_usec;
}

void benchmark(int m, int n){
    float *h_A, *h_x, *h_y, *r_y;
    float *d_A, *d_x, *d_y, *cublas_y;

    h_A = (float *)malloc(sizeof(float)*m*n);
    h_x = (float *)malloc(sizeof(float)*n);
    h_y = (float *)malloc(sizeof(float)*m);

    r_y = (float *)malloc(sizeof(float)*m);

    for(int i=0; i<m*n; i++)
        h_A[i] = i%10+0.01;
    
    for(int i=0; i<n; i++)
        h_x[i] = i%10+0.1;
    
    for(int i=0; i<m; i++)
        h_y[i] = 0.1;

    cudaMalloc(&d_A, sizeof(float)*m*n);
    cudaMalloc(&d_x, sizeof(float)*n);
    cudaMalloc(&d_y, sizeof(float)*m);
    cudaMalloc(&cublas_y, sizeof(float)*m);

    cudaMemcpy(d_A, h_A, sizeof(float)*m*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sizeof(float)*m, cudaMemcpyHostToDevice);
    cudaMemcpy(cublas_y, h_y, sizeof(float)*m, cudaMemcpyHostToDevice);

    float alpha = 0.2;
    float beta = 0.4;


    /*
    cublas sgemv
    */

    cublasHandle_t handle;
    cublasCreate(&handle);
    /*d_A
    cublasSgemv 处理的是列优先，所以传入的矩阵设置成n*m的矩阵，并且计算其转置才是m*n的行优先的矩阵
    */
    cublasStatus_t status = cublasSgemv(handle, CUBLAS_OP_T, n, m, &alpha, d_A, n, d_x, 1, &beta, cublas_y, 1);
    cudaMemcpy(r_y, cublas_y, sizeof(float)*m, cudaMemcpyDeviceToHost);

    if(status != CUBLAS_STATUS_SUCCESS){
            printf("cublas error\n");
        }
     

    custom_sgemv(d_A, d_x, d_y, m,  n,  alpha,  beta);
    cudaMemcpy(h_y, d_y, sizeof(float)*m, cudaMemcpyDeviceToHost);
    float error = check_correct(h_y, r_y, m);
    printf("error between cublas and costom sgemv: %f\n", error);
    
    /*
    measure performance
    */

    int start, end;
    int iter = 10;
    start = get_us();
    for(int i=0; i<iter; i++){
        status = cublasSgemv(handle, CUBLAS_OP_T, n, m, &alpha, d_A, n, d_x, 1, &beta, cublas_y, 1);
    }
    if(status != CUBLAS_STATUS_SUCCESS){
            printf("cublas error\n");
        }
    cudaDeviceSynchronize();
    end = get_us();
    float cublas_cost = (float)(end-start)/iter;
    
    start = get_us();
    for(int i=0; i<iter; i++){
        custom_sgemv(d_A, d_x, d_y, m, n, alpha, beta);
    }
    cudaDeviceSynchronize();
    end = get_us();
    float cost = (float)(end-start)/iter;

    float flops = 2*m*n;

    printf("M:N = %d:%d, Costom Sgemv / Cublas Sgemv, time cost(us): %f / %f, GFLOPs: %f / %f, speedup: %f \n", m,n,cost, cublas_cost,  flops/cost/1000, flops/cublas_cost/1000, cublas_cost/cost);
    

     



    



    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(cublas_y);

    free(h_A);
    free(h_x);
    free(h_y);
    free(r_y);






    

}

int main() {
   
    for(int m=512; m<=5120; m+=512)
        benchmark(m, m);
    return 0;
}