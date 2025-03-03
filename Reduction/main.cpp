#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernel.h"


float cpu_reduce_sum_compute(float *A, int N){
    float val = 0;
    for(int i=0; i<N; i++){
       val+=A[i];
    }

    return val;
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

float benchmark(int n){
    float *h_A, *h_r;
    float *d_A, *d_r;
    int block_number = 108*4;

    h_A = (float *)malloc(sizeof(float)*n);
    h_r = (float *)malloc(sizeof(float)*block_number);


    for(int i=0; i<n; i++)
        h_A[i] = 0.01;
    


    cudaMalloc(&d_A, sizeof(float)*n);
    cudaMalloc(&d_r,sizeof(float)*block_number);
 

    cudaMemcpy(d_A, h_A, sizeof(float)*n, cudaMemcpyHostToDevice);

    

    custom_reduce_sum(d_A, n, d_r, block_number);
    cudaMemcpy(h_r, d_r, sizeof(float)*block_number, cudaMemcpyDeviceToHost);

    float res = cpu_reduce_sum_compute(h_A, n);
    float final_res = 0;
    for(int i=0; i<block_number; i++)
        final_res+=h_r[i];
    
    printf("error costom reduce res: %f, target res: %f\n",final_res ,res);
    
    /*
    measure performance
    */

    int start, end;
    int iter = 10;
    
  
    start = get_us();
    for(int i=0; i<iter; i++){
        custom_reduce_sum(d_A, n, d_r, block_number);
    }
    cudaDeviceSynchronize();
    end = get_us();
    float cost = (float)(end-start)/iter;

    float bytes = sizeof(float)*n;
    float bandwidth_the  = 1555; //a100
    float current_bandwidth = bytes*1000000/cost/1024/1204/1024;
    printf("N = %d, Costom Reduce cost: %f us, bandwith : %f GB/s, ratio: %f\n", n, cost, current_bandwidth, current_bandwidth/bandwidth_the);
    


    cudaFree(d_A);
    cudaFree(d_r);


    free(h_A);
    free(h_r);

    return current_bandwidth;
    

}

int main() {
    int num = 10;
    float bandwidths[num];
    for(int m=1; m<=num; m++){
        float bd = benchmark(m*1024*1024);
        bandwidths[m-1] = bd;
    }

    FILE *file= fopen("test.txt","w");
    if (file == NULL) {
        printf("无法打开文件！\n");
        return 1;
    }
    for(int i=0; i<num; i++){
        fprintf(file, "%.2f\n", bandwidths[i]);
    }
        
    return 0;
}