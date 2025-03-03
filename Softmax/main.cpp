#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>
#include<limits.h>
#include<math.h>
#include<algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "kernel.h"


int get_us(){

    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec*1000000+ t.tv_usec;
}


void cpu_softmax(float *A, int m, int n){
    for(int i=0; i<m; i++){
        float max_val = -1000000; //our setting in A is >0
        int start_ = i*n;
        for(int j=0; j<n; j++){
            max_val = std::max(max_val, A[start_+j]);
        }
        float norm_sum = 0;
        for(int j=0; j<n; j++){
            norm_sum+=expf(A[start_+j]-max_val);
        }

        for(int j=0; j<n; j++){
            A[start_+j] = expf(A[start_+j]-max_val)/norm_sum;
        }

    }
}

float check_correct(float *A, float *A1, int m, int n){
    float max_error = 0;
    for(int i=0; i<m*n; i++){
        max_error = std::max(max_error, abs(A[i]-A1[i]));
    }

    return max_error;
}


float benchmark(int m, int n){
    float *h_A, *h_R;
    float *d_A;


    h_A = (float *)malloc(sizeof(float)*n*m);
    h_R = (float *)malloc(sizeof(float)*n*m);
    

    for(int i=0; i<n*m; i++)
        h_A[i] = ((float)rand())/RAND_MAX;
    
    
    cudaMalloc(&d_A, sizeof(float)*n*m);
 

    cudaMemcpy(d_A, h_A, sizeof(float)*n*m, cudaMemcpyHostToDevice);

    

    fused_softmax(d_A, m, n);
    cudaMemcpy(h_R, d_A, sizeof(float)*m*n, cudaMemcpyDeviceToHost);

    cpu_softmax(h_A, m,n);
    float error = check_correct(h_A, h_R, m,n);
    
    printf("hA[0]: %f, TA[0]: %f, error fused softmax: %f\n", h_R[0], h_A[0], error);
    
    /*
    measure performance
    */

    int start, end;
    int iter = 10;
    fused_softmax(d_A, m, n);
    cudaDeviceSynchronize();

    start = get_us();
    for(int i=0; i<iter; i++){
       fused_softmax(d_A, m, n);
    }
    cudaDeviceSynchronize();
    end = get_us();
    float cost = (float)(end-start)/iter;

    float bytes = sizeof(float)*n*m*2;
    float bandwidth_the  = 1555; //a100
    float current_bandwidth = bytes*1000000/cost/1024/1204/1024;
    printf("M:N = %d : %d, Costom Reduce cost: %f us, bandwith : %f GB/s, ratio: %f\n", m, n, cost, current_bandwidth, current_bandwidth/bandwidth_the);
    


    cudaFree(d_A);
    


    free(h_A);
    free(h_R);

    return current_bandwidth;
    

}

int main() {
    
    //m 一般小于 n，m为batch size的话一般也就128，256，如果是attention矩阵，则是L，一般到1024
    //n 是feature，所以大，以512 开始

    int m = 1024;
    int n = 512;
    int num = 20;
    float bandwidths[num];
    for(int k=1; k<=num; k++){
        float bd = benchmark(m, n*k);
        bandwidths[k-1] = bd;
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