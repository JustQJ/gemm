#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << #call << ": " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "CUBLAS error in " << #call << std::endl; \
            exit(1); \
        } \
    } while (0)

int main() {
    const int M = 1024, K = 1024, N = 1024;
    float h_A[M*K] = {1.0f};
    float h_B[K*N] = {1.0f};
    float h_C[M*N] = {0};  // 初始化为零

    float *d_A, *d_B, *d_C;

    // CUDA内存分配并检查错误
    CUDA_CHECK(cudaMalloc((void**)&d_A, M*K * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, K*N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_C, M*N * sizeof(float)));

    // 将数据从主机复制到设备
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K*N * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f;
    float beta = 1.0f;

    // 调用 cublasSgemm 计算 C = alpha * A * B + beta * C
    CUBLAS_CHECK(cublasSgemm(
        handle, 
        CUBLAS_OP_N, CUBLAS_OP_N,  // 不转置 A 和 B
        M, N, K,                  // C 的尺寸 MxN，A 的尺寸 MxK，B 的尺寸 KxN
        &alpha, 
        d_A, M,                   // A 矩阵及其 leading dimension
        d_B, K,                   // B 矩阵及其 leading dimension
        &beta, 
        d_C, M                    // C 矩阵及其 leading dimension
    ));

    // 将结果从设备复制回主机
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M*N * sizeof(float), cudaMemcpyDeviceToHost));

    // 打印部分结果以验证
    std::cout << "Result matrix C (first 10 elements): \n";
    for (int i = 0; i < 10; ++i) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // 清理
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
