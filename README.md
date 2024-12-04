# GEMM Kerenls

some useful kerenls created with cuda and trion

## Matrix Multiplication
optimize the gemm to achieve cublas performance

### Matirx-Matrix
计算矩阵乘法
$$
C^{m\times n} = A^{m\times k} * B^{k\times n}
$$
#### CPU单线程计算
```c++
for(int i=0; i<n; i++){
    for(int j=0; j<n; j++){
        for(int k=0; k<n; k++){
            C[i][j] += A[i][k]*B[k][j];
        }
    }
}
```

#### GPU 基础实现
















### Vector-Matirx

## Reduction

## Attention

### Flash Attention

### Ring Attention


## Softmax



