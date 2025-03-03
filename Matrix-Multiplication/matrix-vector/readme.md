## 矩阵向量乘

$$

y^m = \alpha A^{m\times n}x^{n} + \beta y^m

$$

该计算是memory bound的，计算其算术密度
$$

ar =  \frac{2mn}{4(mn+m+n)}<1
$$

## 算子思路
### naive
因为只有m个结果，所以一般多个thread同时计算一个结果。在大模型的推理过程中，一般m和n大小相同，在1000~10000，所以一般使用一个swap处理一个结果，即读取A的一行，然后和x计算得到结果。由于A中的元素都只使用一次，所以没有必要使用shared memory。

```cpp
#define TILE_M 16 //small when m is samll, large when m is large, such has enough blocks
#define TILE_N 32

#define FULL_MASK 0xffffffff


// each block has 32 warp, each warp process one element, each each load one element each time with float 
__global__ void __kernel_sgemv(float *A, float *x, float *y, int m, int n, float alpha, float beta){

    int wrap_id = threadIdx.y;
    int line_id = threadIdx.x%32;
    int row = blockIdx.x*TILE_M+wrap_id;

    if(row<m){
        float val = 0;
        for(int i=0; i<n; i+=TILE_N){
            int nn = i+line_id;
            if(nn<n){
                float a = A[row*n + nn];
                float b = x[nn];
                val+=a*b;
            }
           
        }
        //synchronize to the first thread in a wrap, line_id=0
        //use shfl_down_sync instruction to reduce all val to the first
        for(int offset=16; offset>0; offset>>=1){
                val+=__shfl_down_sync(FULL_MASK, val, offset);
        }
        
        if(line_id==0){
            y[row] = alpha * val + beta * y[row];
        }  
    }

}

void custom_sgemv(float *A, float *x, float *y, int m, int n, float alpha, float beta){

    dim3 block(32, TILE_M); //each swap in same row, process one output element
    dim3 grid((m+TILE_M-1)/TILE_M);
    __kernel_sgemv<<<grid, block>>>(A, x, y, m,n, alpha, beta);

}
```

上面的代码中，一个block的x维度为32，刚好一个warp，所以一个线程的y坐标就表示其需要处理的行。block的y维度的大小根据m进行选择，使得block的数量合适，不能太少。 每次一个线程加载一个数，做乘加。

一个warp中的线程将自己的行处理完后，需要规约到当前warp的第一个线程，这里需要使用到线程级别的规约。`__shfl_down_sync()`指令将`id+offset`线程中的值返回到`id`线程中。
```cpp
for(int offset=16; offset>0; offset>>=1){
    val+=__shfl_down_sync(FULL_MASK, val, offset);
}
```
`FULL_MASK` 表示那些线程参与到当前的指令中，一般是整个warp，就是`0xffffffff`。          


### 向量化加载

在GPU的访存中，使用向量加载指令可以减少指令的数量，即使用LD128比使用LD32更能有效利用带宽。所以可以使用`float4`进行加载数据，一个线程一次加载4个float，而不是一个float。

```cpp
#define TILE_M 16 //wrap number each block, change with the m to ensure same enough number of blocks
#define TILE_N 32 

//use vector load float4 to increase memory usage
//each warp deals with deal result
__global__ void __kernel_sgemv(float *A, float *x, float *y, int m, int n, float alpha, float beta){
    int line_id = threadIdx.x % 32;
    int warp_id = threadIdx.y;
    int row = warp_id + blockIdx.x*blockDim.y;

    if(row<m){
        float val = 0;
        //int iter = n/32/4; //each time load 32*4 elemets for using float4
        for(int i=0; i<n; i+=32*4){
            int nn = i+line_id*4;
            if(nn+4<n){
                float4 a = reinterpret_cast<float4 *>(&A[row*n+nn])[0];
                float4 b = reinterpret_cast<float4 *>(&x[nn])[0];
                val+=a.x*b.x;
                val+=a.y*b.y;
                val+=a.z*b.z;
                val+=a.w*b.w;
            }else if(nn<n){ //not enough 4, so load with float
                for(int nnn=nn; nnn<n; nnn++){
                    float a = A[row*n+nnn];
                    float b = x[nnn];
                    val+=a*b;
                }
            }
            
        }

        //shf down to line_id=0;
        for(int offset=16; offset>0; offset>>=1){
            val+= __shfl_down_sync(0xffffffff, val, offset);
        }
        if(line_id==0){
            y[row] = alpha * val + beta* y[row];
        }
    }

}



void custom_sgemv(float *A, float *x, float *y, int m, int n, float alpha, float beta){
    dim3 block(32, TILE_M);
    dim3 grid((m+TILE_M-1)/TILE_M);
    __kernel_sgemv<<<grid, block>>>(A, x, y, m, n, alpha, beta);
}


```

使用下面的命令，查看反汇编的sass指令，可以看到`LDG.E.128` 的指令。而不使用float4，则只有`LDG.E` 表示32bit加载指令。
```
cuobjdump ./build/main -sass


.....
/*0888*/                   LDG.E.128 R4, [R4] ;                            /* 0xeed6200000070404 */
/*0890*/                   LDG.E.128 R8, [R8] ;                            /* 0xeed6200000070808 */
```

### Tensor Core

使用`half`指令，并使用Tensor Core进行计算。

**!TODO**

### 总结

Sgemv的优化较少，因为用不到shared memory。简单版本测试，和cublas的性能差不多。