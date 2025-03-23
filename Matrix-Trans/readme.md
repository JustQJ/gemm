## 转置矩阵
y[j][i] = x[i][j]

## 每个thread处理一个元素

在写入y时，同一个swap的线程写出的地址不连续。
```
__global__ void kernel_mat_trans_fp32(float *y, float *x, const int m, const int n){
    int col = blockDim.x*blockIdx.x+threadIdx.x;
    int row = blockDim.y*blockIdx.y+threadIdx.y;

    if(col<n && row<m){
        y[col*m+row] = x[row*n+col]; 
    }

}
```

## float4 向量化加载

在写入y时，同一个swap的线程写出的地址不连续。

```
__global__ void kernel_mat_trans_fp32_4(float *y, float *x, const int m, const int n){
    int col = (blockDim.x*blockIdx.x + threadIdx.x)*4;
    int row = blockDim.y*blockIdx.y + threadIdx.y;

    if(col+3<n && row<m){
        float4 reg_x = FLOAT4(x[row*n+col]); //vector load, but can't store with vector
        y[(col)*m + row] = reg_x.x;
        y[(col+1)*m + row] = reg_x.y;
        y[(col+2)*m + row] = reg_x.z;
        y[(col+3)*m + row] = reg_x.w;
    }else if(col<n && row<m){
        #pragma unroll
        for(int ii=col; ii<n; ii++){
            y[ii*m+row] = x[row*n+ii];
        }
    }    
}
```

## shared memory

使得同一个swap的线程写出的地址连续
```
__global__ void kernel_mat_trans_fp32_4_shared(float *y, float *x, const int m, const int n){
    int global_x = threadIdx.x+blockDim.x*blockIdx.x;
    int global_y = threadIdx.y+blockDim.y*blockIdx.y;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = threadIdx.x+blockDim.x*threadIdx.y;
    __shared__ float tile[WARP_SIZE][WARP_SIZE*4]; //each thread load 4 elements into the shared memory
    if(global_x*4+3<n && global_y<m){
        FLOAT4(tile[tidy][tidx*4]) = FLOAT4(x[global_y*n + global_x*4]);
    }

    __syncthreads();
    
    if(global_x*4+3<n && global_y<m){
        //load data from the shared memory with y direction 
        int num_tid_per_col = WARP_SIZE/4;
        float4 reg_x;
        reg_x.x = tile[(tid%num_tid_per_col)*4][tid/num_tid_per_col];
        reg_x.y = tile[(tid%num_tid_per_col)*4+1][tid/num_tid_per_col];
        reg_x.z = tile[(tid%num_tid_per_col)*4+2][tid/num_tid_per_col];
        reg_x.w = tile[(tid%num_tid_per_col)*4+3][tid/num_tid_per_col];

        FLOAT4(y[(blockDim.x*blockIdx.x*4+ tid/8)*m + blockDim.y*blockIdx.y+ (tid%8)]) = reg_x;

    }

}
```

上面的代码在从shared memory中读取数据时，同一个warp的相邻8个线程读取的是同一个bank的数据，所以存在较大的memory bank conflict。

## 避免Memory Bank Conflict

将shared memory进行padding一个位置，并改变线程的读取方式。下面的代码中，每个thread从shared memory中读取四个相邻的元素，同一个warp的32个线程读取32行，同一个列的元素。由于padding了，所以虽然是同一列，但是在内存的排布上是不同的bank。同时，在写入y时，虽然使用的不是float4写入，但是保证了每次写入的一个warp的所有线程写入的是相邻的地址。

在`0~31`($tidx\in [0,31], tidy=0$)线程第一次读取shared memory时，`reg_x.x = tile[tidx][4*tidy]`，可以计算每个实际读取的一维数组的位置，并计算所在的bank。
$$
((tidx*(32*4+1))+4*tidy)\%32 = ((tidx*(32*4+1))+4*0)\%32 =((tidx*(32*4+1)))\%32 = tidx \\
$$
所以每个线程读取的bank和其tidx相同，分布在`0~31`，属于不同的bank，不会发生冲突。

同理，对于其他次读取(`4*tidy+1, 4*tidy+2, 4*tidy+3`)和其他warp的线程，同样可以计算得到每次读取时一个warp的线程读取的不是同一个bank。


在写入y时，每次写入不同的行（地址不连续），但是每次写入时，同一个warp写入的地址连续即可。

```
__global__ void kernel_mat_trans_fp32_4_shared_smc(float *y, float *x, const int m, const int n){
    int global_x = threadIdx.x+blockDim.x*blockIdx.x;
    int global_y = threadIdx.y+blockDim.y*blockIdx.y;
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int tid = threadIdx.x+blockDim.x*threadIdx.y;
    __shared__ float tile[WARP_SIZE][WARP_SIZE*4+1]; //padding one to avoid conflict
  
    float4 reg_x;
    if(global_x*4+3<n && global_y<m){
        reg_x = FLOAT4(x[global_y*n + global_x*4]);

        FLOAT4(tile[tidy][tidx*4]) = FLOAT4(x[global_y*n + global_x*4]);


    }

    __syncthreads();
    
    if(global_x*4+3<n && global_y<m){
        //load data from the shared memory with y direction 
        int num_tid_per_col = WARP_SIZE/4;
        float4 reg_x;
        reg_x.x = tile[tidx][4*tidy];
        reg_x.y = tile[tidx][4*tidy+1];
        reg_x.z = tile[tidx][4*tidy+2];
        reg_x.w = tile[tidx][4*tidy+3];

        // FLOAT4(y[(blockDim.x*blockIdx.x*4+ tid/8)*m + blockDim.y*blockIdx.y+ (tid%8)]) = reg_x;

        y[(blockDim.x*blockIdx.x*4+ 4*tidy)*m + blockDim.y*blockIdx.y+ tidx] = reg_x.x;
        y[(blockDim.x*blockIdx.x*4+ 4*tidy+1)*m + blockDim.y*blockIdx.y+ tidx] = reg_x.y;
        y[(blockDim.x*blockIdx.x*4+ 4*tidy+2)*m + blockDim.y*blockIdx.y+ tidx] = reg_x.z;
        y[(blockDim.x*blockIdx.x*4+ 4*tidy+3)*m + blockDim.y*blockIdx.y+ tidx] = reg_x.w;

    }

}


```