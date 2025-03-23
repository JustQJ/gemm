## 单个元素的处理

### 处理float32
```
__global__ void kernel_elementwise_add_fp32(float *c, float *a, float *b, int n){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx<n){
        c[idx] = a[idx]+b[idx];
    }

}
```

### 使用float4向量化加载

```
__global__ void kernel_elementwise_add_fp32_4(float *c, float *a, float *b, int n){
    int idx = 4*(threadIdx.x + blockDim.x*blockIdx.x);
    if(idx+3<n){
        float4 reg_a = FLOAT4(a[idx]);
        float4 reg_b = FLOAT4(b[idx]);
        float4 reg_c;

        reg_c.x = reg_a.x + reg_b.x;
        reg_c.y = reg_a.y + reg_b.y;
        reg_c.z = reg_a.z + reg_b.z;
        reg_c.w = reg_a.w + reg_b.w;    

        FLOAT4(c[idx]) = reg_c;
    }else{
        
        for(int ii=idx; ii<n; ii++){
            c[ii] = a[ii]+b[ii];
        }   

    }

}
```


### 处理float16

```
__global__ void kernel_elementwise_add_half(half *c, half *a, half *b, int n){

    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    if(idx<n){
        c[idx] = __hadd(a[idx], b[idx]);
    }

}
```


### 使用half2向量指令
```
__global__ void kernel_elementwise_add_half2(half *c, half *a, half *b, int n){

    int idx = (threadIdx.x + blockDim.x*blockIdx.x)*2;
    if(idx<n){
        half2 reg_a = HALF2(a[idx]);
        half2 reg_b = HALF2(b[idx]);
        HALF2(c[idx]) = __hadd2(reg_a, reg_b);
    }
}
```


### 使用float4向量化加载half

```
__global__ void kernel_elementwise_add_half2_4(half *c, half *a, half *b, int n){
    int idx = (threadIdx.x +blockDim.x*blockIdx.x)*8;
    if(idx+7<n){
        half2 reg_a[4];
        half2 reg_b[4];
        half2 reg_c[4];
        FLOAT4(reg_a) = FLOAT4(a[idx]);
        FLOAT4(reg_b) = FLOAT4(b[idx]);

        reg_c[0] = __hadd2(reg_a[0], reg_b[0]);
        reg_c[1] = __hadd2(reg_a[1], reg_b[1]);
        reg_c[2] = __hadd2(reg_a[2], reg_b[2]);
        reg_c[3] = __hadd2(reg_a[3], reg_b[3]);

        FLOAT4(c[idx]) = FLOAT4(reg_c);
    }else{
        half2 reg_a;
        half2 reg_b;
        for(int ii=idx; ii<n; ii+=2){
            reg_a = HALF2(a[ii]);
            reg_b = HALF2(b[ii]);
            HALF2(c[ii]) = __hadd2(reg_a, reg_b);
        }
    }
}
```