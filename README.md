
## 相关资料
https://github.com/DefTruth/CUDA-Learn-Notes

https://github.com/gpu-mode/lectures

https://docs.nvidia.com/cuda/cuda-c-programming-guide/

https://docs.nvidia.com/cuda/cuda-math-api/index.html

## GPU 基础

检查GPU的各项参数，包括SM的数量，shared memory的大小等。

```c++


inline int _ConvertSMVer2Cores(int major, int minor) {
  // Defines for GPU Architecture types (using the SM version to determine
  // the # of cores per SM
  typedef struct {
    int SM;  // 0xMm (hexidecimal notation), M = SM Major version,
    // and m = SM minor version
    int Cores;
  } sSMtoCores;

  sSMtoCores nGpuArchCoresPerSM[] = {
      {0x30, 192},
      {0x32, 192},
      {0x35, 192},
      {0x37, 192},
      {0x50, 128},
      {0x52, 128},
      {0x53, 128},
      {0x60,  64},
      {0x61, 128},
      {0x62, 128},
      {0x70,  64},
      {0x72,  64},
      {0x75,  64},
      {0x80,  64},
      {0x86, 128},
      {0x87, 128},
      {0x89, 128},
      {0x90, 128},
      {0xa0, 128},
      {0xa1, 128},
      {0xc0, 128},
      {-1, -1}};

  int index = 0;

  while (nGpuArchCoresPerSM[index].SM != -1) {
    if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
      return nGpuArchCoresPerSM[index].Cores;
    }

    index++;
  }

  // If we don't find the values, we default use the previous one
  // to run properly
  printf(
      "MapSMtoCores for SM %d.%d is undefined."
      "  Default to use %d Cores/SM\n",
      major, minor, nGpuArchCoresPerSM[index - 1].Cores);
  return nGpuArchCoresPerSM[index - 1].Cores;
}


int print_gpu_info(){
     int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);  // 获取可用的GPU数量

    if (deviceCount == 0) {
        printf("There is no available device that supports CUDA.\n");
        return -1;
    }

    for (int deviceId = 0; deviceId < deviceCount; ++deviceId) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, deviceId);

        printf("Device ID: %d\n", deviceId);

        // SM数量
        printf("  SM Number (SMs): %d\n", deviceProp.multiProcessorCount);

        // 每个SM的共享内存大小
        printf("  Shared memory per block: %ld KB\n", deviceProp.sharedMemPerBlock/ (1024));

        printf("  Shared memory per SM: %ld KB\n", deviceProp.sharedMemPerMultiprocessor / (1024));

        // 每个线程块的最大线程数
        printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);

        printf("  Max threads per SM: %d\n", deviceProp.maxThreadsPerMultiProcessor);

        printf("  Warp size: %d\n", deviceProp.warpSize);


        // 最大线程数维度
        printf("  Max dimension of threads per block: (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);

        printf("  L2 Cache Size: %d KB\n", deviceProp.l2CacheSize / 1024);

        // 设备全局内存大小
        printf("  Global memory: %ld GB\n", deviceProp.totalGlobalMem / (1024 * 1024 * 1024) );

        //计算内存带宽和理论算力
        unsigned int memoryClockRate = deviceProp.memoryClockRate;  // 内存时钟频率 khz
        unsigned int memoryBusWidth = deviceProp.memoryBusWidth;    // 内存位宽 bit
        int memoryBandwidth = 2 * memoryClockRate * (memoryBusWidth / 8) / 1e6;  // 内存带宽 GB/s
        printf("  Memory Bandwidth: %d GB/s\n", memoryBandwidth);

        // 不同的架构每个SM有不同的核心数，例如volta架构每个SM有64个CUDA核心
        // 和线程数不同
        
        int cudacores_persm = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);

        printf("  CUDA Cores per SM: %d\n", cudacores_persm);
        
        // deviceProp.clockRate 是 khz，所以要除以 1.0e9，得到 THz
        printf("  Theoretical TFLOPS: %f\n", deviceProp.multiProcessorCount* cudacores_persm * 2.0 * deviceProp.clockRate / 1.0e9);


        // 最大块数维度
        printf("\n");

       
    }

     return 0;
}

```
RTX4090 的参数
```
Device ID: 0
  SM Number (SMs): 128
  Shared memory per block: 48 KB
  Shared memory per SM: 100 KB
  Max threads per block: 1024
  Max threads per SM: 1536
  Warp size: 32
  Max dimension of threads per block: (1024, 1024, 64)
  L2 Cache Size: 73728 KB
  Global memory: 23 GB
  Memory Bandwidth: 1008 GB/s
  CUDA Cores per SM: 128
  Theoretical TFLOPS: 82.575360
```
其中，带宽的计算是根据内存的时钟频率和内存总线宽度（DDR要乘以2表示双倍速率），即
$$
Bandwidth = memoryClock(hz) * memoryBusWidth(bit) * 2  = memoryClock(hz) * (memoryBusWidth(bit)/8) * 2 / 10^9 (GB/s)
$$

理论算力是cuda core的数量和时钟频率的乘积，由于支持FMA(fused multply-add)操作，即一次可以同时执行加乘操作，需要乘以2，即
$$
Compute = clock(hz) * cores * 2 / 10^{12} TFLOPS
$$


在GPU计算过程中，计算和内存访问是并行的，因此整体的时间由两者大的决定，即
$$
T = \max \{T_c, T_m\} \\ 
T_c = \frac{\# ops}{Compute}  \\
T_m = \frac{\# bytes}{Bandwidth}
$$
则如果$T_c > T_m$，有
$$
\frac{\# ops}{Compute} > \frac{\# bytes}{Bandwidth} \\
\frac{\# ops}{\# bytes} > \frac{Compute}{Bandwidth} \\
\text{arithmetic intensity} = \frac{\# ops}{\# bytes}
$$
因此，当arithmetic intensity大于零界点$\frac{Compute}{Bandwidth} $时，我们认为是计算密集型任务，反之，则是内存密集型任务，需要优化内存读取，提高计算密度。这就是roofline model。                        
实际上如果考虑L2 cache，内存带宽应该会比理论值高一些。               

https://www.53ai.com/news/qianyanjishu/1881.html

https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf

https://blog.zidea.site/p/cuda%E7%BC%96%E7%A8%8B-%E9%80%9A%E7%94%A8%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95gemm%E5%92%8Ccuda%E4%BC%98%E5%8C%96/#cuda%E6%9D%A5%E5%92%AF

https://www.zhihu.com/search?type=content&q=cuda%20gemm

https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html

### 优化出发点
![Cuda Optimization Techniques](figs/cuda-optim.png)

### 内存合并

cuda中的访存是以warp为单位的，即一个warp中的线程发出的访存指令会被合并成一个memory request，而这个request会形成一个或者多个memory transaction，分为`32-byte transaction, 64-byte transaction, 128-byte transaction`。一个线程可以一次访问1,2,4,8,16 bytes，那么一个warp就能形成32, 64, 128, 256, 512 bytes的request。然后根据内存是否连续合并成几个transaction。例如每个线程访问一个float4，如果内存连续，就需要发起4次 `128-byte transaction`。

**？待弄清楚的点：** 如何确定到达发起的是那一种 transaction。例如一个线程访问一个float，那么形成了128 bytes的request，那么是发起4次 `32-byte transaction`，还是发起1次 `128-byte transaction`。 是否和指令有关，例如使用的是`LD.32`，就只能发起`32-byte transaction`，使用的是`LD.128`，就能发起`128-byte transaction`。


使用`LD.128`的优势是什么，如果使用`LD.32`也能发起`128-byte transaction`，那么LD.128的优势是提高缓存利用？


### CUDA支持的数据类型
`float`: 对应`torch::kFloat32`

`float2`:

`float4`:

`half` (`__half`): 对应`torch::kHalf`

`half2`:

`nv_bfloat16`(`__nv_bfloat16`): 对应`torch:kBFloat16`

`nv_bfloat162`:


https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__HALF.html


| Format               | Total Bits | Sign Bit (S) | Exponent Bits (E) | Mantissa Bits (M) | Exponent Bias | Use Case                                                                 |
|----------------------|------------|--------------|-------------------|-------------------|---------------|--------------------------------------------------------------------------|
| Double (float64)     | 64         | 1            | 11                | 52                | 1023          | High precision, scientific computations                                 |
| Float (float32)      | 32         | 1            | 8                 | 23                | 127           | General-purpose floating-point arithmetic                               |
| Float16              | 16         | 1            | 5                 | 10                | 15            | Low precision, graphics, and small-scale applications                    |
| Bfloat16             | 16         | 1            | 8                 | 7                 | 127           | Machine learning and deep learning (large range, low precision)         |

十进制浮点数转化为二进制，整数部分和小数部分分别转化为二进制，然后写成科学计数法。整数部分使用余2，获得每个数；小数部分使用乘2，取整数部分。
例如，对于12.25，12的二进制为`1100`，`0.25*2=0.5，0.5*2=1.0`，所以小数部分为`01`，所以整体的二进制位`1100.01`，转为科学计数法为$1.10001*2^3$。转化为二进制：是正数，所以`S=0`，指数位为`E = 3+127=130`。这里加上的Bias，因为有负数。尾数位为`10001`，后面其他位用0补齐。这里隐藏了小数点前面的`1`，可以多表示一位。 

因此，从二进制转化位小数的公式是
$$
(-1)^S (1+M) * 2^{E-Bias}
$$
这里还是二进制的科学计数法，还需要转化为十进制的表示。

### Occupancy

#### CUDA Occupancy 的作用

CUDA Occupancy（占用率）是衡量CUDA程序中硬件资源利用率的指标，表示每个流多处理器（SM）上活跃的线程束（warps）与最大支持的线程束数量的比值。高占用率通常意味着更高的硬件资源利用率，但并非总是带来最佳性能，因为性能还受内存带宽、指令延迟等因素影响。

#### 计算方法

CUDA Occupancy 的计算公式为：

$$
\text{Occupancy} = \frac{\text{Active Warps}}{\text{Maximum Warps}}
$$

其中：
- **Active Warps** 是每个SM上活跃的线程束数量。
- **Maximum Warps** 是每个SM支持的最大线程束数量，取决于硬件架构。

#### 1. 确定每个线程的资源需求
- **线程数**：每个线程块的线程数（`threads_per_block`）。
- **寄存器数**：每个线程使用的寄存器数量（`registers_per_thread`）。
- **共享内存**：每个线程块使用的共享内存量（`shared_memory_per_block`）。

#### 2. 计算每个SM的资源限制
- **最大线程块数**：每个SM支持的最大线程块数（`max_blocks_per_sm`）。
- **最大线程数**：每个SM支持的最大线程数（`max_threads_per_sm`）。
- **最大寄存器数**：每个SM的寄存器总量（`total_registers_per_sm`）。
- **最大共享内存**：每个SM的共享内存总量（`total_shared_memory_per_sm`）。

#### 3. 计算资源限制下的线程块数
- **线程限制**：`blocks_by_threads = max_threads_per_sm / threads_per_block`
- **寄存器限制**：`blocks_by_registers = total_registers_per_sm / (registers_per_thread * threads_per_block)`
- **共享内存限制**：`blocks_by_shared_memory = total_shared_memory_per_sm / shared_memory_per_block`

#### 4. 确定实际线程块数
取上述三个限制的最小值：

$$
\text{blocks\_per\_sm} = \min(\text{blocks\_by\_threads}, \text{blocks\_by\_registers}, \text{blocks\_by\_shared\_memory})
$$

就是计算了允许的活跃的最大的block的数量，受到硬件资源限制

#### 5. 计算活跃线程束
$$
\text{Active Warps} = \text{blocks\_per\_sm} \times \left( \frac{\text{threads\_per\_block}}{\text{warpsize}} \right)
$$

block数量乘以每个block的warp的数量

#### 6. 计算占用率
$$
\text{Occupancy} = \frac{\text{Active Warps}}{\text{Maximum Warps}}
$$

#### 示例

假设：
- `threads_per_block = 256`
- `registers_per_thread = 32`
- `shared_memory_per_block = 8192` bytes
- `max_threads_per_sm = 2048`
- `total_registers_per_sm = 65536`
- `total_shared_memory_per_sm = 49152` bytes
- `warpsize = 32`
- `max_warps_per_sm = 64`

计算步骤：
1. **线程限制**：`2048 / 256 = 8`
2. **寄存器限制**：`65536 / (32 * 256) = 8`
3. **共享内存限制**：`49152 / 8192 = 6`
4. **实际线程块数**：`min(8, 8, 6) = 6`
5. **活跃线程束**：`6 * (256 / 32) = 48`
6. **占用率**：`48 / 64 = 75%`

#### 工具支持

CUDA提供了 `nvcc` 编译器的 `--ptxas-options=-v` 选项和 `CUDA Occupancy Calculator` 工具，帮助开发者分析和优化占用率。



在CUDA编程中，**Occupancy（占用率）** 是一个重要的性能优化指标，但它并不是越高越好。合适的占用率取决于具体的应用场景和硬件架构。以下是一些关于占用率的指导原则和优化建议：

---

#### 什么是合适的占用率？

1. **高占用率的优点**：
   - 更高的占用率意味着更多的活跃线程束（warps），可以更好地隐藏内存延迟和指令延迟。
   - 对于**内存密集型**（memory-bound）任务，高占用率通常有助于提高性能。

2. **高占用率的缺点**：
   - 高占用率可能导致每个线程可用的寄存器减少，从而增加寄存器溢出（register spilling），降低性能。
   - 对于**计算密集型**（compute-bound）任务，高占用率可能不会带来明显的性能提升，甚至可能因为资源竞争而降低性能。

3. **合适的占用率范围**：
   - 一般来说，**50%-75%** 的占用率是一个合理的范围。
   - 对于内存密集型任务，可以尝试接近 **75%-100%** 的占用率。
   - 对于计算密集型任务，占用率可以适当降低，重点关注指令级并行性和寄存器使用。



#### 影响占用率的因素

1. **线程块大小（Block Size）**：
   - 线程块大小直接影响占用率。较大的线程块可以提高占用率，但可能受限于寄存器或共享内存。
   - 较小的线程块可能降低占用率，但可以提高资源利用率。

2. **寄存器使用**：
   - 每个线程使用的寄存器数量越多，SM上可以同时调度的线程块越少，占用率越低。
   - 如果寄存器使用过多，可能导致寄存器溢出（register spilling），将数据存储到全局内存中，显著降低性能。

3. **共享内存使用**：
   - 每个线程块使用的共享内存越多，SM上可以同时调度的线程块越少，占用率越低。
   - 需要根据任务需求合理分配共享内存。

4. **硬件限制**：
   - 不同GPU架构的硬件资源（如寄存器数量、共享内存大小、线程束调度器数量）不同，影响占用率的上限。


#### 如何优化占用率

1. **调整线程块大小**：
   - 尝试不同的线程块大小（如128、256、512等），找到性能和占用率的最佳平衡点。
   - 使用CUDA Occupancy Calculator工具或CUDA Profiler（如Nsight Compute）分析不同线程块大小的影响。

2. **减少寄存器使用**：
   - 通过优化代码，减少每个线程的寄存器使用量。
   - 使用编译器选项（如 `-maxrregcount`）限制寄存器的使用。

3. **优化共享内存使用**：
   - 减少每个线程块的共享内存使用量。
   - 使用动态共享内存或重新设计算法以减少共享内存需求。

4. **使用CUDA工具**：
   - 使用 `nvcc` 的 `--ptxas-options=-v` 选项查看寄存器和共享内存的使用情况。
   - 使用CUDA Profiler（如Nsight Compute、nvprof）分析占用率和性能瓶颈。


#### 示例：优化占用率

假设一个CUDA内核的占用率较低（如30%），可以通过以下步骤优化：

1. **分析资源使用**：
   - 使用 `--ptxas-options=-v` 查看寄存器和共享内存的使用情况。
   - 例如：
     ```
     ptxas info    : Used 64 registers, 4096 bytes smem, 400 bytes cmem[0]
     ```

2. **调整线程块大小**：
   - 尝试将线程块大小从256调整为128或512，观察占用率和性能变化。

3. **减少寄存器使用**：
   - 通过代码优化或使用 `-maxrregcount` 选项限制寄存器使用。
   - 例如：
     ```bash
     nvcc -maxrregcount=32 -o my_program my_program.cu
     ```

4. **优化共享内存**：
   - 减少共享内存的使用量，或使用动态共享内存。

5. **验证优化效果**：
   - 使用CUDA Profiler验证优化后的占用率和性能。


#### 总结

- 合适的占用率通常在 **50%-75%** 之间，具体取决于任务类型（内存密集型或计算密集型）。
- 高占用率并不总是意味着高性能，需要综合考虑寄存器使用、共享内存使用和硬件限制。
- 通过调整线程块大小、优化寄存器和共享内存使用，可以提高占用率和性能。
- 使用CUDA工具（如CUDA Occupancy Calculator、Nsight Compute）可以帮助分析和优化占用率。

## GEMM Kerenls

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

#### CUDA 优化
1. 基础版本

2. global memory 到 shared memory                 
    对矩阵进行分块，每个block处理一块矩阵，每次加载一个块的数据到shared memory，减少从global memory读取的代价。

3. shared memory 到 register

    进一步对shared memory进行分块，减少从shared memory到register的读取时间，每个线程处理一块数据，而不是一个线程处理一个数据。

4. 解决memory bank conflict 

    同一个swap的线程访问同一个bank会产生内存冲突，从而导致访问的指令变多

5. vector load 

6. 使用预取，double buffer掩盖延迟

7. tensor core 的使用


![Cutlass Organization](figs/cutlass-levels.png)



### Vector-Matirx

## Reduction

## Softmax

## Attention

### Flash Attention

### Ring Attention






