#include<cuda_runtime.h>




#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define SHARED_SIZE 4096

//float4 load data

//one warp process one row 
__global__ void __kernel_fused_softmax_warp_level(float *A, int m, int n){

    int warp_id = threadIdx.y;
    int line_id = threadIdx.x % WARP_SIZE;
    int row = blockIdx.x*blockDim.y + warp_id;

    if(row<m){
        //max
        
        
        float *Current_Row = A+row*n;
        float max_val = -INFINITY;
        float norm_sum = 0; //exp(Current_Row[0]-max_val)
        for(int i=line_id*4; i<n; i+=WARP_SIZE*4){
            if(i+3<n){
                float4 cur =  reinterpret_cast<float4 *>(&Current_Row[i])[0];
                float temp_max = max(cur.x, cur.y);
                temp_max = max(temp_max, cur.z);
                temp_max = max(temp_max, cur.w);

                if(temp_max>max_val){
                    norm_sum = norm_sum*expf(max_val-temp_max);
                    max_val = temp_max;
                }

                norm_sum += expf(cur.x - max_val);
                norm_sum += expf(cur.y - max_val);
                norm_sum += expf(cur.z - max_val);
                norm_sum += expf(cur.w - max_val);

            }else{
                for(int ii=i; ii<n; ii++){
                    float cur = Current_Row[ii];
                    if(max_val<cur){
                        norm_sum = norm_sum*expf(max_val-cur);
                        max_val = cur;
                    }
                    norm_sum += expf(cur-max_val);
                }
            }
            
        }

        //reduce the max to the first thread
        float temp_max_val = max_val;
        for(int offset=16; offset>0; offset>>=1){
            temp_max_val = max(temp_max_val, __shfl_down_sync(0xffffffff, temp_max_val, offset));
        }

        //broadcast max_val to all threads
        float global_max_val = __shfl_sync(0xffffffff, temp_max_val, 0);

        //update the norm_sum
        norm_sum = norm_sum*expf(max_val-global_max_val);

        //reduce the norm_sum to the first thread
        for(int offset=16; offset>0; offset>>=1){
            norm_sum += __shfl_down_sync(0xffffffff, norm_sum, offset);
        }

        //broadcast the normsum to all threads

        norm_sum = __shfl_sync(0xffffffff, norm_sum, 0);



        for(int i=line_id*4; i<n; i+=WARP_SIZE*4){
            if(i+3<n){
                float4 cur =  reinterpret_cast<float4 *>(&Current_Row[i])[0];
                cur.x =  expf(cur.x - global_max_val)/norm_sum;
                cur.y =  expf(cur.y - global_max_val)/norm_sum;
                cur.z =  expf(cur.z - global_max_val)/norm_sum;
                cur.w =  expf(cur.w - global_max_val)/norm_sum;

                reinterpret_cast<float4 *>(&Current_Row[i])[0] = cur;

            }else{
                for(int ii=i; ii<n; ii++){
                    float cur = Current_Row[ii];
                    Current_Row[ii] = expf(cur-global_max_val)/norm_sum;
                }
            }
            
        }

    }
}

//one block process one row when n>1024

__global__ void __kernel_fused_softmax_block_level(float *A, int m, int n){

    int tid = threadIdx.x;
    // int warp_id = tid / WARP_SIZE;
    // int line_id = tid % WARP_SIZE;
    // int warp_num = blockDim.x/WARP_SIZE;
    int row = blockIdx.x;

    if(row<m){
        float *Current_Row = A+row*n;
        __shared__ float data[BLOCK_SIZE];
        
        float max_val = -INFINITY;
        float norm_base = 0;
        for(int i=tid*4; i<n; i+=BLOCK_SIZE*4){
            
            if(i+3<n){
                float4 cur =  reinterpret_cast<float4 *>(&Current_Row[i])[0];
                float temp_max = max(cur.x, cur.y);
                temp_max = max(temp_max, cur.z);
                temp_max = max(temp_max, cur.w);

                if(temp_max>max_val){
                    norm_base = norm_base*expf(max_val-temp_max);
                    max_val = temp_max;
                }

                norm_base += expf(cur.x - max_val);
                norm_base += expf(cur.y - max_val);
                norm_base += expf(cur.z - max_val);
                norm_base += expf(cur.w - max_val);

            }else{
                for(int ii=i; ii<n; ii++){
                    float cur = Current_Row[ii];
                    if(max_val<cur){
                        norm_base = norm_base*expf(max_val-cur);
                        max_val = cur;
                    }
                    norm_base += expf(cur-max_val);
                }
            }

           
        }

        data[tid] = max_val;
        __syncthreads();
        for(int offset=BLOCK_SIZE/2; offset>0; offset>>=1){
            if(tid<offset){
                data[tid] = max(data[tid], data[tid+offset]);
            }
            __syncthreads();
        }

        float global_max_val = data[0];
         __syncthreads();
        norm_base = norm_base*expf(max_val-global_max_val);
        
        //reduce the norm_base
        data[tid] = norm_base;

        __syncthreads();
        for(int offset=BLOCK_SIZE/2; offset>0; offset>>=1){
            if(tid<offset){
                data[tid] += data[tid+offset];
            }
            __syncthreads();
        }

        norm_base = data[0];

        for(int i=tid*4; i<n; i+=BLOCK_SIZE*4){
            // float cur = Current_Row[i];
            // Current_Row[i] = expf(cur-global_max_val)/norm_base;


            if(i+3<n){
                float4 cur =  reinterpret_cast<float4 *>(&Current_Row[i])[0];
                cur.x =  expf(cur.x - global_max_val)/norm_base;
                cur.y =  expf(cur.y - global_max_val)/norm_base;
                cur.z =  expf(cur.z - global_max_val)/norm_base;
                cur.w =  expf(cur.w - global_max_val)/norm_base;

                reinterpret_cast<float4 *>(&Current_Row[i])[0] = cur;

            }else{
                for(int ii=i; ii<n; ii++){
                    float cur = Current_Row[ii];
                    Current_Row[ii] = expf(cur-global_max_val)/norm_base;
                }
            }
        }


    }



}


// template<const int BS=256>
// __global__ void fused_softmax(float *A, float *B, int m, int n){
//     int row = blockIdx.x;
//     int tid = threadIdx.x;
//     // int block_size = blockDim.x;

//     float *Current_rowA = A+row*n;
//     float *Current_rowB = B+row*n;

//     float max_val=-INFINITY;
//     float norm_base = 0;
//     float4 temp;

//     __shared__ float data[BS];
//     for(int i=tid*4; i<n; i+=BS*4){
//         temp = ((float4 *)(&Current_rowA[i]))[0];
//         float temp_max = max(temp.x, temp.y);
//         temp_max = max(temp.z, temp_max);
//         temp_max = max(temp.w, temp_max);
//         if(temp_max>max_val){
//             norm_base = norm_base*exp(max_val-temp_max);
//             max_val = temp_max;
//         } 

//         norm_base += exp(temp.x-max_val);
//         norm_base += exp(temp.y-max_val);
//         norm_base += exp(temp.z-max_val);
//         norm_base += exp(temp.w-max_val);

//     }

//     //reduce the max_val with shared memory
//     data[tid] = max_val;
//     __syncthreads();

    
//     for(int offset=BS/2; offset>0; offset>>=1){
//         if(tid<offset){
//             data[tid] = max(data[tid], data[tid+offset]);
//         }

//         __syncthreads();
//     }

//     float global_max = data[0];

//     //correct the norm base
//     norm_base = norm_base*exp(max_val-global_max);

//     //reduce the norm base 
//     __syncthreads();
//     data[tid] = norm_base;
//     __syncthreads();
//     //reduce the norm base
//     for(int offset=BS/2; offset>0; offset>>=1){
//         if(tid<offset){
//             data[tid] = data[tid] + data[tid+offset];
//         }

//         __syncthreads();
//     }

//     norm_base = data[0]; //get the right normal base


//     //compute the results

    
//     for(int i=tid*4; i<n; i+=BS*4){
//         temp = ((float4 *)(&Current_rowA[i]))[0];
        
//         temp.x = exp(temp.x-global_max)/norm_base;
//         temp.y = exp(temp.y-global_max)/norm_base;
//         temp.z = exp(temp.z-global_max)/norm_base;
//         temp.w = exp(temp.w-global_max)/norm_base;

        
//         ((float4 *)(&Current_rowB[i]))[0] = temp;

//     }



// }





















// use shared memory to load data first when n<=4096
__global__ void __kernel_fused_softmax_block_level_shared(float *A, int m, int n){

    int tid = threadIdx.x;
    // int warp_id = tid / WARP_SIZE;
    // int line_id = tid % WARP_SIZE;
    // int warp_num = blockDim.x/WARP_SIZE;
    int row = blockIdx.x;

    if(row<m){
        float *Current_Row = A+row*n;
        __shared__ float data[BLOCK_SIZE];

        __shared__ float AR[SHARED_SIZE];
        for(int i=tid*4; i<n; i+=BLOCK_SIZE*4){
            if(i+3<n){
                reinterpret_cast<float4 *>(&AR[i])[0] = reinterpret_cast<float4 *>(&Current_Row[i])[0];
            }else{
                for(int ii=i; ii<n; ii++)
                    AR[ii] = Current_Row[ii]; 
            }
            
        }
        __syncthreads();
        
        float max_val = -INFINITY;
        float norm_base = 0;
        for(int i=tid; i<n; i+=BLOCK_SIZE){
            float cur = AR[i];
            if(cur>max_val){
                norm_base = norm_base * expf(max_val-cur);
                max_val = cur;
            }

            norm_base += expf(cur-max_val);
        }

        data[tid] = max_val;
        __syncthreads();
        for(int offset=BLOCK_SIZE/2; offset>0; offset>>=1){
            if(tid<offset){
                data[tid] = max(data[tid], data[tid+offset]);
            }
            __syncthreads();
        }

        float global_max_val = data[0];

        norm_base = norm_base*expf(max_val-global_max_val);
        
        //reduce the norm_base
        data[tid] = norm_base;

        __syncthreads();
        for(int offset=BLOCK_SIZE/2; offset>0; offset>>=1){
            if(tid<offset){
                data[tid] += data[tid+offset];
            }
            __syncthreads();
        }

        norm_base = data[0];

        for(int i=tid; i<n; i+=BLOCK_SIZE){
            
            // float cur = AR[i];
            // Current_Row[i] = expf(cur-global_max_val)/norm_base;

            AR[i] = expf(AR[i]-global_max_val)/norm_base;
        }

        __syncthreads();


        for(int i=tid*4; i<n; i+=BLOCK_SIZE*4){
            if(i+3<n){
                reinterpret_cast<float4 *>(&Current_Row[i])[0] = reinterpret_cast<float4 *>(&AR[i])[0];
            }else{
                for(int ii=i; ii<n; ii++)
                    Current_Row[ii] = AR[ii]; 
            }
            
        }

    }



}




void fused_softmax(float *A, int m, int n){
    int warp_level_n = 1024;
    if(n<=warp_level_n){
         int row_per_block = 4;
        dim3 block(WARP_SIZE, row_per_block);
        dim3 grid((m+row_per_block-1)/row_per_block);

        __kernel_fused_softmax_warp_level<<<grid, block>>>(A, m, n);
    }else if(n<=SHARED_SIZE){
        dim3 block(BLOCK_SIZE);
        dim3 grid(m);
        __kernel_fused_softmax_block_level_shared<<<grid, block>>>(A, m, n);
    }else{
        dim3 block(BLOCK_SIZE);
        dim3 grid(m);
        __kernel_fused_softmax_block_level<<<grid, block>>>(A, m, n);
    }
   
    
}