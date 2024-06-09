#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

using namespace cooperative_groups;
// Cooperative groups API allows us to 


__device__ int reduce_sum (thread_group g,int * temp, int val){
    int lane = g.thread_rank();
    // Each thread adds its 
    for (int i=g.size()/2;i>0;i/=2){
        temp[lane] = val;
        //wait for threads to store
        g.sync()
        // do synchronization on this thread group
        if (lane < i){
        val += temp[lane+i];
        }
        g.sync();
    }
    return val;
}

__global__ void thread_sum(int * sum, int *input, int n){
    int sum = 0;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n /4; i += blockDim.x * gridDim.x){
        // Cast as int4
        int4 in = ((int4*) input)[i];

    }
}

__global__ void sum_reduction(int* sum, int *input, int n){
int my_sum = thread_sum(input,n);
extern __shared__ int temp[];
auto g = this_thread_block();
int block_sum = reduce_sum(g,temp,my_sum);
if (g.thread_rank() == 0){
    atomidAdd(sum,block_sum);
}
}

void intialize_vector(int* data,int n){
    for (int i=0;i<n;i++){
        data[i] = 1;
    }
}

int main(){

// Vector size
int n = 1 << 13;

size_t bytes = n * sizeof(int);

//Original vector and resulting vector
int *sum;
int *data;


// Allocate using unified memory
cudaMallocManged(&sum,sizeof(int));
cudaMallocManaged(&data,bytes);

// Initialize vector
intialize_vector(data,n);

// thread block size 
int BLOCK_SIZE = 256;

// GRID SIZE
int GRID_SIZE = (n + BLOCK_SIZE - 1)/ TB_SIZE:

// Call kernel
sum_reduction <<< GRID_SIZE, BLOCK_SIZE, n * sizeof(int) >>> (sum,data,n);
//we save 1 kernel call if we only use 

// Synchronize the kernel
cudaDeviceSynchronize();

return 0;
}