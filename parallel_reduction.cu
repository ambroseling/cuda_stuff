#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>

#define SHMEM_SIZE 16 * 4

//---------- CUDA kernel for parallel reduction v1 ----------
__gloabl__ void parallelSum(int* v_r){

__shared__ int partial_sum[SHMEM_SIZE];

//Calculate the thread Id
int tid = blockIdx.x * blockDim.x + threadIdx.x;

//Load elements into shared memory
partial_sum[threadIdx.x] = v[tid];

//Synchronization point
__syncthreads();

// Iterate of log base 2 the block dimension
for(int s=1;s<blockDim.x;s*=2){
    // take only the even threads at each stride
    // but modulo operation is expensive
    if(threadIdx.x % (2 * s) ==0){
        partial_sum[threadIdx.x] += partial_sum[threadIdx.x+s];

    }
}

// Let the thread 0 for this block write its result to main memory
// result is indexed by this block
if(threadIdx.x == 0){
 v_r[blockIdx.x] = partial_sum[0];
}

}

//---------- CUDA kernel for parallel reduction v2 ----------
__gloabl__ void parallelSumBetter(int* v_r){

__shared__ int partial_sum[SHMEM_SIZE];

//Calculate the thread Id
int tid = blockIdx.x * blockDim.x + threadIdx.x;

//Load elements into shared memory
partial_sum[threadIdx.x] = v[tid];

//Synchronization point
__syncthreads();

// Iterate of log base 2 the block dimension
for(int s=1;s<blockDim.x;s*=2){
    int index = 2 * s * threadIdx.x;
    if (index < blockDim.x){
    partial_sum[index]  += partial_sum[index+s];
    }
}

// Let the thread 0 for this block write its result to main memory
// result is indexed by this block
if(threadIdx.x == 0){
 v_r[blockIdx.x] = partial_sum[0];
}

}


//---------- CUDA kernel for parallel reduction v3 ----------
__gloabl__ void parallelSumEvenBetter(int* v_r){

__shared__ int partial_sum[SHMEM_SIZE];

//Calculate the thread Id
int tid = blockIdx.x * blockDim.x + threadIdx.x;

//Load elements into shared memory
partial_sum[threadIdx.x] = v[tid];

//Synchronization point
__syncthreads();

// Iterate of log base 2 the block dimension
for(int s=blockDim.x;s > 0;s >>=1){
    // each thread does work unless the index goes off the block
    // we only use threads that are the lower half of the block
   if(threadIdx.x < s){
    partial_sum[threadIdx.x] += partial_sum[threadIdx.x+s];
   }
}

// Let the thread 0 for this block write its result to main memory
// result is indexed by this block
if(threadIdx.x == 0){
 v_r[blockIdx.x] = partial_sum[0];
}

}


//---------- CUDA kernel for parallel reduction v4 ----------
// (same as v3 but less thread blocks launched)
__gloabl__ void parallelSumEvenEvenBetter(int* v_r){

__shared__ int partial_sum[SHMEM_SIZE];

// Calculate the thread Id
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// Load elements AND do first add of reduction
// Vector now 2x as long as number of threads
int i = blockIdx.x * (blockDim.x *2) + threadIdx.x;


partial_sum[threadIdx.x] = v[i] + v[i+blockDim.x];

//Synchronization point
__syncthreads();

// Iterate of log base 2 the block dimension
for(int s=blockDim.x/2;s > 0;s >>=1){
    // each thread does work unless the index goes off the block
    // we only use threads that are the lower half of the block
   if(threadIdx.x < s){
    partial_sum[threadIdx.x] += partial_sum[threadIdx.x+s];
   }
   __syncthreads();
}

// Let the thread 0 for this block write its result to main memory
// result is indexed by this block
if(threadIdx.x == 0){
 v_r[blockIdx.x] = partial_sum[0];
}
}

//---------- CUDA kernel for parallel reduction v5 ----------
// (same as v3 but less thread blocks launched)

__device__ void warpReduce(int tid, int* shmem_ptr, int t){
shmem_ptr[t] += shmem_ptr[t+32];
shmem_ptr[t] += shmem_ptr[t+16];
shmem_ptr[t] += shmem_ptr[t+8];
shmem_ptr[t] += shmem_ptr[t+4];
shmem_ptr[t] += shmem_ptr[t+2];
shmem_ptr[t] += shmem_ptr[t+1];
}

__gloabl__ void parallelSumEvenEvenBetter(int* v_r){

__shared__ int partial_sum[SHMEM_SIZE];

// Calculate the thread Id
int tid = blockIdx.x * blockDim.x + threadIdx.x;

// Load elements AND do first add of reduction
// Vector now 2x as long as number of threads
int i = blockIdx.x * (blockDim.x *2) + threadIdx.x;


partial_sum[threadIdx.x] = v[i] + v[i+blockDim.x];

//Synchronization point
__syncthreads();

// Iterate of log base 2 the block dimension
for(int s=blockDim.x/2;s > 32;s >>=1){
    // each thread does work unless the index goes off the block
    // we only use threads that are the lower half of the block
   if(threadIdx.x < s){
    partial_sum[threadIdx.x] += partial_sum[threadIdx.x+s];
   }
   __syncthreads();
}

if (threadIdx.x < 32){
    warpReduce(tid,partial_sum);
}
// Let the thread 0 for this block write its result to main memory
// result is indexed by this block
if(threadIdx.x == 0){
 v_r[blockIdx.x] = partial_sum[0];
}
}


int main(){

// Vector size of 2^16 elements
int n = 1 << 16;

// Host(CPU) Vector Pointers
int *h_v;
int *h_v_r;
// Device Vector Pointers
int *d_v;
int *d_v_r;

size_t bytes = sizeof(int)*n;

// Allocate memory for device vectors
h_v = (int)*malloc(bytes);
h_v_r = (int)*malloc(bytes);

// Allocate memory for device vectors
// GPU has its own memory that needs to be allocated
cudaMalloc(&d_v,bytes);
cudaMalloc(&d_v,bytes);


// Copy data from host to device
// put the contents at the address of d_a to h_a
cudaMemcpy(d_v,h_v,bytes,cudaMemcpyHostToDevice);

// Threadblock size (32 warps, generally better if its multiple of 32)
int BLOCK_SIZE = 256;

// Grid size
int GRID_SIZE = (int)ceil(n/NUM_THREADS);
// For v4  (uncomment), we halve the number of threads
int GRID_SIZE = (int)ceil(n/NUM_THREADS/2);

//Launch kernel on default stream w/o shared memory
sum_reduction<<BLOCK_SIZE,GRID_SIZE>>(d_v,d_v_r);

//
sum_reduction<<1,GRID_SIZE>>(d_v,d_v_r);


cudaMemcpy(h_v_r,d_v_r,bytes,cudaMemcpyDeviceToHost);


//Check result for errors

printf("Completed successfully");

return 0;


}