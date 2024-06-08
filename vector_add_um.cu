
#include <cuda_runtime.h>
#include <device_launch_parameters.h>



// CUDA kernel for vector addition
__global__ void vectorAdd(int*a, int*b, int*c, int n){
    /**
     * @brief 
     * Inputs: vectors a and b
     * Outputs: vectors c
     */
    //Calculate global thread ID (tid)
    // blockIdx.x : which block am i in
    // blockDim.x : the block size
    // threadId.x : which thread inside the block
    int tid = (blockIdx.x * blockDim.x)+ threadId.x
    // Vector boundary guard
    // make sure we're not accessing out of bound memory
    if (tid < n){
        //
        c[tid] = a[tid] + b[tid];
    }
}

void errorCheck(int* a, int* b, int*c, n){
    for(int i =0 ;i<n;i++){
        assert(c[i] = a[i] + b[i]);
    }
}

int main(){

// Vector size of 2^16 elements
int id = cudaGetDevice(&id);

// Declare number of elements per- array
int n = 1 << 16;

// Size of each arrays in bytes
size_t bytes = n * sizeof(int);

// Unified Memory pointers
// these pointers can be accessed on both sides (CPU and GPU)
int *a, *b, *c;

size_t bytes = sizeof(int)*n;

// Allocate memory for device vectors
// GPU has its own memory that needs to be allocated
cudaMallocManaged(&a,bytes);
cudaMallocManaged(&b,bytes);
cudaMallocManaged(&c,bytes);

// Threadblock size (32 warps, generally better if its multiple of 32)
// Block size
int NUM_THREADS = 256;

// Grid size
int NUM_BLOCKS = (int)ceil(n/NUM_THREADS);

// prefetch the vectors a and b to device
cudaMemPrefetchAsync(a,bytes,id);
cudaMemPrefetchAsync(b,bytes,id);

//Launch kernel on default stream w/o shared memory
//         block size , grid size
vectorAdd<<<NUM_BLOCKS,NUM_THREADS>>>(a,b,c,n);

// this marks that all events on the GPU is complete
// wait for all previous operations before using values
cudaDeviceSynchronize();

// prefetching c to the host
cudaMemPrefetchAsync(c,bytes,cudaDeviceId);

//Check result for errors
error_check(h_a,h_b,h_c,n);

printf("Completed successfully");

return 0;


}