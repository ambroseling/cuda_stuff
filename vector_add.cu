
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

int main(){

// Vector size of 2^16 elements
int n = 1 << 16;

// Host(CPU) Vector Pointers
int *h_a, *h_b, *h_c;

// Device Vector Pointers
int *d_a, *d_b, *d_c;

size_t bytes = sizeof(int)*n;

// Allocate memory for device vectors
h_a = (int)*malloc(bytes);
h_b = (int)*malloc(bytes);
h_c = (int)*malloc(bytes);


// Allocate memory for device vectors
// GPU has its own memory that needs to be allocated
cudaMalloc(&d_a,bytes);
cudaMalloc(&d_b,bytes);
cudaMalloc(&d_c,bytes);

// Copy data from host to device
// put the contents at the address of d_a to h_a
cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice);
cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice);

// Threadblock size (32 warps, generally better if its multiple of 32)
int NUM_THREADS = 256;

// Grid size
int NUM_BLOCKS = (int)ceil(n/NUM_THREADS);

//Launch kernel on default stream w/o shared memory
vectorAdd<<<NUM_BLOCKS,NUM>>>(d_a,d_b,d_c);


cudaMemcpy(h_c,d_c,bytes,cudaMemcpyDeviceToHost);


//Check result for errors
error_check(h_a,h_b,h_c,n);

printf("Completed successfully");

return 0;


}