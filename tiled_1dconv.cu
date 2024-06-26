#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>


// In verion 2, 
// Length of our convolution filter
#define FILTER_LENGTH 7

// Allocate space for the mask in constant memory
__constant__ int filter[MASK_LENGTH];

//
/*
1-D convolution kernel:
array = padded array
result = result array
n = number of elements in the array
*/
__global__ void convolution_1d_tiled (int* array, int* reuslt, int n){
    //Global thread ID calculation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // r: calculate the radius of the mask
    int r = FILTER_LENGTH/2;

    // d: the total number of padded elements
    int d = 2 * r;

    // n_padded: size of the padded shared memory array
    int n_padded = blockDim.x + d;

    // Offset for the second set of loads in shared memory
    int offset = threadIdx.x + blockDim.x

    // Global offset for the array in DRAM
    int g_offset = blockDim.x * blockIdx.x + offset;

    // Load the lower elements first starting at padding
    // This ensures divergence only eonce

    s_array[threadIdx.x] = array[tid];

    // Load in the remaining upper elements
    if(offset < n_padded){
        s_array[offset] = array[g_offset];
    }

    __syncthreads();

    //Temp value for calculation
    int temp = 0;

    // Go over each element of the mask
    for (int j = 0; j<m; j++){
        // Ignore elements that hang off
        temp += s_array[threadIdx.x + j] * filter[j];
    }
    //write back the result
    result[tid] = temp;
}

__global__ void convolution_1d_even_better(int* array, int* reuslt, int n, int m){
    //Global thread ID calculation
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the radius of the mask
    int r = m/2;

    // Calculate the starting point for the element
    int start = tid - r;

    //Temp value for calculation
    int temp = 0;

    // Go over each element of the mask
    for (int j = 0; j<m; j++){
        // Ignore elements that hang off
        if(((start + j) >= 0) && (start + j <n)){
            temp += array[start + j] * filter[j];
        }
    }
    //write back the result
    result[tid] = temp;
}

int main(){

// Number of elements in the result array
int n = 1 << 20; 

// Size of array in bytes
size_t bytes= sizeof(int) * n;

// Numbe rof elements in the convolution filter 
int m = 7;

// Size of filter in bytes
size_t bytes_m = FILTER_LENGTH * sizeof(int);

// Radius for padding the array
int r = FILTER_LENGTH / 2;
int n_p = n + r*2; // length of the array with padded radis on both sides

// Allocate the array (include edge elements)...
int* h_signal = new int[n_p];

//Intiialize the input 
for (int i=0;i<n;i++){
    if((i < r) || (i >= n_p)){
    h_signal[i] = 0;
    }
    else{
    h_signal[i] = rand() % 100;
    }
}

// Filter array
int* h_filter = new int[FILTER_LENGTH];

for (int i=0;i<m;i++){
    h_filter[i] = rand() % 10;
}

// Result array
int* h_result = new int[n];

// Allocate space on the device
int* d_signal, *d_filter, *d_result;
cudaMalloc(&d_signal,bytes_n);
cudaMalloc(&d_result,bytes_n);

//Copy the data to the device
cudaMemcpy(d_signal,h_signal,bytes_n,cudaMemcpyHostToDevice);

//Version 1
cudaMemcpy(d_filter,h_filter,bytes_n,cudaMemcpyHostToDevice);
//Version 2 & 3
cudaMemcpyToSymbol(mask,h_filter,bytes_n)


// Number of threads in the thread block
int BLOCK_SIZE = 256;

// Number of thread blocks
int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
// Amount of shared memory we need per thread block 
// padded by the overhanging radius on each side
size_t SHMEM = (BLOCK_SIZE + r*2) * sizeof(int)

convolution1d <<<GRID_SIZE,BLOCK_SIZE>>>(d_signal,d_filter,d_result,n,m);

cudaMemcpy(d_result,h_result,bytes_n,cudaMemcpyDeviceToHost);

verify_result(h_a,)

}