#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>


// Length of our convolution filter 7 x 7 conv filter
#define FILTER_LENGTH 7
#define FILTER_OFFSET (FILTER_LENGTH /2)
// Allocate space for the mask in constant memory
__constant__ int filter[FILTER_LENGTH*FILTER_LENGTH];

//
/*
2-D convolution kernel:
matrix = Input matrix
result = Convolution result array
n = number of elements in the array
*/
__global__ void convolution_2d(int* matrix, int* reuslt, int n, int m){
    //Global thread ID calculation
    //Global thread ID calculation
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Starting index for calculation
    int start_r = row - FILTER_OFFSET;
    int start_c = col - FILTER_OFFSET;

    // Temp value for accumulating the resut
    int temp = 0;

    // Iterate through all the rows and the columns of the filter
    for (int i =0; i< FILTER_LENGTH;i++){
        for (int j=0;j<FILTER_LENGTH;j++){
            if((start_r + i) >= 0 || (start_r + i) < n){
            if((start_c + j) >= 0 || (start_c + j) < n){
            filter += mask[i*FILTER_LENGTH+j] * matrix[(start_r +i)*n+(start_c+j)];
            }
            }
        }
    }

    //write back the result
    result[row*n+col] = temp;
}

void verify_result(int* matrix, int* result,int n){
    int offset_r;
    int offset_c;
    for(int i=0;i<n;i++){
        for (int j=0;j<n;j++){
            int temp = 0
            for(int k=0;k<FILTER_LENGTH;k++){
                offset_r = i - FILTER_LENGTH/2;
                for(l = 0;l<FILTER_LENGTH;l++){
                offset_c = j = FILTER_LENGTH/2;
                if((i - offset_r) >=0 || (i - offset_r) < n){
                if((j - offset_c) >=0 || (j - offset_c) < n){
                    temp += matrix[(offset_r*n)+offset_c] * filter[k*FILTER_LENGTH+j];
                }
                }
                }
            }
        }
    }
}

int main(){

// Number of elements in the result array
int n = 1 << 10; 

// Size of array in bytes
size_t bytes_n = sizeof(int) * n * n;

// Numbe rof elements in the convolution filter 
int m = 7;

// Size of filter in bytes
size_t bytes_m = FILTER_LENGTH * FILTER_LENGTH * sizeof(int);

// Radius for padding the array
int r = FILTER_LENGTH / 2;
int n_p = n + r*2; // length of the array with padded radius on both sides

// Allocate the array (include edge elements)...
int* h_signal = new int[n_p * n_p];

//Intiialize the input matrix
for(int i=0;i<n;i++){
for (int j=0;j<n;j++){
    if((i < r) || (i >= n_p)){
        if((j < r) || (j >= n_p)){
    h_signal[i*n+j] = 0;
        }
    }
    else{
    h_signal[i] = rand() % 100;
    }
}

}

// Filter matrix
int* h_filter = new int[FILTER_LENGTH * FILTER_LENGTH];

for (int i=0;i<FILTER_LENGTH;i++){
    for(int j = 0;j<FILTER_LENGTH;j++){
    h_filter[i] = rand() % 10;
    }
}

// Result array
int* h_result = new int[n];

// Allocate space on the device
int* d_signal, *d_filter, *d_result;
cudaMalloc(&d_signal,bytes_n);
cudaMalloc(&d_result,bytes_n);

//Copy the data to the device
cudaMemcpy(d_signal,h_signal,bytes_n,cudaMemcpyHostToDevice);
cudaMemcpyToSymbol(filter,h_filter,bytes_n)


// Number of threads in the thread block
int BLOCK_SIZE = 16;

// Number of thread blocks
int GRID_SIZE = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

dim3 block_dim(THREADS,THREADS);
dim3 grid_dim(GRID_SIZE,GRID_SIZE);

convolution_2d <<<grid_dim,block_dim>>>(d_signal,d_result,n,m);

cudaMemcpy(d_result,h_result,bytes_n,cudaMemcpyDeviceToHost);

verify_result(h_a,)

}