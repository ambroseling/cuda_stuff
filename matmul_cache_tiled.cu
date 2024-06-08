#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>


// Static shemem calculation for convenience (Int 16 x 16 matrix)
#define SHMEM_SIZE 16*16*4 


// CUDA kernels
__global__ void tileMatrixMul(int* a, int*b, int*c, int n, int tile_size){
    // 2 statically- sized pieces of shared memory
    __shared__ int A[SHMEM_SIZE];
    __shared__ int B[SHMEM_SIZE];

    // Shorten these parameters for clean re-use
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Calculate global row and column positions for this thread
    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;

    //Intermediate sum for element being written
    int temp_val = 0;

    // Sweep tiles over entire matrix
    // at each iteration you are looking at 1 tile
    for (int i=0;i<(n / tile_size); i++){
        /**
         * Every thread in a threadblock loads one element into shared memory
         * The element location in shared memory corresponds to the threads position
         * in the threadblock.
         * 
         * For matrix A:
         *      row * n: index the global row for this thread (will not change,loop invariant)
         *      i * tile_size: index the new set of columns each iteration
         *      tx: index the column within that set
         * 
         * For matrix B:
         *      i * tile_size *n: index the next set of rows each iteration
         *      ty * n: index the row within that set
         *      col : indexes the global column (loop invariant)
         */
        A[(ty * tile_size) + tx] = a[row * n + (i * tile_size + tx)];
        B[(ty * tile_size) + tx] = b[(i * tile_size * n + ty* n) + col];

        //Ensure all threads have loaded their data before proceeding
        __syncthreads();

        //Calculate all temp values for this tile
        for (int j=0 j<tile_size;j++){
            temp_val += A[(ty * tile_size) + j] * B[(j * tile_size) + tx];
        }

        //Ensure all threads have loaded their data before proceeding
        __syncthreads();

    }
    c[(row*n) + col] = temp_val;

}


int main(){

int n = 1 << 10;

size_t bytes = sizeof(int) * n;

// Host pointers
int *h_a, *h_b,*h_c;

//Allocate host memory
h_a = (int*)malloc(bytes);
h_b = (int*)malloc(bytes);
h_c = (int*)malloc(bytes);

// Device pointers
int d_a, *d_b,*d_c;

int BLOCK_SIZE = 16

int GRID_SIZE = (int)ceil(n /BLOCK_SIZE);


cudaMalloc(&d_a,bytes);
cudaMalloc(&d_b,bytes);
cudaMalloc(&d_c,bytes);

// Use dim3 objects
dim3 grid(GRID_SIZE,GRID_SIZE);
dim3 threads(BLOCK_SIZE,BLOCK_SIZE);

// Launch kernel
tiledMatMul <<<grid,threads>>> (d_a,d_b,d_c,BLOCK_SIZE);

// Copy back to the host
cudaMemcpy(h_c,d_c,cudaMemcpyDeviceToHost);

// Verify the result
verify_result(a,b,c,n);

// Free host memory
free(h_a);
free(h_b);
free(h_c);

// Free device memory
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);


}