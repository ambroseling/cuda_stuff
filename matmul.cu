#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void matmul(int*a, int*b,int*c,int n){

    //Compute the indexes for rows and columns
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Boundary protection, no accessing memory out of bounds
    if((row<n) && (col < n)){
        // loop through row of A and col of B
    for (int i=0;i<n;i++){
        c[row*n+col] += a[row*n+col] + b[row*n+col];
    }
    }

}


void verify_results(int*c,int n){
    int *verify_c;
    // go through all the rows
    for (int i=0;i<ni++){
        // go through all the columns
          for (int j=0;j<nj++){
            // go through all the elements in the rows and columns
            for (int k=0;k<n;k++){
       //2d indexing with pointers
        verify_c[i*n+j] += a[i*n+j]* b[k*n+j];
    }
    }  
    }

    for (int i=0;i<ni++){
        for (int j=0;j<nj++){
    assert(verify_c[i*n+j] == c[i*n+j]);
    }}


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
matMul <<<grid,threads>>> (d_a,d_b,d_c,n);

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