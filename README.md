# cuda_stuff

This repo will document my journey of learning CUDA, some things relating to GPU programming.

### Background

SIMT Model 
Single Instruction Multiple Threads (SIMT)
- Parallel Vector Addition


Threads
- lowest granularity of execution
- executes instruction

Warps 
- lowest **schedulable** entity
- executes instructions in lock-step
- not every thread needs to execute all instructions

Thread Blocks
- what we program in
- is assigned to a single shader core (cores of the GPU)
- can be 3D

Grids
- how a computation problem is mapped to the GPU
- Part of the GPU launch parameters
    - Grid size: how many thread blocks are in a grid
    - Thread-block size: how many threads are within a thread block 

Matrix multiplication
- Basic flow:
    - assign a thread for each element of C
    - each thread traverses 1 row of A, and one column of B
    - each thread writes the results to its assigned element of C

2-D indexing for threadblocks
- tiny 2x2 thread blocks
- blockIdx and threadIdx in both x and y dimensions
- blockDim is constant in both x and y dimensions

```cuda
// which block                which thread
Row = blockIdx.y * blockDim.y + threadIdx.y
Col = blockIdx.x * blockDim.x + threadIdx.x
```

### Thread hierarchy
How do we index a thread?
``


### Cache Tiling
- DRAM is slow, we want to minimze the number of memory related stalls
- more computation / unit time
- Use shared memory, user managed L1 Cache
- Tiled matrix multiplication:
    - Naive: each thread takes care of 1 entry of the output matrix
    - Tiled: 
        - calculte index for loading into shared memory
        - A[y][k] * B[k][x]
        - constant row, loop varuing column
        - constant column, loop varuing row

### Coalescing
- Row major order (one linear set of addresses)
- Aligned accesses:
    - each thread accesses a different column
    - columns are adjacent in memory
    - multiple adjacent accesses can be coalesced into a single wide access
    - rows are **NOT** adjacent in memory 
    - multiple accesses to memory that are independent 