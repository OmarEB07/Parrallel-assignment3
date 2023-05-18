#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matrixMultiplication(float* a, float* b, float* c, int M, int N)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0.0;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        if (row < M && t * TILE_SIZE + tx < N) {
            tile_a[ty][tx] = a[row * N + t * TILE_SIZE + tx];
        } else {
            tile_a[ty][tx] = 0.0;
        }

        if (col < N && t * TILE_SIZE + ty < N) {
            tile_b[ty][tx] = b[(t * TILE_SIZE + ty) * N + col];
        } else {
            tile_b[ty][tx] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tile_a[ty][k] * tile_b[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}

int main()
{
    int M = 1000;
    int N = 500;

    float* a, * b, * c;
    float* d_a, * d_b, * d_c;
    int size_a = M * N * sizeof(float);
    int size_b = N * N * sizeof(float);
    int size_c = M * N * sizeof(float);

    // Allocate memory on host
    a = (float*)malloc(size_a);
    b = (float*)malloc(size_b);
    c = (float*)malloc(size_c);

    // Initialize input matrices
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            a[i * N + j] = i + j;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            b[i * N + j] = i - j;
        }
    }

    // Allocate memory on device
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    // Copy input matrices from host to device
    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);

    // Define grid and block sizes
    dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, 1);
    dim3 dimBlock(TILE_SIZE, TILE_SIZE, 1);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);

    // Call kernel function
    matrixMultiplication<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, N);

    // Record stop event
    cudaEventRecord(stop);

    // Wait for the completion of all device operations
    cudaDeviceSynchronize();

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy output matrix from device to host
    cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);

// Print output matrix
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", c[i * N + j]);
        }
        printf("\n");
    }

    // Print execution time
    printf("Execution time: %.2f ms\n", milliseconds);

    // Free memory
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
