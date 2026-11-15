#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>

#define GREEN   "\033[0;32m"
#define RESET   "\033[0m"


__global__ void matrixMul(float *matrix_a, float *matrix_b, float *matrix_c, unsigned int M, unsigned int K, unsigned int N) {

    unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row >= M || col >= N) 
        return;

    float acc = 0.0f;
    for(unsigned int i = 0; i < K; i++) {
        acc += matrix_a[row * K + i] * matrix_b[i * N + col];
    }

    matrix_c[row * N + col] = acc;
}


int main(int argc, char **argv)
{
    const unsigned int M = (argc > 1)? atoi(argv[1]) : (1u << 12);
    const unsigned int K = (argc > 2)? atoi(argv[2]) : (1u << 12);
    const unsigned int N = (argc > 3)? atoi(argv[3]) : (1u << 12);

    // matrix_a --> M x K, matrix_b --> K x N
    // Pointers to CPU memory
    float *matrix_a = (float*)malloc(M * K * sizeof(float));
    float *matrix_b = (float*)malloc(K * N * sizeof(float));
    float *matrix_c = (float*)malloc(M * N * sizeof(float));

    for(int i = 0; i < M * K; i++) {
        matrix_a[i] = rand();
    }

    for(int i = 0; i < K * N; i++) {
        matrix_b[i] = rand();
    }

    // Allocate memory on GPU
    float *matrix_a_d, *matrix_b_d, *matrix_c_d;

    cudaMalloc((void**)&matrix_a_d, M * K * sizeof(float));
    cudaMalloc((void**)&matrix_b_d, K * N * sizeof(float));
    cudaMalloc((void**)&matrix_c_d, M * N * sizeof(float));

    // Copy arrays on GPU
    cudaMemcpy(matrix_a_d, matrix_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_b_d, matrix_b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch a grid on GPU
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    cudaEvent_t start, stop;
    cudaEventCreateWithFlags(&start, cudaEventBlockingSync); // on a time sliced shared GPU :)
    cudaEventCreateWithFlags(&stop, cudaEventBlockingSync);
    float ms = 0.0f;
    cudaEventRecord(start, 0);
    matrixMul <<< gridSize, blockSize >>>(matrix_a_d, matrix_b_d, matrix_c_d, M, K, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);

    printf(GREEN "Time taken by GPU: %f ms\n" RESET, ms);

    // Copy arrays from GPU
    cudaMemcpy(matrix_c, matrix_c_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate memory on GPU
    cudaFree(matrix_a_d);
    cudaFree(matrix_b_d);
    cudaFree(matrix_c_d);

    // Dealocate memory on CPU
    free(matrix_a);
    free(matrix_b);
    free(matrix_c);

    return 0;
}