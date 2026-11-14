#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>


__global__ void addVector(float *a, float *b, float *c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < N)
        c[idx] = a[idx] + b[idx];
}

int main(int argc, char **argv)
{
    const unsigned int N = (argc > 1)? atoi(argv[1]) : (1 << 25);

    // CPU memory pointers
    float *x = (float*)malloc(N * sizeof(float));
    float *y = (float*)malloc(N * sizeof(float));
    float *z = (float*)malloc(N * sizeof(float));

    for(unsigned int i = 0; i < N; i++) {
        x[i] = rand();
        y[i] = rand();
    }

    float *x_d, *y_d, *z_d;

    // Aloocate memory on GPU
    cudaMalloc((void**)&x_d, N * sizeof(float));
    cudaMalloc((void**)&y_d, N * sizeof(float));
    cudaMalloc((void**)&z_d, N * sizeof(float));
    
    // Copy arrays into GPU
    cudaMemcpy(x_d, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Call kernel to launch grid (array of threads)
    const unsigned int blockSize = 512;
    const unsigned int gridSize = (N + blockSize - 1) / blockSize;
    
    addVector<<< gridSize, blockSize >>>(x_d, y_d, z_d, N);

    // Copy array from GPU
    cudaMemcpy(z, z_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    for(unsigned int i = 0; i < N; i++) {
        assert(z[i] == (x[i] + y[i]));
    }
    printf("Valid\n");
    
    // Dealocate memory on GPU
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
    
    // Dealocate memory on CPU
    free(x);
    free(y);
    free(z);
}