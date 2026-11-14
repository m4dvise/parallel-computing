#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>


__global__ void grayScale(float *a, float *b, float *c, float *d, int M, int N) {
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;

    if(row < M && col < N) {
        d[row * N + col] = 0.299f * a[row * N + col] + 0.587f * b[row * N + col] + 0.114f * c[row * N + col];
    }
}


int main(int argc, char **argv)
{
    const unsigned int M = (argc > 1)? atoi(argv[1]) : (1u << 12); // height
    const unsigned int N = (argc > 2)? atoi(argv[2]) : (1u << 12); // width

    // CPU memory pointers
    float *red = (float*)malloc(M * N * sizeof(float));
    float *green = (float*)malloc(M * N * sizeof(float));
    float *blue = (float*)malloc(M * N * sizeof(float));
    float *gray = (float*)malloc(M * N * sizeof(float));


    for(unsigned int i = 0; i < M * N; i++) {
        red[i] = (rand() % 256); // same out put every time, set seed to diff numbers if need be
        green[i] = (rand() % 256);
        blue[i] = (rand() % 256);
    }

    float *red_d, *green_d, *blue_d, *gray_d;

    // Allocate memory on GPU
    cudaMalloc((void**)&red_d, M * N * sizeof(float));
    cudaMalloc((void**)&green_d, M * N * sizeof(float));
    cudaMalloc((void**)&blue_d, M * N * sizeof(float));
    cudaMalloc((void**)&gray_d, M * N * sizeof(float));

    // Copy arrays into GPU
    cudaMemcpy(red_d, red, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(green_d, green, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(blue_d, blue, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch a grid (array of threads)
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    grayScale<<< gridSize, blockSize >>>(red_d, green_d, blue_d, gray_d, M, N);

    // Copy array from GPU
    cudaMemcpy(gray, gray_d, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    for(unsigned int i = 0; i < M * N; i++) {
        if(fabsf(gray[i] - (0.299f * red[i] + 0.587f * green[i] + 0.114f * blue[i])) > 1e-4f) {
            printf("Invalid\n");
            exit(1);
        }
    }

    printf("Valid\n");

    // Deallocate memory on GPU
    cudaFree(red_d);
    cudaFree(green_d);
    cudaFree(blue_d);
    cudaFree(gray_d);

    // Deallocate memory on CPU
    free(red);
    free(green);
    free(blue);
    free(gray);
}