/*
 * See 7_CUDALibraries/simpleDevLibCUBLAS/ in NVIDIA CUDA Samples for more details.
 */
#include <cublas_v2.h>

extern "C"
__global__ void test(int matrix_side,
                     const float *alpha,
                     const float *A,
                     const float *B,
                     const float *beta,
                     float *C) {
  cublasHandle_t blas;
  cublasCreate(&blas); 

  cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, matrix_side, matrix_side, matrix_side,
              alpha, A, matrix_side, B, matrix_side, beta, C, matrix_side);

  cublasDestroy(blas);
}
