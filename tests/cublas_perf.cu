#include <cublas_v2.h>


extern "C"
__global__ void create_cublas(cublasHandle_t *pBlas) {
  cublasCreate(pBlas);
}


extern "C"
__global__ void destroy_cublas(const cublasHandle_t blas) {
  cublasDestroy(blas);
}


extern "C"
__global__ void test(const int N,
                     const cublasHandle_t blas,
                     const int matrix_side,
                     const float *alpha,
                     const float *A,
                     const float *B,
                     const float *beta,
                     float *C) {
  for (int i = 0; i < N; i++) {
    cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, matrix_side, matrix_side, matrix_side,
                alpha, A, matrix_side, B, matrix_side, beta, C, matrix_side);
  }
}


extern "C"
__global__ void test_full(
                     const int N,
                     const int matrix_side,
                     const float *alpha,
                     const float *A,
                     const float *B,
                     const float *beta,
                     float *C) {
  for (int i = 0; i < N; i++) {
    cublasHandle_t blas;
    cublasCreate(&blas);
    cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, matrix_side, matrix_side, matrix_side,
                alpha, A, matrix_side, B, matrix_side, beta, C, matrix_side);
    cublasDestroy(blas);
  }
}


extern "C"
__global__ void dummy(const float *in, float *out) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  out[idx] = in[idx];
}
