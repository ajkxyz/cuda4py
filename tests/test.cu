__device__ float g_a = 0;

extern "C" __global__ void test(float *a, float *b, const float c) {
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  a[i] += b[i] * c;
}
