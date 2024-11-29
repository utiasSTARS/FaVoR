#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

template <typename scalar_t>
__global__ void adam_upd_cuda_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ exp_avg,
    scalar_t* __restrict__ exp_avg_sq,
    const size_t N,
    const double step_size, const double beta1, const double beta2, const double eps) {

  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<N) {
    exp_avg[index] = beta1 * exp_avg[index] + (1-beta1) * grad[index];
    exp_avg_sq[index] = beta2 * exp_avg_sq[index] + (1-beta2) * grad[index] * grad[index];
    param[index] -= step_size * exp_avg[index] / (sqrt(exp_avg_sq[index]) + eps);
  }
}

template <typename scalar_t>
__global__ void masked_adam_upd_cuda_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ exp_avg,
    scalar_t* __restrict__ exp_avg_sq,
    const size_t N,
    const double step_size, const double beta1, const double beta2, const double eps) {

  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<N && grad[index]!=0) {
    exp_avg[index] = beta1 * exp_avg[index] + (1-beta1) * grad[index];
    exp_avg_sq[index] = beta2 * exp_avg_sq[index] + (1-beta2) * grad[index] * grad[index];
    param[index] -= step_size * exp_avg[index] / (sqrt(exp_avg_sq[index]) + eps);
  }
}

template <typename scalar_t>
__global__ void adam_upd_with_perlr_cuda_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ exp_avg,
    scalar_t* __restrict__ exp_avg_sq,
    scalar_t* __restrict__ perlr,
    const size_t N,
    const double step_size, const double beta1, const double beta2, const double eps) {

  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<N) {
    exp_avg[index] = beta1 * exp_avg[index] + (1-beta1) * grad[index];
    exp_avg_sq[index] = beta2 * exp_avg_sq[index] + (1-beta2) * grad[index] * grad[index];
    param[index] -= step_size * perlr[index] * exp_avg[index] / (sqrt(exp_avg_sq[index]) + eps);
  }
}

void adam_upd_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    const int64_t step, const double beta1, const double beta2, const double lr, const double eps) {

  const size_t N = param.numel();

  const int64_t threads = 256;
  const int64_t blocks = (N + threads - 1) / threads;

  const double step_size = lr * sqrt(1 - pow(beta2, (double)step)) / (1 - pow(beta1, (double)step));

  AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "adam_upd_cuda", ([&] {
    adam_upd_cuda_kernel<scalar_t><<<blocks, threads>>>(
        param.data_ptr<scalar_t>(),
        grad.data_ptr<scalar_t>(),
        exp_avg.data_ptr<scalar_t>(),
        exp_avg_sq.data_ptr<scalar_t>(),
        N, step_size, beta1, beta2, eps);
  }));
}

void masked_adam_upd_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    const int64_t step, const double beta1, const double beta2, const double lr, const double eps) {

  const size_t N = param.numel();

  const int64_t threads = 256;
  const int64_t blocks = (N + threads - 1) / threads;

  const double step_size = lr * sqrt(1 - pow(beta2, (double)step)) / (1 - pow(beta1, (double)step));

  AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "masked_adam_upd_cuda", ([&] {
    masked_adam_upd_cuda_kernel<scalar_t><<<blocks, threads>>>(
        param.data_ptr<scalar_t>(),
        grad.data_ptr<scalar_t>(),
        exp_avg.data_ptr<scalar_t>(),
        exp_avg_sq.data_ptr<scalar_t>(),
        N, step_size, beta1, beta2, eps);
  }));
}

void adam_upd_with_perlr_cuda(
    torch::Tensor param,
    torch::Tensor grad,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    torch::Tensor perlr,
    const int64_t step, const double beta1, const double beta2, const double lr, const double eps) {

  const size_t N = param.numel();

  const int64_t threads = 256;
  const int64_t blocks = (N + threads - 1) / threads;

  const double step_size = lr * sqrt(1 - pow(beta2, (double)step)) / (1 - pow(beta1, (double)step));

  AT_DISPATCH_FLOATING_TYPES(param.scalar_type(), "adam_upd_with_perlr_cuda", ([&] {
    adam_upd_with_perlr_cuda_kernel<scalar_t><<<blocks, threads>>>(
        param.data_ptr<scalar_t>(),
        grad.data_ptr<scalar_t>(),
        exp_avg.data_ptr<scalar_t>(),
        exp_avg_sq.data_ptr<scalar_t>(),
        perlr.data_ptr<scalar_t>(),
        N, step_size, beta1, beta2, eps);
  }));
}

