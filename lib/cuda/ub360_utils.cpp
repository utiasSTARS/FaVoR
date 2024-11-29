#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

torch::Tensor cumdist_thres_cuda(torch::Tensor dist, double thres);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor cumdist_thres(torch::Tensor dist, double thres) {
  CHECK_INPUT(dist);
  return cumdist_thres_cuda(dist, thres);
}

TORCH_LIBRARY(TORCH_EXTENSION_NAME, m) {
  m.def("cumdist_thres", &cumdist_thres); //"Generate mask for cumulative dist."
}

