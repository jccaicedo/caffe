// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/q_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>

using std::max;
using std::abs;

namespace caffe {

template <typename Dtype>
__global__ void kernel_max_div(const int num, const int dim,
    const Dtype* scale, Dtype* data) {
  CUDA_KERNEL_LOOP(index, num * dim) {
    int n = index / dim;
    data[index] /= scale[n];
  }
}

template <typename Dtype>
__global__ void kernel_get_absmax(const int num, const int dim, 
    const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num) {
    Dtype absMax = -FLT_MAX;
    for (int i = 0; i < dim; ++i) {
      absMax = max( abs(data[index * dim + i]), absMax);// * data[index * dim + i];
    }
    out[index] = absMax;
  }
}

template <typename Dtype>
void QLearningLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* scale_data = scale_.mutable_gpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  CUDA_CHECK(cudaMemcpy(top_data, bottom_data,
      sizeof(Dtype) * bottom[0]->count(), cudaMemcpyDeviceToDevice));
  // We need to normalize output vector to avoid numerical issues
  // Compute dot products
  kernel_get_absmax<Dtype><<<CAFFE_GET_BLOCKS(num), CAFFE_CUDA_NUM_THREADS>>>(
    num, dim, bottom_data, scale_data);
  // Do division
  kernel_max_div<Dtype><<<CAFFE_GET_BLOCKS(num * dim),
                              CAFFE_CUDA_NUM_THREADS>>>(
      num, dim, scale_data, top_data);
}

// TODO(Yangqing): implement the GPU version of softmax.
template <typename Dtype>
Dtype QLearningLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  int num = top[0]->num();
  int dim = top[0]->count() / top[0]->num();
  CUDA_CHECK(cudaMemcpy(bottom_diff, top_diff,
      sizeof(Dtype) * top[0]->count(), cudaMemcpyDeviceToDevice));
  // Compute inner1d(top_diff, top_data) and subtract them from the bottom diff
  // cuda dot returns the result to cpu, so we temporarily change the pointer
  // mode
  /*CUBLAS_CHECK(cublasSetPointerMode(Caffe::cublas_handle(),
      CUBLAS_POINTER_MODE_DEVICE));
  Dtype* scale_data = scale_.mutable_gpu_data();
  for (int i = 0; i < num; ++i) {
    caffe_gpu_dot<Dtype>(dim, top_diff + i * dim,
        top_data + i * dim, scale_data + i);
  }
  CUBLAS_CHECK(cublasSetPointerMode(Caffe::cublas_handle(),
      CUBLAS_POINTER_MODE_HOST));
  // subtraction
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, dim, 1, -1.,
      scale_.gpu_data(), sum_multiplier_.gpu_data(), 1., bottom_diff);*/
  // elementwise multiplication

  LOG(INFO) << "Backward GPU not implemented for QLearningLayer";
  caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
  return Dtype(0);
}

INSTANTIATE_CLASS(QLearningLayer);


}  // namespace caffe
