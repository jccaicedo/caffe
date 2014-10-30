// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/common_layers.hpp"
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
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
}

// TODO(Yangqing): implement the GPU version of softmax.
template <typename Dtype>
void QLearningLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  int num = top[0]->num();
  int dim = top[0]->count() / top[0]->num();
  CUDA_CHECK(cudaMemcpy(bottom_diff, top_diff,
      sizeof(Dtype) * top[0]->count(), cudaMemcpyDeviceToDevice));

  LOG(INFO) << "Backward GPU not implemented for QLearningLayer";
  //caffe_gpu_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

INSTANTIATE_CLASS(QLearningLayer);


}  // namespace caffe
