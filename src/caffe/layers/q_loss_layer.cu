// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/q_layer.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void QLearningWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the qlearning prob values.
  qlearning_bottom_vec_[0] = bottom[0];
  qlearning_layer_->Forward(qlearning_bottom_vec_, &qlearning_top_vec_);
}

template <typename Dtype>
Dtype QLearningWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  // TODO(Yangqing): implement the GPU version of qlearning.
  return Backward_cpu(top, propagate_down, bottom);
}

INSTANTIATE_CLASS(QLearningWithLossLayer);


}  // namespace caffe
