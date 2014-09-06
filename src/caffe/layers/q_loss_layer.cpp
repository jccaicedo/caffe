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
void QLearningWithLossLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "QLearningLoss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 0) << "QLearningLoss Layer takes no blob as output.";
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
void QLearningWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_bottom_vec_[0] = bottom[0];
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
Dtype QLearningWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  // First, compute the diff
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  const Dtype* prob_data = prob_.cpu_data();
  memcpy(bottom_diff, prob_data, sizeof(Dtype) * prob_.count());
  const Dtype* label = (*bottom)[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    int action = static_cast<int>(label[i*3 + 0]);
    Dtype reward = label[i*3 + 1];
    Dtype discountedMaxQ = label[i*3 + 2];
    for (int j = 0; j < dim; ++j) {
      if (action == j) {
        Dtype y = (reward + discountedMaxQ);
        bottom_diff[i * dim + action] -= y;
        loss += (y - prob_data[i * dim + action])*(y - prob_data[i * dim + action]);
      } else {
        // Only update connections associated to the action
        bottom_diff[i * dim + action] = 0.0;
      }
    }
  }
  // Scale down gradient
  caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
  return loss / num;
}


INSTANTIATE_CLASS(QLearningWithLossLayer);


}  // namespace caffe
