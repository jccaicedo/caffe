// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void QLearningWithLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  qlearning_bottom_vec_.clear();
  qlearning_bottom_vec_.push_back(bottom[0]);
  qlearning_top_vec_.clear();
  qlearning_top_vec_.push_back(&prob_);
  qlearning_layer_->SetUp(qlearning_bottom_vec_, &qlearning_top_vec_);
}

template <typename Dtype>
void QLearningWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  qlearning_layer_->Reshape(qlearning_bottom_vec_, &qlearning_top_vec_);
  if (top->size() >= 2) {
    (*top)[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void QLearningWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the qlearning prob values.
  qlearning_bottom_vec_[0] = bottom[0];
  qlearning_layer_->Forward(qlearning_bottom_vec_, &qlearning_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype loss = 0;
  int num = prob_.num();
  int dim = prob_.count() / num;
  for (int i = 0; i < num; ++i) {
    int action = static_cast<int>(label[i*3 + 0]);
    Dtype reward = label[i*3 + 1];
    Dtype discountedMaxQ = label[i*3 + 2];
    for (int j = 0; j < dim; ++j) {
      if (action == j) {
        Dtype y = (reward + discountedMaxQ);
        loss += (y - prob_data[i * dim + action])*(y - prob_data[i * dim + action]);
      }
    }
  }
  (*top)[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void QLearningWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = (*bottom)[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    for (int i = 0; i < num; ++i) {
      int action = static_cast<int>(label[i*3 + 0]);
      Dtype reward = label[i*3 + 1];
      Dtype discountedMaxQ = label[i*3 + 2];
      for (int j = 0; j < dim; ++j) {
        if (action == j) {
          Dtype y = (reward + discountedMaxQ);
          bottom_diff[i * dim + action] -= y;
          //LOG(INFO) << i << " " << action << " " << reward << " " << discountedMaxQ << " " << prob_data[i * dim + action];
        } else {
          // Only update connections associated to the action
          bottom_diff[i * dim + j] = 0.0;
        }
      }
    }
    // Scale down gradient
    caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(QLearningWithLossLayer);
#endif

INSTANTIATE_CLASS(QLearningWithLossLayer);


}  // namespace caffe
