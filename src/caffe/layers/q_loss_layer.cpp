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
void QLearningWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  //CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  //CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  //CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void QLearningWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const Dtype* qvalues = bottom[0]->cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype loss = 0;
  int num = diff_.num();
  int dim = diff_.count() / num;
  for (int i = 0; i < num; ++i) {
    int action = static_cast<int>(label[i*3 + 0]);
    Dtype reward = label[i*3 + 1];
    Dtype discountedMaxQ = label[i*3 + 2];
    for (int j = 0; j < dim; ++j) {
      if (action == j) {
        Dtype y = (reward + discountedMaxQ);
        loss += (y - qvalues[i * dim + action])*(y - qvalues[i * dim + action]);
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
    const Dtype* qvalues = (*bottom)[0]->cpu_data();
    const Dtype* label = (*bottom)[1]->cpu_data();
    int num = diff_.num();
    int dim = diff_.count() / num;
    for (int i = 0; i < num; ++i) {
      int action = static_cast<int>(label[i*3 + 0]);
      Dtype reward = label[i*3 + 1];
      Dtype discountedMaxQ = label[i*3 + 2];
      for (int j = 0; j < dim; ++j) {
        if (action == j) {
          Dtype y = (reward + discountedMaxQ);
          bottom_diff[i * dim + action] = qvalues[i * dim + action] - y;
          LOG(INFO) << i << " [" << action << "; " << reward << "; " << discountedMaxQ << "] " << qvalues[i * dim + action] << " # " << bottom_diff[i * dim + action];
        } else {
          // Only update connections associated to the action
          bottom_diff[i * dim + j] = Dtype(0);
        }
      }
    }
    // Scale down gradient
    caffe_scal(diff_.count(), Dtype(1) / num, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(QLearningWithLossLayer);
#endif

INSTANTIATE_CLASS(QLearningWithLossLayer);


}  // namespace caffe
