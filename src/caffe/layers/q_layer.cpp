// Copyright 2013 Yangqing Jia
//
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"
#include "caffe/loss_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>

using std::max;
using std::abs;

namespace caffe {

template <typename Dtype>
void QLearningLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "QLearning Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "QLearning Layer takes a single blob as output.";
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void QLearningLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
}

template <typename Dtype>
void QLearningLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  LOG(INFO) << "Aparently unused QLearningLayer Backward_cpu (not implemented)";
}


INSTANTIATE_CLASS(QLearningLayer);


}  // namespace caffe
