// Copyright 2013 Yangqing Jia
//
#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
//#include "caffe/vision_layers.hpp"
#include "caffe/q_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <cmath>

using std::max;
using std::abs;

namespace caffe {

template <typename Dtype>
void QLearningLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "QLearning Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "QLearning Layer takes a single blob as output.";
  (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  sum_multiplier_.Reshape(1, bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  for (int i = 0; i < sum_multiplier_.count(); ++i) {
    multiplier_data[i] = 1.;
  }
  scale_.Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void QLearningLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  memcpy(top_data, bottom_data, sizeof(Dtype) * bottom[0]->count());

  /*#include <sstream> 
  for (int i = 0; i < num; ++i) {
    std::ostringstream s;
    for (int j = 0; j < dim; ++j) {
      s << bottom_data[i * dim + j] << " ";
    }
    LOG(INFO) << s.str();
  }*/

  // we need to divide by the absmax to avoid numerical issues
  /*for (int i = 0; i < num; ++i) {
    scale_data[i] = abs(bottom_data[i*dim]);
    for (int j = 0; j < dim; ++j) {
      scale_data[i] = max(scale_data[i], abs(bottom_data[i * dim + j]));
    }
  }
  // Do division
  for (int i = 0; i < num; ++i) {
    caffe_scal<Dtype>(dim, Dtype(1.) / scale_data[i], top_data + i * dim);
  }*/
}

template <typename Dtype>
Dtype QLearningLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int num = top[0]->num();
  int dim = top[0]->count() / top[0]->num();
  memcpy(bottom_diff, top_diff, sizeof(Dtype) * top[0]->count());
  // elementwise multiplication
  //caffe_mul<Dtype>(top[0]->count(), bottom_diff, top_data, bottom_diff);
  LOG(INFO) << "Aparently unused QLearningLayer Backward_cpu (not implemented)";
  return Dtype(0);
}


INSTANTIATE_CLASS(QLearningLayer);


}  // namespace caffe
