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
  qlearning_bottom_vec_.clear();
  qlearning_bottom_vec_.push_back(bottom[0]);
  qlearning_top_vec_.push_back(&prob_);
  qlearning_layer_->SetUp(qlearning_bottom_vec_, &qlearning_top_vec_);
}

template <typename Dtype>
void QLearningWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the qlearning prob values.
  qlearning_bottom_vec_[0] = bottom[0];
  qlearning_layer_->Forward(qlearning_bottom_vec_, &qlearning_top_vec_);
  LOG(INFO) << "QLearningWithLoss Forward CPU";
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
  //Dtype avg[10] = {};
  //Dtype dif[10] = {};
  for (int i = 0; i < num; ++i) {
    int action = static_cast<int>(label[i*3 + 0]);
    Dtype reward = label[i*3 + 1];
    //if (reward > 0) reward *= 2;
    //else reward *= 3;
    Dtype discountedMaxQ = label[i*3 + 2];
    for (int j = 0; j < dim; ++j) {
      //avg[j] += prob_data[i*dim+j];
      if (action == j) {
        Dtype y = (reward + discountedMaxQ);
        bottom_diff[i * dim + action] -= y;
        //dif[j] += bottom_diff[i*dim + action];
        loss += (y - prob_data[i * dim + action])*(y - prob_data[i * dim + action]);
        //LOG(INFO) << i << " " << action << " " << reward << " " << discountedMaxQ << " " << prob_data[i * dim + action] << " " << loss;
      } else {
        // Only update connections associated to the action
        bottom_diff[i * dim + j] = 0.0;
      }
    }
  }
  //for (int j = 0; j < dim; ++j) {avg[j] /= num; dif[j] /= num; }
  //LOG(INFO) << "Prob Data: " << avg[0] << " " << avg[1] << " " << avg[2] << " " << avg[3] << " " << avg[4] << " " << avg[5] << " " << avg[6] << " " << avg[7] << " " << avg[8] << " " << avg[9];
  //LOG(INFO) << "Diff Data: " << dif[0] << " " << dif[1] << " " << dif[2] << " " << dif[3] << " " << dif[4] << " " << dif[5] << " " << dif[6] << " " << dif[7] << " " << dif[8] << " " << dif[9];

  // Scale down gradient
  caffe_scal(prob_.count(), Dtype(1) / num, bottom_diff);
  //LOG(INFO) << " Loss " << loss/num;
  return loss / num;
}


INSTANTIATE_CLASS(QLearningWithLossLayer);


}  // namespace caffe
