// Copyright 2013 Yangqing Jia

#ifndef CAFFE_Q_LAYER_HPP_
#define CAFFE_Q_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "leveldb/db.h"
#include "pthread.h"
#include "boost/scoped_ptr.hpp"
#include "hdf5.h"

#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class QLearningLayer : public Layer<Dtype> {
 public:
  explicit QLearningLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
     const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  // sum_multiplier is just used to carry out sum using blas
  Blob<Dtype> sum_multiplier_;
  // scale is an intermediate blob to hold temporary results.
  Blob<Dtype> scale_;
};


// QLearningWithLossLayer is a layer that implements QLearning and then computes
// the loss - it is preferred over QLearning + multinomiallogisticloss in the
// sense that during training, this will produce more numerically stable
// gradients. During testing this layer could be replaced by a QLearning layer
// to generate probability outputs.
template <typename Dtype>
class QLearningWithLossLayer : public Layer<Dtype> {
 public:
  explicit QLearningWithLossLayer(const LayerParameter& param)
      : Layer<Dtype>(param), qlearning_layer_(new QLearningLayer<Dtype>(param)) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);
  virtual Dtype Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom);
  virtual Dtype Backward_gpu(const vector<Blob<Dtype>*>& top,
     const bool propagate_down, vector<Blob<Dtype>*>* bottom);

  shared_ptr<QLearningLayer<Dtype> > qlearning_layer_;
  // prob stores the output probability of the layer.
  Blob<Dtype> prob_;
  // Vector holders to call the underlying qlearning layer forward and backward.
  vector<Blob<Dtype>*> qlearning_bottom_vec_;
  vector<Blob<Dtype>*> qlearning_top_vec_;
};



}  // namespace caffe

#endif  // CAFFE_Q_LAYER_HPP_
