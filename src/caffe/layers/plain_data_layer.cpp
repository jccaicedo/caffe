// PlainDataLayer: based on window_data_layer.cpp by Ross Girshick
//
// Based on data_layer.cpp by Yangqing Jia.

#include <stdint.h>
#include <pthread.h>

#include <algorithm>
#include <string>
#include <vector>
#include <map>
#include <fstream>  // NOLINT(readability/streams)
#include <utility>


#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

using std::string;
using std::map;
using std::pair;

// caffe.proto > LayerParameter
//   'source' field specifies the window_file
//   'cropsize' indicates the desired warped size

namespace caffe {


template <typename Dtype>
PlainDataLayer<Dtype>::~PlainDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void PlainDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // SetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  CHECK_EQ(bottom.size(), 0) << "Plain data Layer takes no input blobs.";
  CHECK_EQ(top->size(), 2) << "Plain data Layer prodcues two blobs as output.";

  // plain_file format
  //    action reward maxNextQ stateFeatures

  LOG(INFO) << "Plain data layer:" << std::endl
      << "  Number of state features: "
      << this->layer_param_.plaindata_param().num_state_features();

  std::ifstream infile(this->layer_param_.plaindata_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.plaindata_param().source() << std::endl;

  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));

  int action, examples = 0;
  while (infile >> action) {
    // read each box
    vector<float> window(PlainDataLayer::NUM + this->layer_param_.plaindata_param().num_state_features());
    float reward, discountedMaxQ, feature;
    infile >> reward >> discountedMaxQ;
    for (int j = 0; j < this->layer_param_.plaindata_param().num_state_features(); ++j) {
      infile >> feature;
      window[PlainDataLayer::NUM + j] = feature;
    }

    window[PlainDataLayer::ACTION] = action;
    window[PlainDataLayer::REWARD] = reward;
    window[PlainDataLayer::DISCOUNTEDMAXQ] = discountedMaxQ;

    // add window to list
    int label = window[PlainDataLayer::ACTION];
    CHECK_GT(label, -1);
    samples_.push_back(window);
    label_hist.insert(std::make_pair(label, 0));
    label_hist[label]++;
    examples++;
  }

  LOG(INFO) << "Number of examples: " << examples+1;

  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }

  (*top)[0]->Reshape(
      this->layer_param_.plaindata_param().batch_size(), 
      this->layer_param_.plaindata_param().num_state_features(), 1, 1);
  this->prefetch_data_.Reshape(
      this->layer_param_.plaindata_param().batch_size(),
      this->layer_param_.plaindata_param().num_state_features(), 1, 1);

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // Structured label: Action, Reward, NextMaxQ
  (*top)[1]->Reshape(this->layer_param_.plaindata_param().batch_size(), 3, 1, 1);
  this->prefetch_label_.Reshape(this->layer_param_.plaindata_param().batch_size(), 3, 1, 1);

  //data_mean_.Reshape(1, 1, 1, 1);
}

template <typename Dtype>
unsigned int PlainDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// Thread fetching the data
template <typename Dtype>
void PlainDataLayer<Dtype>::InternalThreadEntry() {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  const int batchsize = this->layer_param_.plaindata_param().batch_size();
  const int stateFeatures = this->layer_param_.plaindata_param().num_state_features(); 

  // zero out batch
  caffe_set(this->prefetch_data_.count(), Dtype(0), top_data);

  int itemid = 0;
  // sample from bg set then fg set
  for (int dummy = 0; dummy < batchsize; ++dummy) {
      // sample a window
      vector<float> window = this->samples_[rand() % this->samples_.size()];

      // get window label
      top_label[itemid*3 + 0] = window[PlainDataLayer<Dtype>::ACTION];
      top_label[itemid*3 + 1] = window[PlainDataLayer<Dtype>::REWARD];
      top_label[itemid*3 + 2] = window[PlainDataLayer<Dtype>::DISCOUNTEDMAXQ];

      // get state features
      for (int q = 0; q < stateFeatures; ++q) {
        top_data[itemid*stateFeatures + q] = window[PlainDataLayer<Dtype>::NUM + q];
      }

      /*#include <sstream>
      std::ostringstream s; 
      for (int q = 0; q < totalStateFeatures; ++q) {
        s << " " << top_state_features[itemid*totalStateFeatures + q];
      }
      LOG(INFO) << s.str();*/

      itemid++;
  }
}

INSTANTIATE_CLASS(PlainDataLayer);

}  // namespace caffe
