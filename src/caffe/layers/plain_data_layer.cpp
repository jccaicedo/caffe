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

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using std::string;
using std::map;
using std::pair;

// caffe.proto > LayerParameter
//   'source' field specifies the window_file
//   'cropsize' indicates the desired warped size

namespace caffe {

template <typename Dtype>
void* PlainDataLayerPrefetch(void* layer_pointer) {
  PlainDataLayer<Dtype>* layer =
      reinterpret_cast<PlainDataLayer<Dtype>*>(layer_pointer);

  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  const Dtype scale = layer->layer_param_.scale();
  const int batchsize = layer->layer_param_.batchsize();
  //const Dtype* mean = layer->data_mean_.cpu_data();
  const int stateFeatures = layer->layer_param_.num_state_features(); 

  // zero out batch
  memset(top_data, 0, sizeof(Dtype)*layer->prefetch_data_->count());

  int itemid = 0;
  // sample from bg set then fg set
  for (int dummy = 0; dummy < batchsize; ++dummy) {
      // sample a window
      vector<float> window = layer->samples_[rand() % layer->samples_.size()];

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

  return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
PlainDataLayer<Dtype>::~PlainDataLayer<Dtype>() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
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
      << this->layer_param_.num_state_features();

  std::ifstream infile(this->layer_param_.source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.source() << std::endl;

  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));

  int action, examples = 0;
  while (infile >> action) {
    // read each box
    vector<float> window(PlainDataLayer::NUM + this->layer_param_.num_state_features());
    float reward, discountedMaxQ, feature;
    infile >> reward >> discountedMaxQ;
    for (int j = 0; j < this->layer_param_.num_state_features(); ++j) {
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
      this->layer_param_.batchsize(), this->layer_param_.num_state_features(), 1, 1);
  prefetch_data_.reset(new Blob<Dtype>(
      this->layer_param_.batchsize(), this->layer_param_.num_state_features(), 1, 1));

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // Structured label: Action, Reward, NextMaxQ
  (*top)[1]->Reshape(this->layer_param_.batchsize(), 3, 1, 1);
  prefetch_label_.reset(
      new Blob<Dtype>(this->layer_param_.batchsize(), 3, 1, 1));

  // check if we want to have mean
  /*if (this->layer_param_.has_meanfile()) {
    BlobProto blob_proto;
    LOG(INFO) << "Loading mean file from" << this->layer_param_.meanfile();
    ReadProtoFromBinaryFile(this->layer_param_.meanfile().c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.width(), data_mean_.height());
    CHECK_EQ(data_mean_.channels(), channels);
  } else {*/
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, 1, 1, 1);
  //}
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  prefetch_label_->mutable_cpu_data();
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CHECK(!pthread_create(&thread_, NULL, PlainDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void PlainDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // Copy the data
  memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_->cpu_data(),
      sizeof(Dtype) * prefetch_data_->count());
  memcpy((*top)[1]->mutable_cpu_data(), prefetch_label_->cpu_data(),
      sizeof(Dtype) * prefetch_label_->count());

  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, PlainDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void PlainDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // Copy the data
  CUDA_CHECK(cudaMemcpy((*top)[0]->mutable_gpu_data(),
      prefetch_data_->cpu_data(), sizeof(Dtype) * prefetch_data_->count(),
      cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy((*top)[1]->mutable_gpu_data(),
      prefetch_label_->cpu_data(), sizeof(Dtype) * prefetch_label_->count(),
      cudaMemcpyHostToDevice));

  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, PlainDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype PlainDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

template <typename Dtype>
Dtype PlainDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(PlainDataLayer);

}  // namespace caffe
