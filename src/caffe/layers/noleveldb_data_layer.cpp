// Copyright 2014 Juan C. Caicedo 
//
// Based on data_layer.cpp by Ross Girshick.

#include <stdint.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <map>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
NoLevelDBDataLayer<Dtype>::~NoLevelDBDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void NoLevelDBDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // SetUp runs through the window_file and creates two structures 
  // that hold windows: one for foreground (object) windows and one 
  // for background (non-object) windows. We use an overlap threshold 
  // to decide which is which.

  CHECK_EQ(bottom.size(), 0) << "NoLevelDB data Layer takes no input blobs.";
  CHECK_EQ(top->size(), 2) << "NoLevelDB data Layer prodcues two blobs as output.";

  // window_file format
  // repeated:
  //    class_index img_path (abs path)

  int channels = this->layer_param_.noleveldb_param().img_channels();

  std::ifstream infile(this->layer_param_.noleveldb_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file " 
      << this->layer_param_.noleveldb_param().source() << std::endl;

  int label, image_index = 0;
  string image_path;
  while (infile >> label >> image_path) {
    image_database_.push_back(std::make_pair(image_path, label));
    image_index += 1;
  }

  LOG(INFO) << "Number of images: " << image_index+1;

  // image
  const int cropsize = this->layer_param_.noleveldb_param().crop_size();
  const int batch_size = this->layer_param_.noleveldb_param().batch_size();
  CHECK_GT(cropsize, 0);
  (*top)[0]->Reshape(batch_size, channels, cropsize, cropsize);
  this->prefetch_data_.Reshape(batch_size, channels, cropsize, cropsize);

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(batch_size, 1, 1, 1);
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1);
}

template <typename Dtype>
unsigned int NoLevelDBDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// Thread fetching the data
template <typename Dtype>
void NoLevelDBDataLayer<Dtype>::InternalThreadEntry() {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  const Dtype scale = this->layer_param_.noleveldb_param().scale();
  const int batchsize = this->layer_param_.noleveldb_param().batch_size();
  const int cropsize = this->layer_param_.noleveldb_param().crop_size();
  const bool mirror = this->layer_param_.noleveldb_param().mirror();
  const Dtype* mean = this->data_mean_.cpu_data();
  const int mean_width = this->data_mean_.width();
  const int mean_height = this->data_mean_.height();
  cv::Size cv_crop_size(cropsize, cropsize);

  // zero out batch
  caffe_set(this->prefetch_data_.count(), Dtype(0), top_data);

  for (int itemid = 0; itemid < batchsize; ++itemid) {

      bool do_mirror = false;
      if (mirror && rand() % 2) {
        do_mirror = true;
      }

      // load the image containing the window
      pair<std::string, int > image = 
              this->image_database_[rand() % this->image_database_.size()];

      cv::Mat cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
      if (!cv_img.data) {
        LOG(ERROR) << "Could not open or find file " << image.first;
        return;
      }
      const int channels = cv_img.channels();
      //LOG(INFO) << "Image " << image.first << " is open (rows:" << cv_img.rows << ",cols:" << cv_img.cols << ")";


      int h_off, w_off;
      // We only do random crop when we do training.
      if (Caffe::phase() == Caffe::TRAIN) {
        h_off = rand() % (cv_img.rows - cropsize);
        w_off = rand() % (cv_img.cols - cropsize);
      } else {
        h_off = (cv_img.rows - cropsize) / 2;
        w_off = (cv_img.cols - cropsize) / 2;
      }

      //LOG(INFO) << "Ready to crop: Label=" << image.second << " h_off=" << h_off << " w_off=" << w_off;

      // Crop image
      cv::Rect roi(w_off, h_off, cropsize, cropsize);
      cv::Mat cv_cropped_img = cv_img(roi);

      //LOG(INFO) << "Image cropped:" << cv_cropped_img.rows << " , " << cv_cropped_img.cols ;
      
      // horizontal flip at random
      if (do_mirror) {
        cv::flip(cv_cropped_img, cv_cropped_img, 1);
      }

      // copy the warped window into top_data
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < cv_cropped_img.rows; ++h) {
          for (int w = 0; w < cv_cropped_img.cols; ++w) {
            Dtype pixel = 
                static_cast<Dtype>(cv_cropped_img.at<cv::Vec3b>(h, w)[c]);

            top_data[((itemid * channels + c) * cropsize + h) * cropsize + w]
                = (pixel
                    - mean[(c * mean_height + h + h_off) 
                           * mean_width + w + w_off])
                  * scale;
          }
        }
      }
      top_label[itemid] = image.second;
  }
}

INSTANTIATE_CLASS(NoLevelDBDataLayer);

}  // namespace caffe
