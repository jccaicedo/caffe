// Copyright 2014 Juan C. Caicedo 
//
// Based on data_layer.cpp by Ross Girshick.

#include <stdint.h>
#include <pthread.h>

#include <string>
#include <vector>
#include <map>
#include <fstream>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using std::string;
using std::map;
using std::pair;

// caffe.proto > LayerParameter
//   'source' field specifies the window_file
//   'cropsize' indicates the desired warped size

namespace caffe {

template <typename Dtype>
void* NoLevelDBDataLayerPrefetch(void* layer_pointer) {
  NoLevelDBDataLayer<Dtype>* layer = 
      reinterpret_cast<NoLevelDBDataLayer<Dtype>*>(layer_pointer);

  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  const Dtype scale = layer->layer_param_.scale();
  const int batchsize = layer->layer_param_.batchsize();
  const int cropsize = layer->layer_param_.cropsize();
  const bool mirror = layer->layer_param_.mirror();
  const Dtype* mean = layer->data_mean_.cpu_data();
  const int mean_width = layer->data_mean_.width();
  const int mean_height = layer->data_mean_.height();
  cv::Size cv_crop_size(cropsize, cropsize);
  const string& crop_mode = layer->layer_param_.det_crop_mode();

  bool use_square = (crop_mode == "square") ? true : false;

  // zero out batch
  memset(top_data, 0, sizeof(Dtype)*layer->prefetch_data_->count());

  for (int itemid = 0; itemid < batchsize; ++itemid) {

      bool do_mirror = false;
      if (mirror && rand() % 2) {
        do_mirror = true;
      }

      // load the image containing the window
      pair<std::string, int > image = 
          layer->image_database_[rand() % layer->image_database_.size()];

      cv::Mat cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
      if (!cv_img.data) {
        LOG(ERROR) << "Could not open or find file " << image.first;
        return (void*)NULL;
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

      // get window label
      //top_label[itemid] = window[NoLevelDBDataLayer<Dtype>::LABEL];
      top_label[itemid] = image.second;

      #if 0
      // useful debugging code for dumping transformed windows to disk
      string file_id;
      std::stringstream ss;
      ss << rand();
      ss >> file_id;
      std::ofstream inf((string("dump/") + file_id + 
          string("_info.txt")).c_str(), std::ofstream::out);
      inf << image.first << std::endl 
          << window[NoLevelDBDataLayer<Dtype>::X1]+1 << std::endl
          << window[NoLevelDBDataLayer<Dtype>::Y1]+1 << std::endl
          << window[NoLevelDBDataLayer<Dtype>::X2]+1 << std::endl
          << window[NoLevelDBDataLayer<Dtype>::Y2]+1 << std::endl
          << do_mirror << std::endl
          << top_label[itemid] << std::endl
          << is_fg << std::endl;
      inf.close();
      std::ofstream top_data_file((string("dump/") + file_id + 
          string("_data.txt")).c_str(), 
          std::ofstream::out | std::ofstream::binary);
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < cropsize; ++h) {
          for (int w = 0; w < cropsize; ++w) {
            top_data_file.write(
                reinterpret_cast<char*>(&top_data[((itemid * channels + c) 
                                                   * cropsize + h) * cropsize + w]),
                sizeof(Dtype));
          }
        }
      }
      top_data_file.close();
      #endif

      //itemid++;
    //}
  }

  return (void*)NULL;
}

template <typename Dtype>
NoLevelDBDataLayer<Dtype>::~NoLevelDBDataLayer<Dtype>() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
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

  int channels = this->layer_param_.img_channels();

  std::ifstream infile(this->layer_param_.source().c_str());
  CHECK(infile.good()) << "Failed to open window file " 
      << this->layer_param_.source() << std::endl;

  int label, image_index = 0;
  string image_path;
  while (infile >> label >> image_path) {
    image_database_.push_back(std::make_pair(image_path, label));
    image_index += 1;
  }

  LOG(INFO) << "Number of images: " << image_index+1;

  // image
  int cropsize = this->layer_param_.cropsize();
  CHECK_GT(cropsize, 0);
  (*top)[0]->Reshape(
      this->layer_param_.batchsize(), channels, cropsize, cropsize);
  prefetch_data_.reset(new Blob<Dtype>(
      this->layer_param_.batchsize(), channels, cropsize, cropsize));

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // label
  (*top)[1]->Reshape(this->layer_param_.batchsize(), 1, 1, 1);
  prefetch_label_.reset(
      new Blob<Dtype>(this->layer_param_.batchsize(), 1, 1, 1));

  // check if we want to have mean
  if (this->layer_param_.has_meanfile()) {
    BlobProto blob_proto;
    LOG(INFO) << "Loading mean file from" << this->layer_param_.meanfile();
    ReadProtoFromBinaryFile(this->layer_param_.meanfile().c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
    CHECK_EQ(data_mean_.num(), 1);
    CHECK_EQ(data_mean_.width(), data_mean_.height());
    CHECK_EQ(data_mean_.channels(), channels);
  } else {
    // Simply initialize an all-empty mean.
    data_mean_.Reshape(1, channels, cropsize, cropsize);
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  prefetch_data_->mutable_cpu_data();
  prefetch_label_->mutable_cpu_data();
  data_mean_.cpu_data();
  DLOG(INFO) << "Initializing prefetch";
  CHECK(!pthread_create(&thread_, NULL, NoLevelDBDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void NoLevelDBDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // Copy the data
  memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_->cpu_data(),
      sizeof(Dtype) * prefetch_data_->count());
  memcpy((*top)[1]->mutable_cpu_data(), prefetch_label_->cpu_data(),
      sizeof(Dtype) * prefetch_label_->count());
  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, NoLevelDBDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void NoLevelDBDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
  CHECK(!pthread_create(&thread_, NULL, NoLevelDBDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype NoLevelDBDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

template <typename Dtype>
Dtype NoLevelDBDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(NoLevelDBDataLayer);

}  // namespace caffe
