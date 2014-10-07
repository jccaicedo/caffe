// QDataLayer: based on window_data_layer.cpp by Ross Girshick
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

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

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
void* QDataLayerPrefetch(void* layer_pointer) {
  QDataLayer<Dtype>* layer =
      reinterpret_cast<QDataLayer<Dtype>*>(layer_pointer);

  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  Dtype* top_data = layer->prefetch_data_->mutable_cpu_data();
  Dtype* top_label = layer->prefetch_label_->mutable_cpu_data();
  Dtype* top_state_features = layer->prefetch_state_features_->mutable_cpu_data();
  const Dtype scale = layer->layer_param_.scale();
  const int batchsize = layer->layer_param_.batchsize();
  const int cropsize = layer->layer_param_.cropsize();
  const int context_pad = layer->layer_param_.det_context_pad();
  const bool mirror = layer->layer_param_.mirror();
  const float fg_fraction = layer->layer_param_.det_fg_fraction();
  const Dtype* mean = layer->data_mean_.cpu_data();
  const int mean_off = (layer->data_mean_.width() - cropsize) / 2;
  const int mean_width = layer->data_mean_.width();
  const int mean_height = layer->data_mean_.height();
  cv::Size cv_crop_size(cropsize, cropsize);
  const string& crop_mode = layer->layer_param_.det_crop_mode();

  bool use_square = (crop_mode == "square") ? true : false;
  const int stateFeatures = layer->layer_param_.num_state_features(); 
  const int numActions = layer->layer_param_.num_actions();
  int totalStateFeatures = stateFeatures + numActions;

  // zero out batch
  memset(top_data, 0, sizeof(Dtype)*layer->prefetch_data_->count());

  /*const int num_fg = static_cast<int>(static_cast<float>(batchsize)
      * fg_fraction);*/

  int itemid = 0;
  // sample from bg set then fg set
  for (int dummy = 0; dummy < batchsize; ++dummy) {
      // sample a window
      vector<float> window = layer->windows_[rand() % layer->windows_.size()];

      bool do_mirror = false;
      // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
      if (mirror && rand() % 2) {
        do_mirror = true;
      }

      // load the image containing the window
      pair<std::string, vector<int> > image =
          layer->image_database_[window[QDataLayer<Dtype>::IMAGE_INDEX]];

      cv::Mat cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
      if (!cv_img.data) {
        LOG(ERROR) << "Could not open or find file " << image.first;
        return reinterpret_cast<void*>(NULL);
      }
      const int channels = cv_img.channels();

      // crop window out of image and warp it
      int x1 = window[QDataLayer<Dtype>::X1];
      int y1 = window[QDataLayer<Dtype>::Y1];
      int x2 = window[QDataLayer<Dtype>::X2];
      int y2 = window[QDataLayer<Dtype>::Y2];

      int pad_w = 0;
      int pad_h = 0;
      if (context_pad > 0 || use_square) {
        // scale factor by which to expand the original region
        // such that after warping the expanded region to cropsize x cropsize
        // there's exactly context_pad amount of padding on each side
        Dtype context_scale = static_cast<Dtype>(cropsize) /
            static_cast<Dtype>(cropsize - 2*context_pad);

        // compute the expanded region
        Dtype half_height = static_cast<Dtype>(y2-y1+1)/2.0;
        Dtype half_width = static_cast<Dtype>(x2-x1+1)/2.0;
        Dtype center_x = static_cast<Dtype>(x1) + half_width;
        Dtype center_y = static_cast<Dtype>(y1) + half_height;
        if (use_square) {
          if (half_height > half_width) {
            half_width = half_height;
          } else {
            half_height = half_width;
          }
        }
        x1 = static_cast<int>(round(center_x - half_width*context_scale));
        x2 = static_cast<int>(round(center_x + half_width*context_scale));
        y1 = static_cast<int>(round(center_y - half_height*context_scale));
        y2 = static_cast<int>(round(center_y + half_height*context_scale));

        // the expanded region may go outside of the image
        // so we compute the clipped (expanded) region and keep track of
        // the extent beyond the image
        int unclipped_height = y2-y1+1;
        int unclipped_width = x2-x1+1;
        int pad_x1 = std::max(0, -x1);
        int pad_y1 = std::max(0, -y1);
        int pad_x2 = std::max(0, x2 - cv_img.cols + 1);
        int pad_y2 = std::max(0, y2 - cv_img.rows + 1);
        // clip bounds
        x1 = x1 + pad_x1;
        x2 = x2 - pad_x2;
        y1 = y1 + pad_y1;
        y2 = y2 - pad_y2;
        CHECK_GT(x1, -1);
        CHECK_GT(y1, -1);
        CHECK_LT(x2, cv_img.cols);
        CHECK_LT(y2, cv_img.rows);

        int clipped_height = y2-y1+1;
        int clipped_width = x2-x1+1;

        // scale factors that would be used to warp the unclipped
        // expanded region
        Dtype scale_x =
            static_cast<Dtype>(cropsize)/static_cast<Dtype>(unclipped_width);
        Dtype scale_y =
            static_cast<Dtype>(cropsize)/static_cast<Dtype>(unclipped_height);

        // size to warp the clipped expanded region to
        cv_crop_size.width =
            static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
        cv_crop_size.height =
            static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
        pad_x1 = static_cast<int>(round(static_cast<Dtype>(pad_x1)*scale_x));
        pad_x2 = static_cast<int>(round(static_cast<Dtype>(pad_x2)*scale_x));
        pad_y1 = static_cast<int>(round(static_cast<Dtype>(pad_y1)*scale_y));
        pad_y2 = static_cast<int>(round(static_cast<Dtype>(pad_y2)*scale_y));

        pad_h = pad_y1;
        // if we're mirroring, we mirror the padding too (to be pedantic)
        if (do_mirror) {
          pad_w = pad_x2;
        } else {
          pad_w = pad_x1;
        }

        // ensure that the warped, clipped region plus the padding
        // fits in the cropsize x cropsize image (it might not due to rounding)
        if (pad_h + cv_crop_size.height > cropsize) {
          cv_crop_size.height = cropsize - pad_h;
        }
        if (pad_w + cv_crop_size.width > cropsize) {
          cv_crop_size.width = cropsize - pad_w;
        }
      }

      cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
      cv::Mat cv_cropped_img = cv_img(roi);
      cv::resize(cv_cropped_img, cv_cropped_img,
          cv_crop_size, 0, 0, cv::INTER_LINEAR);

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

            top_data[((itemid * channels + c) * cropsize + h + pad_h)
                     * cropsize + w + pad_w]
                = (pixel
                    - mean[(c * mean_height + h + mean_off + pad_h)
                           * mean_width + w + mean_off + pad_w])
                  * scale;
          }
        }
      }

      // get window label
      top_label[itemid*3 + 0] = window[QDataLayer<Dtype>::ACTION];
      top_label[itemid*3 + 1] = window[QDataLayer<Dtype>::REWARD];
      top_label[itemid*3 + 2] = window[QDataLayer<Dtype>::DISCOUNTEDMAXQ];

      // get state features
      for (int q = 0; q < stateFeatures; ++q) {
        top_state_features[itemid*totalStateFeatures + q] = window[QDataLayer<Dtype>::NUM + q];
      }
      for (int q = 0; q < numActions; ++q) {
        if (q == window[QDataLayer<Dtype>::PREV_ACTION]) {
          top_state_features[itemid*totalStateFeatures + stateFeatures + q] = 1.0;
        } else {
          top_state_features[itemid*totalStateFeatures + stateFeatures + q] = 0.0;
        }
      }

      #if 0
      // useful debugging code for dumping transformed windows to disk
      string file_id;
      std::stringstream ss;
      // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
      ss << rand();
      ss >> file_id;
      std::ofstream inf((string("dump/") + file_id +
          string("_info.txt")).c_str(), std::ofstream::out);
      inf << image.first << std::endl
          << window[QDataLayer<Dtype>::X1]+1 << std::endl
          << window[QDataLayer<Dtype>::Y1]+1 << std::endl
          << window[QDataLayer<Dtype>::X2]+1 << std::endl
          << window[QDataLayer<Dtype>::Y2]+1 << std::endl
          << do_mirror << std::endl
          << top_label[itemid][0] << std::endl
          << is_fg << std::endl;
      inf.close();
      std::ofstream top_data_file((string("dump/") + file_id +
          string("_data.txt")).c_str(),
          std::ofstream::out | std::ofstream::binary);
      for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < cropsize; ++h) {
          for (int w = 0; w < cropsize; ++w) {
            top_data_file.write(reinterpret_cast<char*>(
                &top_data[((itemid * channels + c) * cropsize + h)
                          * cropsize + w]),
                sizeof(Dtype));
          }
        }
      }
      top_data_file.close();
      #endif

      itemid++;
  }

  return reinterpret_cast<void*>(NULL);
}

template <typename Dtype>
QDataLayer<Dtype>::~QDataLayer<Dtype>() {
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
}

template <typename Dtype>
void QDataLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // SetUp runs through the window_file and creates two structures
  // that hold windows: one for foreground (object) windows and one
  // for background (non-object) windows. We use an overlap threshold
  // to decide which is which.

  CHECK_EQ(bottom.size(), 0) << "Q data Layer takes no input blobs.";
  CHECK_EQ(top->size(), 3) << "Q data Layer prodcues three blobs as output.";

  // window_file format
  // repeated:
  //    # image_index
  //    img_path (abs path)
  //    channels
  //    height
  //    width
  //    num_windows
  //    class_index action reward maxNextQ x1 y1 x2 y2 stateFeatures ... prevAction

  LOG(INFO) << "Q data layer:" << std::endl
      << "  Number of actions: "
      << this->layer_param_.num_actions() << std::endl
      << "  Number of state features: "
      << this->layer_param_.num_state_features();

  std::ifstream infile(this->layer_param_.source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.source() << std::endl;

  map<int, int> label_hist;
  label_hist.insert(std::make_pair(0, 0));

  string hashtag;
  int image_index, channels;
  while (infile >> hashtag >> image_index) {
    CHECK_EQ(hashtag, "#");
    // read image path
    string image_path;
    infile >> image_path;
    // read image dimensions
    vector<int> image_size(3);
    infile >> image_size[0] >> image_size[1] >> image_size[2];
    channels = image_size[0];
    image_database_.push_back(std::make_pair(image_path, image_size));

    // read each box
    int num_windows;
    infile >> num_windows;
    for (int i = 0; i < num_windows; ++i) {
      vector<float> window(QDataLayer::NUM + this->layer_param_.num_state_features());
      int action, x1, y1, x2, y2, prevAction;
      float reward, discountedMaxQ, feature;
      infile >> action >> reward >> discountedMaxQ >> x1 >> y1 >> x2 >> y2;
      for (int j = 0; j < this->layer_param_.num_state_features(); ++j) {
        infile >> feature;
        window[QDataLayer::NUM + j] = feature;
      }
      infile >> prevAction;

      window[QDataLayer::IMAGE_INDEX] = image_index;
      window[QDataLayer::ACTION] = action;
      window[QDataLayer::REWARD] = reward;
      window[QDataLayer::DISCOUNTEDMAXQ] = discountedMaxQ;
      window[QDataLayer::X1] = x1;
      window[QDataLayer::Y1] = y1;
      window[QDataLayer::X2] = x2;
      window[QDataLayer::Y2] = y2;
      window[QDataLayer::PREV_ACTION] = prevAction;

      // add window to list
      int label = window[QDataLayer::ACTION];
      CHECK_GT(label, -1);
      windows_.push_back(window);
      label_hist.insert(std::make_pair(label, 0));
      label_hist[label]++;
    }

    if (image_index % 100 == 0) {
      LOG(INFO) << "num: " << image_index << " "
          << image_path << " "
          << image_size[0] << " "
          << image_size[1] << " "
          << image_size[2] << " "
          << "windows to process: " << num_windows;
    }
  }

  LOG(INFO) << "Number of images: " << image_index+1;

  for (map<int, int>::iterator it = label_hist.begin();
      it != label_hist.end(); ++it) {
    LOG(INFO) << "class " << it->first << " has " << label_hist[it->first]
              << " samples";
  }

  LOG(INFO) << "Amount of context padding: "
      << this->layer_param_.det_context_pad();

  LOG(INFO) << "Crop mode: " << this->layer_param_.det_crop_mode();

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
  // Structured label: Action, Reward, NextMaxQ
  (*top)[1]->Reshape(this->layer_param_.batchsize(), 3, 1, 1);
  prefetch_label_.reset(
      new Blob<Dtype>(this->layer_param_.batchsize(), 3, 1, 1));
  // State features
  int totalFeatures = this->layer_param_.num_state_features() + this->layer_param_.num_actions();
  (*top)[2]->Reshape(this->layer_param_.batchsize(), totalFeatures, 1, 1);
  prefetch_state_features_.reset(
      new Blob<Dtype>(this->layer_param_.batchsize(), totalFeatures, 1, 1)
  );

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
  CHECK(!pthread_create(&thread_, NULL, QDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
  DLOG(INFO) << "Prefetch initialized.";
}

template <typename Dtype>
void QDataLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // First, join the thread
  CHECK(!pthread_join(thread_, NULL)) << "Pthread joining failed.";
  // Copy the data
  memcpy((*top)[0]->mutable_cpu_data(), prefetch_data_->cpu_data(),
      sizeof(Dtype) * prefetch_data_->count());
  memcpy((*top)[1]->mutable_cpu_data(), prefetch_label_->cpu_data(),
      sizeof(Dtype) * prefetch_label_->count());
  memcpy((*top)[2]->mutable_cpu_data(), prefetch_state_features_->cpu_data(),
      sizeof(Dtype) * prefetch_state_features_->count());

  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, QDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

template <typename Dtype>
void QDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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
  CUDA_CHECK(cudaMemcpy((*top)[2]->mutable_gpu_data(),
      prefetch_label_->cpu_data(), sizeof(Dtype) * prefetch_state_features_->count(),
      cudaMemcpyHostToDevice));

  // Start a new prefetch thread
  CHECK(!pthread_create(&thread_, NULL, QDataLayerPrefetch<Dtype>,
      reinterpret_cast<void*>(this))) << "Pthread execution failed.";
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype QDataLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

template <typename Dtype>
Dtype QDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(QDataLayer);

}  // namespace caffe
