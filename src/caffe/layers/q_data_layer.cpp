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
//   'crop_size' indicates the desired warped size

namespace caffe {

template <typename Dtype>
QDataLayer<Dtype>::~QDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void QDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
      << this->layer_param_.qdata_param().num_actions() << std::endl
      << "  Number of state features: "
      << this->layer_param_.qdata_param().num_state_features();

  std::ifstream infile(this->layer_param_.qdata_param().source().c_str());
  CHECK(infile.good()) << "Failed to open window file "
      << this->layer_param_.qdata_param().source() << std::endl;

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
      vector<float> window(QDataLayer::NUM + this->layer_param_.qdata_param().num_state_features());
      int action, x1, y1, x2, y2, prevAction;
      float reward, discountedMaxQ, feature;
      infile >> action >> reward >> discountedMaxQ >> x1 >> y1 >> x2 >> y2;
      for (int j = 0; j < this->layer_param_.qdata_param().num_state_features(); ++j) {
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
      << this->layer_param_.qdata_param().context_pad();

  LOG(INFO) << "Crop mode: " << this->layer_param_.qdata_param().crop_mode();

  // image
  const int crop_size = this->layer_param_.qdata_param().crop_size();
  const int batch_size = this->layer_param_.qdata_param().batch_size();
  CHECK_GT(crop_size, 0);
  (*top)[0]->Reshape(batch_size, channels, crop_size, crop_size);
  this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();
  // Structured label: Action, Reward, NextMaxQ
  (*top)[1]->Reshape(batch_size, 3, 1, 1);
  this->prefetch_label_.Reshape(batch_size, 3, 1, 1);
  // State features
  int totalFeatures = this->layer_param_.qdata_param().num_state_features() 
                    + this->layer_param_.qdata_param().num_actions();
  (*top)[2]->Reshape(batch_size, totalFeatures, 1, 1);
  this->prefetch_state_features_.Reshape(batch_size, totalFeatures, 1, 1);
}

template <typename Dtype>
unsigned int QDataLayer<Dtype>::PrefetchRand() {
  CHECK(prefetch_rng_);
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  return (*prefetch_rng)();
}

// Thread fetching the data
template <typename Dtype>
void QDataLayer<Dtype>::InternalThreadEntry() {
  // At each iteration, sample N windows where N*p are foreground (object)
  // windows and N*(1-p) are background (non-object) windows

  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
  Dtype* top_state_features = this->prefetch_state_features_.mutable_cpu_data();
  const Dtype scale = this->layer_param_.qdata_param().scale();
  const int batchsize = this->layer_param_.qdata_param().batch_size();
  const int crop_size = this->layer_param_.qdata_param().crop_size();
  const int context_pad = this->layer_param_.qdata_param().context_pad();
  const bool mirror = this->layer_param_.qdata_param().mirror();
  const Dtype* mean = this->data_mean_.cpu_data();
  const int mean_off = (this->data_mean_.width() - crop_size) / 2;
  const int mean_width = this->data_mean_.width();
  const int mean_height = this->data_mean_.height();
  cv::Size cv_crop_size(crop_size, crop_size);
  const string& crop_mode = this->layer_param_.qdata_param().crop_mode();

  bool use_square = (crop_mode == "square") ? true : false;
  const int stateFeatures = this->layer_param_.qdata_param().num_state_features(); 
  const int numActions = this->layer_param_.qdata_param().num_actions();
  int totalStateFeatures = stateFeatures + numActions;

  // zero out batch
  caffe_set(this->prefetch_data_.count(), Dtype(0), top_data);

  int itemid = 0;
  // sample from bg set then fg set
  for (int dummy = 0; dummy < batchsize; ++dummy) {
      // sample a window
      vector<float> window = this->windows_[rand() % this->windows_.size()];

      bool do_mirror = false;
      // NOLINT_NEXT_LINE(runtime/threadsafe_fn)
      if (mirror && rand() % 2) {
        do_mirror = true;
      }

      // load the image containing the window
      pair<std::string, vector<int> > image =
          this->image_database_[window[QDataLayer<Dtype>::IMAGE_INDEX]];

      cv::Mat cv_img = cv::imread(image.first, CV_LOAD_IMAGE_COLOR);
      if (!cv_img.data) {
        LOG(ERROR) << "Could not open or find file " << image.first;
        return;
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
        // such that after warping the expanded region to crop_size x crop_size
        // there's exactly context_pad amount of padding on each side
        Dtype context_scale = static_cast<Dtype>(crop_size) /
            static_cast<Dtype>(crop_size - 2*context_pad);

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
            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_width);
        Dtype scale_y =
            static_cast<Dtype>(crop_size)/static_cast<Dtype>(unclipped_height);

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
        // fits in the crop_size x crop_size image (it might not due to rounding)
        if (pad_h + cv_crop_size.height > crop_size) {
          cv_crop_size.height = crop_size - pad_h;
        }
        if (pad_w + cv_crop_size.width > crop_size) {
          cv_crop_size.width = crop_size - pad_w;
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

            top_data[((itemid * channels + c) * crop_size + h + pad_h)
                     * crop_size + w + pad_w]
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

      /*#include <sstream>
      std::ostringstream s; 
      for (int q = 0; q < totalStateFeatures; ++q) {
        s << " " << top_state_features[itemid*totalStateFeatures + q];
      }
      LOG(INFO) << s.str();*/

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
        for (int h = 0; h < crop_size; ++h) {
          for (int w = 0; w < crop_size; ++w) {
            top_data_file.write(reinterpret_cast<char*>(
                &top_data[((itemid * channels + c) * crop_size + h)
                          * crop_size + w]),
                sizeof(Dtype));
          }
        }
      }
      top_data_file.close();
      #endif
      itemid++;
  }
}


INSTANTIATE_CLASS(QDataLayer);

}  // namespace caffe
