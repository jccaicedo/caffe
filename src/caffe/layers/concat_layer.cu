// Copyright 2014 Sergio Guadarrama

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  if (concat_dim_ == 0) {
    int offset_num = 0;
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      caffe_gpu_copy(bottom[i]->count(), bottom_data,
        top_data+(*top)[0]->offset(offset_num));
      offset_num += bottom[i]->num();
    }
  } else if (concat_dim_ == 1) {
    int offset_channel = 0;
    for (int i = 0; i < bottom.size(); ++i) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      int num_elem =
        bottom[i]->channels()*bottom[i]->height()*bottom[i]->width();
      /*if (i==1) {
        #include <sstream>
        LOG(INFO) << " CONCAT ELEMENTS: num_elem=" << num_elem << " NUM_= " << NUM_;
          Blob<Dtype> aux(128,20,1,1);
          Dtype* aux_data = aux.mutable_cpu_data();
          CUDA_CHECK(cudaMemcpy(aux_data, bottom_data, bottom[i]->count()*sizeof(Dtype), cudaMemcpyDeviceToHost));

        for (int q = 0; q < 3; ++q) {
          std::ostringstream s; 
          //return ((n * channels_ + c) * height_ + h) * width_ + w;
          for (int r = 0; r < num_elem; r++) {
            s << " " << aux_data[q*num_elem + r];
          }
          LOG(INFO) << q << s.str() << " eov" << q;
        }
      }*/
      for (int n = 0; n < NUM_; ++n) {
        caffe_gpu_copy(num_elem, bottom_data+bottom[i]->offset(n),
          top_data+(*top)[0]->offset(n, offset_channel));
      }
      offset_channel += bottom[i]->channels();
    }
  } else {
    LOG(FATAL) << "concat_dim along dim" << concat_dim_ <<
      " not implemented yet";
  }
}

template <typename Dtype>
Dtype ConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  if (concat_dim_ == 0) {
    int offset_num = 0;
    for (int i = 0; i < bottom->size(); ++i) {
      Blob<Dtype>* blob = (*bottom)[i];
      Dtype* bottom_diff = blob->mutable_gpu_diff();
      caffe_gpu_copy(blob->count(),
        top_diff+top[0]->offset(offset_num), bottom_diff);
      offset_num += blob->num();
    }
  } else if (concat_dim_ == 1) {
    int offset_channel = 0;
    for (int i = 0; i < bottom->size(); ++i) {
      Blob<Dtype>* blob = (*bottom)[i];
      Dtype* bottom_diff = blob->mutable_gpu_diff();
      int num_elem = blob->channels()*blob->height()*blob->width();
      for (int n = 0; n < NUM_; ++n) {
        caffe_gpu_copy(num_elem, top_diff+top[0]->offset(n, offset_channel),
          bottom_diff+blob->offset(n));
      }
      offset_channel += blob->channels();
    }
  } else {
    LOG(FATAL) << "concat_dim along dim" << concat_dim_ <<
      " not implemented yet";
  }
  return Dtype(0.);
}

INSTANTIATE_CLASS(ConcatLayer);

}  // namespace caffe
