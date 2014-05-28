// Copyright 2014 Juan C. Caicedo

#ifndef REGIONS_UTIL_HPP_
#define REGIONS_UTIL_HPP_

#include <string>

using std::string;

namespace caffe {

struct bbox {
  int x1;
  int y1;
  int x2;
  int y2;
};

template <typename Dtype>
Dtype* AllocateGpuBlob(int regions, int channels, int cropsize);

void* LoadImageToGpuMat(const string& imageName);

template <typename Dtype>
Dtype* CropAndResizeBoxes_GpuMat(void* srcPrt, int ** boxes,
                              int totalBoxes, int context_pad, const Dtype* meanImg);

template <typename Dtype>
Dtype* CropAndResizeBoxes_Debug(const string& img, int ** boxes,
                              int totalBoxes, int context_pad, const Dtype* meanImg);


template <typename Dtype>
void copyRegionToBlob(const unsigned char* sourceData, Dtype* destData,
                      size_t srcstep, int region, int rows, int cols,
                      int channels, int cropsize, const Dtype* meanImg,
			bbox padding);

template <typename Dtype>
void copyBlobToRegion(const Dtype* blob, unsigned char* image,
                      size_t srcstep, int region, int rows, int cols,
                      int channels, int cropsize, const Dtype* meanImg,
			bbox padding);

} // namespace

#endif // REGIONS_UTIL_HPP_
