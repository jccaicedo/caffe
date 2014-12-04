// Copyright 2014 Juan C. Caicedo

#include <cmath>
#include <cstdlib>
#include <cstring>

#include <stdio.h>

#include "caffe/common.hpp"
#include "caffe/util/regions.hpp"

namespace caffe {

// Kernel to copy an image already in the GPU to the Blob
template <typename Dtype>
__global__ void copyRegionToBlob_kernel(const unsigned char* sourceData, 
                                        Dtype* destData, size_t srcstep, 
                                        int region, int rows, int cols,
                                        int channels, int cropsize, const Dtype* meanImg,
					bbox padding) {
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  int w_off = 14, h_off = 14, meanSize = 256;

  if((padding.x1 <= xIndex) && (padding.y1 <= yIndex) &&
     (xIndex < cols - padding.x2) && (yIndex < rows - padding.y2)){
    const int pixelId = yIndex * srcstep + (3 * xIndex);
    for(int c = 0; c < channels; ++c) {
      int meanImgCoord = (c * meanSize + yIndex + h_off) * meanSize + xIndex + w_off;
      destData[((region * channels + c) * cropsize + yIndex) * cropsize + xIndex] 
          = static_cast<Dtype>(sourceData[pixelId + c])
            - meanImg[ meanImgCoord ];

      //if((xIndex < 100 && xIndex > 95) && (yIndex < 100 && yIndex > 95))
      //if((xIndex == 0) && (yIndex == 0))
      //printf("pixelCoord %d,%d => %d meanImgCoord => %d\n", yIndex,xIndex, pixelId+c ,meanImgCoord);
      //printf("meanPixelCoord %d,%d,%d => %d meanPixelValue => %f\n", yIndex,xIndex,c, meanImgCoord, meanImg[ meanImgCoord ]);
    }
  }
}

// Kernel to copy a portion of the blob to a GPU image
template <typename Dtype>
__global__ void copyBlobToRegion_kernel(const Dtype* blob, unsigned char* image,
                                        size_t srcstep, int region, int rows, int cols, 
                                        int channels, int cropsize, const Dtype* meanImg,
					bbox padding) {
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
  int w_off = 14, h_off = 14, meanSize = 256;

   if( (padding.x1 <= xIndex) && (padding.y1 <= yIndex) && 
       (xIndex < cols - padding.x2) && (yIndex < rows - padding.y2)){
    const int pixelId = yIndex * srcstep + (3 * xIndex);
    for(int c = 0; c < channels; ++c) {
      int meanImgCoord = (c * meanSize + yIndex + h_off) * meanSize + xIndex + w_off;
      image[pixelId + c] = static_cast<unsigned char>(
      blob[((region * channels + c) * cropsize + yIndex) * cropsize + xIndex]
      + meanImg[ meanImgCoord ] );
    }
  }
}

// Kernel to cover a region in GPU image
__global__ void coverRegion_kernel(unsigned char* sourceData, size_t srcstep, 
                                   unsigned char* otherData, size_t otherStep,
                                   int otherRows, int otherCols, bbox region, bool zeros) {
  const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  int channels = 3;
  if((region.x1 <= xIndex) && (region.y1 <= yIndex) &&
     (xIndex < region.x2) && (yIndex < region.y2)){
    const int pixelSource = yIndex * srcstep + (channels * xIndex);
    if(zeros) {
      for(int c = 0; c < channels; ++c) {
          sourceData[pixelSource + c] = static_cast<unsigned char>(0);
      }
    } else {
      int xOther = xIndex;
      int yOther = yIndex;
      if(xOther >= otherCols) {
        xOther = xOther - otherCols;
      }
      if(yIndex >= otherRows) {
        yOther = yOther - otherRows;
      }
      const int pixelOther = yOther * otherStep + (channels * xOther);
      for(int c = 0; c < channels; ++c) {
          sourceData[pixelSource + c] = otherData[pixelOther + c];
      }
    }
  }
}


// Call to the kernel to copy from region to blob
template <typename Dtype>
void copyRegionToBlob(const unsigned char* sourceData, Dtype* destData,
                      size_t srcstep, int region, int rows, int cols, 
                      int channels, int cropsize, const Dtype* meanImg, bbox padding) {
  dim3 blockD(32, 32);
  const dim3 grid((cols + blockD.x - 1)/blockD.x, (rows + blockD.y - 1)/blockD.y);
  copyRegionToBlob_kernel<<<grid, blockD>>>(
                                sourceData, destData, srcstep, region, 
                                rows, cols, channels, cropsize, meanImg, padding);
  CUDA_POST_KERNEL_CHECK;
}

// Call to the kernel to copy from blob to region
template <typename Dtype>
void copyBlobToRegion(const Dtype* blob, unsigned char* image,
                      size_t srcstep, int region, int rows, int cols, 
                      int channels, int cropsize, const Dtype* meanImg, bbox padding) {
  dim3 blockD(32, 32);
  const dim3 grid((cols + blockD.x - 1)/blockD.x, (rows + blockD.y - 1)/blockD.y);
  copyBlobToRegion_kernel<<<grid, blockD>>>(
                                  blob, image, srcstep, region, rows,
                                  cols, channels, cropsize, meanImg, padding);
  CUDA_POST_KERNEL_CHECK;
}

// Call to the kernel to cover a region in an image
void coverRegion(unsigned char* sourceData, size_t srcstep, 
                 unsigned char* otherData, size_t otherStep,
                 int otherRows, int otherCols, bbox region, bool zeros) {
  dim3 blockD(32, 32);
  const dim3 grid((region.x2 + blockD.x - 1)/blockD.x, (region.y2 + blockD.y - 1)/blockD.y);
  coverRegion_kernel<<<grid, blockD>>>(
                                sourceData, srcstep, otherData, otherStep, otherRows, otherCols, region, zeros);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiations
template 
__global__ void copyRegionToBlob_kernel<float>(const unsigned char* sourceData,
                                        float* destData, size_t srcstep, 
                                        int region, int rows, int cols,
                                        int channels, int cropsize, const float* meanImg,
					bbox padding);
template
void copyRegionToBlob<float>(const unsigned char* sourceData, float* destData,
                      size_t srcstep, int region, int rows, int cols,
                      int channels, int cropsize, const float* meanImg, bbox padding);


template
__global__ void copyBlobToRegion_kernel<float>(const float* blob, unsigned char* image,
                                        size_t srcstep, int region, int rows,
                                        int cols, int channels, int cropsize, const float* meanImg,
					bbox padding);

template
void copyBlobToRegion<float>(const float* blob, unsigned char* image,
                             size_t srcstep, int region, int rows, int cols, 
                             int channels, int cropsize, const float* meanImg, bbox padding);


} // namespace caffe
