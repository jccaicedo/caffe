// Copyright 2014 Juan C. Caicedo

// Make sure to add opencv_gpu to the Makefile in line 78
#include <opencv2/opencv.hpp>
#include <opencv2/gpu/gpu.hpp>

#include <string>

#include "caffe/common.hpp"
#include "caffe/util/regions.hpp"
#include "caffe/proto/caffe.pb.h"

using std::string;

namespace caffe {

template <typename Dtype>
cv::gpu::GpuMat CropImage(cv::gpu::GpuMat* cv_img, int x1, int y1, int x2, int y2, int context_pad, int cropsize, bbox & padding) {
      //const int context_pad = 16;
      bool use_square = false;
      cv::Size cv_crop_size(cropsize, cropsize);

      padding.x1 = 0;
      padding.y1 = 0;
      padding.x2 = 0;
      padding.y2 = 0;

      if (context_pad > 0 || use_square) {
	// Amount of resizing needed to make room for context
        Dtype context_scale = static_cast<Dtype>(cropsize) /
            static_cast<Dtype>(cropsize - 2*context_pad);

	// Compute center and size of box
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
	// Extend the box from the center to a larger region to include context
        x1 = static_cast<int>(round(center_x - half_width*context_scale));
        x2 = static_cast<int>(round(center_x + half_width*context_scale));
        y1 = static_cast<int>(round(center_y - half_height*context_scale));
        y2 = static_cast<int>(round(center_y + half_height*context_scale));

	// Shift coordinates inside the image boundaries
        int unclipped_height = y2-y1+1;
        int unclipped_width = x2-x1+1;
        padding.x1 = std::max(0, -x1);
        padding.y1 = std::max(0, -y1);
        padding.x2 = std::max(0, x2 - cv_img->cols + 1);
        padding.y2 = std::max(0, y2 - cv_img->rows + 1);
        x1 = x1 + padding.x1;
        x2 = x2 - padding.x2;
        y1 = y1 + padding.y1;
        y2 = y2 - padding.y2;

        CHECK_GT(x1, -1);
        CHECK_GT(y1, -1);
        CHECK_LT(x2, cv_img->cols);
        CHECK_LT(y2, cv_img->rows);

	// New height and width
        int clipped_height = y2-y1+1;
        int clipped_width = x2-x1+1;

	// Amount of rescaling to match the "original" warping
        Dtype scale_x =
            static_cast<Dtype>(cropsize)/static_cast<Dtype>(unclipped_width);
        Dtype scale_y =
            static_cast<Dtype>(cropsize)/static_cast<Dtype>(unclipped_height);

	// Dimensions of the clipped and warped region
        cv_crop_size.width =
            static_cast<int>(round(static_cast<Dtype>(clipped_width)*scale_x));
        cv_crop_size.height =
            static_cast<int>(round(static_cast<Dtype>(clipped_height)*scale_y));
	
	// Amount of padding required on each direction
        padding.x1 = static_cast<int>(round(static_cast<Dtype>(padding.x1)*scale_x));
        padding.x2 = static_cast<int>(round(static_cast<Dtype>(padding.x2)*scale_x));
        padding.y1 = static_cast<int>(round(static_cast<Dtype>(padding.y1)*scale_y));
        padding.y2 = static_cast<int>(round(static_cast<Dtype>(padding.y2)*scale_y));

	// Make sure padding still fits the crop size (may not due to rounding)
        if (padding.y1 + cv_crop_size.height > cropsize) {
          cv_crop_size.height = cropsize - padding.y1;
        }
        if (padding.x1 + cv_crop_size.width > cropsize) {
          cv_crop_size.width = cropsize - padding.x1;
        }
      }

      // Extract the region from the original image
      cv::Rect roi(x1, y1, x2-x1+1, y2-y1+1);
      cv::gpu::GpuMat dev_roi = (*cv_img)(roi).clone();
      cv::gpu::GpuMat dev_crop, cv_cropped_img;
      // Resize the image to the crop size
      cv::gpu::resize(dev_roi, dev_crop,
          cv_crop_size, 0, 0, cv::INTER_LINEAR); //INTER_CUBIC
 
      cv::gpu::copyMakeBorder(dev_crop, cv_cropped_img, padding.y1, padding.y2, padding.x1, padding.x2,
                         cv::BORDER_CONSTANT, cv::Scalar(0,0,0));

     return cv_cropped_img;
}

// Allocate memory for a blob with multiple images
template <typename Dtype>
Dtype* AllocateGpuBlob(int regions, int channels, int cropsize) {
  Dtype* blob;
  const int totalBytes = sizeof(Dtype)*regions*cropsize*cropsize*channels;
  CUDA_CHECK(cudaMalloc(&blob,totalBytes));
  CUDA_CHECK(cudaMemset(blob, 0, totalBytes));
  return blob;
}

// Load an image from disk to the GPU
void* LoadImageToGpuMat(const string& imageName) {
  cv::Mat src_host = cv::imread(imageName, CV_LOAD_IMAGE_COLOR);
  LOG(INFO) << "Cropping and resizing regions for image " << imageName << " (" << src_host.rows << "x" << src_host.cols << ")";
  cv::gpu::GpuMat* src = new cv::gpu::GpuMat();
  src->upload(src_host);
  return static_cast<void*>(src);
}

// Crop and resize regions in the GPU
template <typename Dtype> 
Dtype* CropAndResizeBoxes_GpuMat(void* srcPrt, int ** boxes,
                              int totalBoxes, int context_pad, int cropsize, const Dtype* meanImg) {
  cv::gpu::GpuMat* src = static_cast<cv::gpu::GpuMat*>(srcPrt);
  Dtype* dev_blob = AllocateGpuBlob<Dtype>(totalBoxes, 3, cropsize);
  for(int i = 0; i < totalBoxes; ++i) {
    bbox pad;
    cv::gpu::GpuMat dev_crop = CropImage<Dtype>(src, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], context_pad, cropsize, pad);
    copyRegionToBlob<Dtype>(dev_crop.data, dev_blob, dev_crop.step, i, dev_crop.rows, dev_crop.cols, 3, cropsize, meanImg, pad);
  }
  return dev_blob;
}

// Crop and resize regions in the GPU
void CoverBoxes_GpuMat(void* srcPrt, int ** boxes, int totalBoxes) {
  cv::gpu::GpuMat* src = static_cast<cv::gpu::GpuMat*>(srcPrt);
  for(int i = 0; i < totalBoxes; ++i) {
    bbox region;
    region.x1 = boxes[i][0];
    region.y1 = boxes[i][1];
    region.x2 = boxes[i][2];
    region.y2 = boxes[i][3];
    coverRegion(src->data, src->step, 3, region);
  }
  // Test to check that the covering function is working properly 
  /*  cv::Mat dst_host2(*src);
    std::stringstream outf2;
    outf2 << "/home/caicedo/out_.jpg";
    cv::imwrite(outf2.str(), dst_host2); */
}

// Explicit instantiation
template float* CropAndResizeBoxes_GpuMat<float>(void* src /*const string& imageName*/, int ** boxes, 
                                   int totalBoxes, int context_pad, int cropsize, const float* meanImg);


// Explicit instantiation
template float* CropAndResizeBoxes_Debug<float>(const string& imageName, int ** boxes, 
                                   int totalBoxes, int context_pad, int cropsize, const float* meanImg);

template float* AllocateGpuBlob(int regions, int channels, int cropsize);

// Auxiliary debugging function
template <typename Dtype>
Dtype* CropAndResizeBoxes_Debug(const string& imageName, int ** boxes, 
                          int totalBoxes, int context_pad, int cropsize, const Dtype* meanImg) {
  void* prt = LoadImageToGpuMat(imageName);
  cv::gpu::GpuMat* src = static_cast<cv::gpu::GpuMat*>(prt);
  Dtype* dev_blob = AllocateGpuBlob<Dtype>(totalBoxes, 3, cropsize);

  // Test to check that the cropping function is working properly 
/*  for(int i = 0; i < 5; ++i) {
    cv::gpu::GpuMat dst = CropImage<Dtype>(src, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]);
    cv::Mat dst_host(dst);
    unsigned pos = imageName.find(".jpg") - 5;
    std::stringstream outf;
    outf << "/home/caicedo/what/out_" << imageName.substr(pos) << "." << i << ".jpg";
    cv::imwrite(outf.str(), dst_host);
  }*/
  // The real job
  for(int i = 0; i < totalBoxes; ++i) {
    bbox pad;
    cv::gpu::GpuMat dev_crop = CropImage<Dtype>(src, boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], context_pad, cropsize, pad);

    //if(pad.x1 + pad.x2 + pad.y1 + pad.y2 > 0){
    cv::Mat dst_host(dev_crop);
    unsigned pos = imageName.find(".jpg") - 5;
    std::stringstream outf;
    outf << "/home/caicedo/what/out_" << imageName.substr(pos) << "." << boxes[i][0] << "_" << boxes[i][1] << "_" << boxes[i][2] << "_" << boxes[i][3] << ".jpg";
    LOG(INFO) << "Box:" << boxes[i][0] << "," << boxes[i][1] << "," << boxes[i][2] << "," << boxes[i][3] << " Pad" << pad.x1 << "," << pad.y1 << "," << pad.x2 << "," << pad.y2;
    cv::imwrite(outf.str(), dst_host);
    //}

    copyRegionToBlob<Dtype>(dev_crop.data, dev_blob, dev_crop.step, i, dev_crop.rows, dev_crop.cols, 3, cropsize, meanImg, pad);
  }
  // Test to chech that copying the image is working properly 
/*  for(int i = 0; i < 5; ++i) {
    cv::gpu::GpuMat dev_dst(227,227,src->type());
    copyBlobToRegion<Dtype>(dev_blob, dev_dst.data, dev_dst.step, i, dev_dst.rows, dev_dst.cols, 3, cropsize, meanImg);
    cv::Mat host_dst;
    dev_dst.download(host_dst);
    unsigned pos = imageName.find(".jpg") - 5;
    std::stringstream outf;
    outf << "/home/caicedo/what/out_" << imageName.substr(pos) << "." << (i+10) << ".jpg";
    cv::imwrite(outf.str(), host_dst);
  }*/
  // Visualize the mean image that the GPU ends up working with 
  /*cv::gpu::GpuMat dev_dst(256,256,src->type());
  bbox padding;
  copyBlobToRegion<Dtype>(meanImg, dev_dst.data, dev_dst.step, 0, dev_dst.rows, dev_dst.cols, 3, 256, meanImg, padding);
  cv::Mat host_dst;
  dev_dst.download(host_dst);
  std::stringstream outf;
  outf << "/home/caicedo/what/out_meanImage.png";
  cv::imwrite(outf.str(), host_dst);*/
  return dev_blob;
}

} // namespace caffe
