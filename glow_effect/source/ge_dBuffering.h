#ifndef GE_DBUFFERING_H
#define GE_DBUFFERING_H

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <vector>
#include <string>

/**
* @file: ge_dBuffering
* @brief Declarations for double-buffered GPU segmentation + mipmap operations
*
* This header declares:
*	- A function that executes both segmentation and mipmap in a single GPU pipeline
*	- (Optional) Additional helper functions or classes to manage double buffering
*
* Include this in yout .cpp/.cu files to access these routines
*/

namespace GE {
	/**
	* @brief Run segmentation + mipmap in a single GPU pass, returning blurred masks on CPU
	*
	* @param trt_path			Path to serialized TensorRT engine
	* @param batchTensor		A 4d float tensor [N, C, H, W] for input images
	* @return A vector of single-channel (or RGBA) OpenCV mats containing the final blurred/mipmapped masks
	*/
	std::vector<cv::Mat> runSegmentationAndMipmap(const std::string& trt_plan, const torch::Tensor& batchTensor);
}

#endif // GE_DBUFFERING_H