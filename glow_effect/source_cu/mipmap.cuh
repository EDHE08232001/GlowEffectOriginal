/*******************************************************************************************************************
 * FILE NAME   :    mipmap.cuh
 *
 * PROJECT NAME:    Cuda Learning
 *
 * DESCRIPTION :    Header file for mipmap generator and access
 *
 * VERSION HISTORY
 * YYYY/MMM/DD      Author          Comments
 * 2022 OCT 10      Yu Liu          Creation
 * 2022 OCT 26      Yu Liu          Moved V-shaped curve into CUDA
 * 2022 OCT 27      Yu Liu          Proved texReadMode = cudaReadModeNormalizedFloat to be a must for linear filter
 *                                  also corrected phase shift by using x+1.f/y+1.f rather than x+0.5/y+0.5
 *
 ********************************************************************************************************************/

#ifndef MIPMAP_CUH
#define MIPMAP_CUH

#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <vector_types.h>
#include "old_movies.cuh"

extern bool button_State[5];

/**
 * @brief CUDA kernel to generate a mipmap level by downscaling an input texture.
 *
 * @param mipOutput  CUDA surface object for the output mipmap.
 * @param mipInput   CUDA texture object for the input image.
 * @param imageW     Width of the output mipmap level.
 * @param imageH     Height of the output mipmap level.
 */
__global__ void d_gen_mipmap(cudaSurfaceObject_t mipOutput, cudaTextureObject_t mipInput, uint imageW, uint imageH);

/**
 * @brief Generates a mipmap chain for a given CUDA mipmapped array.
 *
 * @param mipmapArray Reference to the CUDA mipmapped array to process.
 * @param size Initial size (extent) of the highest resolution mipmap level.
 */
void gen_mipmap(cudaMipmappedArray_t& mipmapArray, cudaExtent size);

/**
 * @brief Kernel to sample a mipmapped texture with varying LOD per pixel.
 *
 * @param texEngine CUDA texture object for the mipmapped texture.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param lod Pointer to an array of LOD values for each pixel.
 * @param dout Output buffer to store the resulting uchar4 color values.
 */
__global__ void d_get_mipmap(cudaTextureObject_t texEngine, const int width, const int height, const float* lod, uchar4* dout);

/**
 * @brief Kernel to sample a mipmapped texture with a uniform LOD.
 *
 * @param texEngine CUDA texture object for the mipmapped texture.
 * @param width Width of the image.
 * @param height Height of the image.
 * @param scale Scale factor used to compute the LOD.
 * @param dout Output buffer to store the resulting uchar4 color values.
 */
__global__ void d_get_mipmap(cudaTextureObject_t texEngine, const int width, const int height, const float scale, uchar4* dout);

/**
 * @brief Retrieves a mipmap image with a uniform blur applied using CUDA.
 *
 * @param mm_array The CUDA mipmapped array containing the mipmap levels.
 * @param img_size Dimensions of the image and number of mipmap levels (int3: {width, height, n_level}).
 * @param scale Scale factor used to compute the LOD for mipmap sampling.
 * @param dout Host output buffer for storing the resulting uchar4 image.
 */
void get_mipmap(cudaMipmappedArray_t mm_array, const int3 img_size, const float scale, uchar4* dout);

/**
 * @brief Filters an image by generating mipmap levels and retrieving a blurred version.
 *
 * @param width Width of the input image.
 * @param height Height of the input image.
 * @param scale Scale factor used for the blur effect.
 * @param src_img Pointer to the input image on the host.
 * @param dst_img Pointer to the output image on the host.
 */
void filter_mipmap(const int width, const int height, const float scale, const uchar4* src_img, uchar4* dst_img);

#endif // MIPMAP_CUH
