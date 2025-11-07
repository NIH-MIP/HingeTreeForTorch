/*-
 * Nathan Lay
 * AI Resource at National Cancer Institute
 * National Institutes of Health
 * May 2021
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <functional>

#include "torch/extension.h"
#include "HingeTreeCommon.cuh"
#include "ImageToMatrix.h"

#include "hingetree_error.h"

typedef c10::IntArrayRef IntArrayRef;

// From: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
// And from: https://stackoverflow.com/questions/39274472/error-function-atomicadddouble-double-has-already-been-defined
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

//#if __CUDA_ARCH__ < 600
#else
static inline __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

namespace {

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void ForwardKernel(const RealType *d_matrix, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, RealType *d_outData, 
    int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64OutChannels, int64_t i64Rows, int64_t i64Cols) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t k = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;

  if (j < i64OutChannels && k < i64Rows) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

    const RealType * const d_row = d_matrix + k*i64Cols;

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, i64TreeDepth, 1);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const RealType margin = std::abs(signedMargin);

    const RealType * const d_leafWeights = d_inWeights + (j*i64WeightsStride + key)*i64InnerWeightsNum;
    RealType * const d_out = d_outData + (j*i64Rows + k)*i64InnerWeightsNum;

    for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
      d_out[l] += d_leafWeights[l] * margin;
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardThresholdsKernel(const RealType *d_matrix, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inThresholdsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64OutChannels, 
    int64_t i64Rows, int64_t i64Cols) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t k = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;

  if (j < i64OutChannels && k < i64Rows) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;
    RealType * const d_thresholdsGradient = d_inThresholdsGradient + j*i64ThresholdStride;

    const RealType * const d_row = d_matrix + k*i64Cols;

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, i64TreeDepth, 1);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const KeyType thresholdIndex = keyMarginTuple.thresholdIndex;

    const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));

    const RealType * const d_leafWeights = d_inWeights + (j*i64WeightsStride + key)*i64InnerWeightsNum;
    const RealType * const d_outGradient = d_outDataGradient + (j*i64Rows + k)*i64InnerWeightsNum;

    RealType tmpSum = RealType(0);

    for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
      tmpSum += d_leafWeights[l] * d_outGradient[l];

    tmpSum *= -sign;

    atomicAdd(d_thresholdsGradient + thresholdIndex, tmpSum); // Do this just once

      //d_thresholdsGradient[thresholdIndex] += -sign * d_leafWeights[l] * d_outGradient[l];
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardWeightsKernel(const RealType *d_matrix, const RealType *d_inThresholds, const int64_t *d_inOrdinals, /*const RealType *d_inWeights,*/
    const RealType *d_outDataGradient, RealType *d_inWeightsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64OutChannels, 
    int64_t i64Rows, int64_t i64Cols) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t k = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;

  if (j < i64OutChannels && k < i64Rows) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

    const RealType * const d_row = d_matrix + k*i64Cols;

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, i64TreeDepth, 1);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const RealType margin = std::abs(signedMargin);

    const RealType * const d_outGradient = d_outDataGradient + (j*i64Rows + k)*i64InnerWeightsNum;
    RealType * const d_leafWeightsGradient = d_inWeightsGradient + (j*i64WeightsStride + key)*i64InnerWeightsNum;

    for (int64_t l = 0; l < i64InnerWeightsNum; ++l) {
      atomicAdd(d_leafWeightsGradient + l, margin * d_outGradient[l]); // Really bad!
      //d_leafWeightsGradient[l] += margin * d_outGradient[l];
    }
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardDataKernel(const RealType *d_matrix, const int64_t *d_indexMatrix, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inDataGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64OutChannels, 
    int64_t i64Rows, int64_t i64Cols) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t k = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;

  if (j < i64OutChannels && k < i64Rows) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

    const RealType * const d_row = d_matrix + k*i64Cols;
    const int64_t * const d_i64IndexRow = d_indexMatrix + k*i64Cols;

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, i64TreeDepth, 1);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const KeyType thresholdIndex = keyMarginTuple.thresholdIndex;
    const int64_t i64FeatureIndex = d_ordinals[thresholdIndex];
    const int64_t i64ImageIndex = d_i64IndexRow[i64FeatureIndex];

    if (i64ImageIndex >= 0) {
      const RealType * const d_leafWeights = d_inWeights + (j*i64WeightsStride + key)*i64InnerWeightsNum;
      const RealType * const d_outGradient = d_outDataGradient + (j*i64Rows + k)*i64InnerWeightsNum;

      const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));
      RealType tmpSum = RealType(0);

      for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
        tmpSum += d_leafWeights[l] * d_outGradient[l];

      tmpSum *= sign;

      atomicAdd(d_inDataGradient + i64ImageIndex, tmpSum); // Do this just once
    }
  }
}

} // end anonymous namespace

template<typename RealType, unsigned int Dimension, typename TreeTraitsType>
torch::Tensor hingetree_conv_gpu_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, 
  IntArrayRef kernelSize, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;
  typedef bleak::ImageToMatrix<RealType, Dimension> ImageToMatrixType;

  if (kernelSize.size() != Dimension || stride.size() != Dimension || padding.size() != Dimension || dilation.size() != Dimension) {
    InvalidArgumentStream ss;
    ss << "Error: kernel, stride, padding, dilation all expected to have " << Dimension << " dimensions.";
    ss.raise();
    return torch::Tensor();
  }

  if (inData.dim() != Dimension+2 || inThresholds.dim() != 3 || inOrdinals.dim() != 3 || inWeights.dim() < 3) {
    InvalidArgumentStream ss;
    ss << "Error: inData expected to have " << Dimension+2 << " dimensions. inThresholds, inOrdinals expected to have 3 dimensions. inWeights expected to be less than 3 dimensions.";
    ss.raise();
    return torch::Tensor();
  }

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0]) {
    InvalidArgumentStream("Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0].").raise();
    return torch::Tensor();
  }

  const int64_t i64Groups = inData.sizes()[1] / inWeights.sizes()[1];

  if (i64Groups*inWeights.sizes()[1] != inData.sizes()[1]) {
    InvalidArgumentStream("Error: inWeights.shape[1] is expected to divide inData.shape[1].").raise();
    return torch::Tensor();
  }

  if ((inWeights.sizes()[0] % i64Groups) != 0) { // Must also divide output channels
    InvalidArgumentStream("inData.shape[1]//inWeights.shape[1] is expected to divide output channels.").raise();
    return torch::Tensor();
  }

  // C x H x W x ...
  int64_t a_i64ImageSize[Dimension+1] = { 0 };
  std::copy_n(inData.sizes().slice(1).data(), Dimension+1, a_i64ImageSize);

  a_i64ImageSize[0] = 1; // Process 1 channel at a time

  ImageToMatrixType clImageToMatrix;

  clImageToMatrix.SetKernelSize(kernelSize.data());
  clImageToMatrix.SetStride(stride.data());
  clImageToMatrix.SetPadding(padding.data());
  clImageToMatrix.SetDilate(dilation.data());

  if (!clImageToMatrix.Good(a_i64ImageSize)) {
    RunTimeErrorStream("Error: Unable to convert image patches to matrix.").raise();
    return torch::Tensor();
  }

  const int64_t i64KernelCount = clImageToMatrix.ComputeKernelCount();

  if (inOrdinals.min().to(torch::kCPU).item<int64_t>() < 0 || inOrdinals.max().to(torch::kCPU).item<int64_t>() >= i64KernelCount) {
    RunTimeErrorStream("Error: An ordinal value is either negative or larger than or equal to the number of channels.").raise();
    return torch::Tensor();
  }

  const int64_t i64NumLeavesPerTree = inWeights.sizes()[2];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);

  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[2] != TreeTraitsType::GetThresholdCount(i64TreeDepth)) {
    RunTimeErrorStream ss;
    ss << "Error: The tree depth exceeds compile-time constraints or the number of thresholds is incorrect for the tree depth (tree depth = " << i64TreeDepth << ").";
    ss.raise();
    return torch::Tensor();
  }

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64InChannels = inData.sizes()[1];
  const int64_t i64OutChannels = inWeights.sizes()[0];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[2];

  std::vector<IntArrayRef::value_type> vSizes;

  vSizes.resize(2);
  vSizes[0] = inData.sizes()[0]; // batch size
  vSizes[1] = inWeights.sizes()[0]; // Number of output channels

  {
    const auto tmpSizes = clImageToMatrix.ComputeOutputSize(a_i64ImageSize);
    vSizes.insert(vSizes.end(), tmpSizes.begin(), tmpSizes.end());
  }

  {
    auto inWeightsSlice = inWeights.sizes().slice(3);
    vSizes.insert(vSizes.end(), inWeightsSlice.begin(), inWeightsSlice.end());
  }

  int64_t i64InnerWeightsNum = 1;
  
  {
    auto inWeightsSlice = inWeights.sizes().slice(3);
    i64InnerWeightsNum = std::accumulate(inWeightsSlice.begin(), inWeightsSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }

  int64_t i64InChannelSize = 1;

  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InChannelSize = std::accumulate(inDataSlice.begin(), inDataSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }

  const int64_t i64OutDataImageSize = clImageToMatrix.ComputeOutputCount(a_i64ImageSize);

  // Index matrix dimensions
  int64_t i64Rows = 0;
  int64_t i64Cols = 0;

  clImageToMatrix.ComputeMatrixDimensions(i64Rows, i64Cols, a_i64ImageSize);
  torch::Tensor indexMatrix;
  torch::Tensor featureMatrix;

  {
    auto clOptions = torch::TensorOptions().dtype(torch::kInt64).device(inData.device());
    indexMatrix = torch::empty({ i64Rows, i64Cols }, clOptions);
  }

  int64_t * const d_indexMatrix = indexMatrix.data_ptr<int64_t>();

  {
    auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());
    featureMatrix = torch::empty({ i64Rows, i64Cols }, clOptions);
  }

  RealType * const d_featureMatrix = featureMatrix.data_ptr<RealType>();

  clImageToMatrix.ExtractIndexMatrixGPU(d_indexMatrix, a_i64ImageSize);

  torch::Tensor outData;

  {
    auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());
    outData = torch::zeros(IntArrayRef(vSizes.data(), vSizes.size()),clOptions);
  }

  RealType * const d_outData = outData.data_ptr<RealType>();

  const RealType * const d_inData = inData.data_ptr<RealType>();
  const RealType * const d_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const d_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const d_inWeights = inWeights.data_ptr<RealType>();

  // Trees vs Patch Rows (m_iRows)
  const dim3 threadsPerBlock(16, 16);
  const dim3 numBlocks((i64OutChannels + threadsPerBlock.x-1) / threadsPerBlock.x, (i64Rows + threadsPerBlock.y-1) / threadsPerBlock.y);

  const int i64WeightsStride = i64InChannels * i64NumLeavesPerTree;
  const int i64ThresholdStride = i64InChannels * i64NumDecisionsPerTree;

  for (int64_t i = 0; i < i64BatchSize; ++i) {
    for (int64_t g = 0; g < i64Groups; ++g) {
      for (int64_t c = 0; c < i64InChannels; ++c) {
        clImageToMatrix.ExtractMatrixGPU(d_featureMatrix, d_inData + ((i*i64Groups + g)*i64InChannels + c)*i64InChannelSize, d_indexMatrix, a_i64ImageSize);

        ForwardKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(d_featureMatrix, d_inThresholds + ((g*i64OutChannels + 0)*i64InChannels + c)*i64NumDecisionsPerTree, 
            d_inOrdinals + ((g*i64OutChannels + 0)*i64InChannels + c)*i64NumDecisionsPerTree, 
            d_inWeights + (((g*i64OutChannels + 0)*i64InChannels + c)*i64NumLeavesPerTree + 0)*i64InnerWeightsNum,
            d_outData + (i*i64Groups + g)*i64OutChannels*i64Rows*i64InnerWeightsNum, 
            i64TreeDepth, i64ThresholdStride, i64WeightsStride, i64InnerWeightsNum, i64OutChannels, i64Rows, i64Cols);
      }
    }
  }

  return outData;
}

template<typename RealType, unsigned int Dimension, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_conv_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad, IntArrayRef kernelSize, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;
  typedef bleak::ImageToMatrix<RealType, Dimension> ImageToMatrixType;

  if (bInOrdinalsGrad) { // NEVER differentiable
    RunTimeErrorStream("Error: Gradient on inOrdinals is requested, but inOrdinals never has a gradient.").raise();
    return std::vector<torch::Tensor>();
  }

  if (kernelSize.size() != Dimension || stride.size() != Dimension || padding.size() != Dimension || dilation.size() != Dimension) {
    InvalidArgumentStream ss;
    ss << "Error: kernel, stride, padding, dilation all expected to have " << Dimension << " dimensions.";
    ss.raise();
    return std::vector<torch::Tensor>();
  }

  if (inData.dim() != Dimension+2 || inThresholds.dim() != 3 || inOrdinals.dim() != 3 || inWeights.dim() < 3) {
    InvalidArgumentStream ss;
    ss << "Error: inData expected to have " << Dimension+2 << " dimensions. inThresholds, inOrdinals expected to have 3 dimensions. inWeights expected to be less than 3 dimensions.";
    ss.raise();
    return std::vector<torch::Tensor>();
  }

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0]) {
    InvalidArgumentStream("Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0].").raise();
    return std::vector<torch::Tensor>();
  }

  const int64_t i64Groups = inData.sizes()[1] / inWeights.sizes()[1];

  if (i64Groups*inWeights.sizes()[1] != inData.sizes()[1]) {
    InvalidArgumentStream("Error: inWeights.shape[1] is expected to divide inData.shape[1].").raise();
    return std::vector<torch::Tensor>();
  }

  if ((inWeights.sizes()[0] % i64Groups) != 0) { // Must also divide output channels
    InvalidArgumentStream("inData.shape[1]//inWeights.shape[1] is expected to divide output channels.").raise();
    return std::vector<torch::Tensor>();
  }

  // C x H x W x ...
  int64_t a_i64ImageSize[Dimension+1] = { 0 };
  std::copy_n(inData.sizes().slice(1).data(), Dimension+1, a_i64ImageSize);

  a_i64ImageSize[0] = 1; // Process 1 channel at a time

  ImageToMatrixType clImageToMatrix;

  clImageToMatrix.SetKernelSize(kernelSize.data());
  clImageToMatrix.SetStride(stride.data());
  clImageToMatrix.SetPadding(padding.data());
  clImageToMatrix.SetDilate(dilation.data());

  if (!clImageToMatrix.Good(a_i64ImageSize)) {
    RunTimeErrorStream("Error: Unable to convert image patches to matrix.").raise();
    return std::vector<torch::Tensor>();
  }

  const int64_t i64KernelCount = clImageToMatrix.ComputeKernelCount();

  if (inOrdinals.min().to(torch::kCPU).item<int64_t>() < 0 || inOrdinals.max().to(torch::kCPU).item<int64_t>() >= i64KernelCount) {
    RunTimeErrorStream("Error: An ordinal value is either negative or larger than or equal to the number of channels.").raise();
    return std::vector<torch::Tensor>();
  }

  const int64_t i64NumLeavesPerTree = inWeights.sizes()[2];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);

  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[2] != TreeTraitsType::GetThresholdCount(i64TreeDepth)) {
    RunTimeErrorStream ss;
    ss << "Error: The tree depth exceeds compile-time constraints or the number of thresholds is incorrect for the tree depth (tree depth = " << i64TreeDepth << ").";
    ss.raise();
    return std::vector<torch::Tensor>();
  }

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64InChannels = inData.sizes()[1];
  const int64_t i64OutChannels = inWeights.sizes()[0];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[2];

  std::vector<IntArrayRef::value_type> vSizes;

  vSizes.resize(2);
  vSizes[0] = inData.sizes()[0]; // batch size
  vSizes[1] = inWeights.sizes()[0]; // Number of output channels

  {
    const auto tmpSizes = clImageToMatrix.ComputeOutputSize(a_i64ImageSize);
    vSizes.insert(vSizes.end(), tmpSizes.begin(), tmpSizes.end());
  }

  {
    auto inWeightsSlice = inWeights.sizes().slice(3);
    vSizes.insert(vSizes.end(), inWeightsSlice.begin(), inWeightsSlice.end());
  }

  if (outDataGrad.sizes() != IntArrayRef(vSizes.data(), vSizes.size())) {
    InvalidArgumentStream ss;
    ss << "Error: outDataGrad.shape does not match the expected shape (" << outDataGrad.sizes() << " != " << IntArrayRef(vSizes.data(), vSizes.size()) << ").";
    ss.raise();
    return std::vector<torch::Tensor>();
  }

  int64_t i64InnerWeightsNum = 1;
  
  {
    auto inWeightsSlice = inWeights.sizes().slice(3);
    i64InnerWeightsNum = std::accumulate(inWeightsSlice.begin(), inWeightsSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }

  int64_t i64InChannelSize = 1;

  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InChannelSize = std::accumulate(inDataSlice.begin(), inDataSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }

  const int64_t i64OutDataImageSize = clImageToMatrix.ComputeOutputCount(a_i64ImageSize);

  // Index matrix dimensions
  int64_t i64Rows = 0;
  int64_t i64Cols = 0;

  clImageToMatrix.ComputeMatrixDimensions(i64Rows, i64Cols, a_i64ImageSize);
  torch::Tensor indexMatrix;
  torch::Tensor featureMatrix;

  {
    auto clOptions = torch::TensorOptions().dtype(torch::kInt64).device(inData.device());
    indexMatrix = torch::empty({ i64Rows, i64Cols }, clOptions);
  }

  int64_t * const d_indexMatrix = indexMatrix.data_ptr<int64_t>();

  {
    auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());
    featureMatrix = torch::empty({ i64Rows, i64Cols }, clOptions);
  }

  RealType * const d_featureMatrix = featureMatrix.data_ptr<RealType>();

  clImageToMatrix.ExtractIndexMatrixGPU(d_indexMatrix, a_i64ImageSize);

  const RealType * const d_outDataGrad = outDataGrad.data_ptr<RealType>();

  const RealType * const d_inData = inData.data_ptr<RealType>();
  const RealType * const d_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const d_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const d_inWeights = inWeights.data_ptr<RealType>();

  // Trees vs Patch Rows (m_iRows)
  const dim3 threadsPerBlock(16, 16);
  const dim3 numBlocks((i64OutChannels + threadsPerBlock.x-1) / threadsPerBlock.x, (i64Rows + threadsPerBlock.y-1) / threadsPerBlock.y);

  const int i64WeightsStride = i64InChannels * i64NumLeavesPerTree;
  const int i64ThresholdStride = i64InChannels * i64NumDecisionsPerTree;

  //auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());
  std::vector<torch::Tensor> vGrads(4);

  if (bInDataGrad) {
    torch::Tensor inDataGrad = torch::zeros_like(inData);
    RealType * const d_inDataGrad = inDataGrad.data_ptr<RealType>();

    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t g = 0; g < i64Groups; ++g) {
        for (int64_t c = 0; c < i64InChannels; ++c) {
          clImageToMatrix.ExtractMatrixGPU(d_featureMatrix, d_inData + ((i*i64Groups + g)*i64InChannels + c)*i64InChannelSize, d_indexMatrix, a_i64ImageSize);

          BackwardDataKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(d_featureMatrix, d_indexMatrix,
            d_inThresholds + ((g*i64OutChannels + 0)*i64InChannels + c)*i64NumDecisionsPerTree,
            d_inOrdinals + ((g*i64OutChannels + 0)*i64InChannels + c)*i64NumDecisionsPerTree,
            d_inWeights + (((g*i64OutChannels + 0)*i64InChannels + c)*i64NumLeavesPerTree + 0)*i64InnerWeightsNum,
            d_outDataGrad + (i*i64Groups + g)*i64OutChannels*i64Rows*i64InnerWeightsNum,
            d_inDataGrad + ((i*i64Groups + g)*i64InChannels + c)*i64InChannelSize, 
            i64TreeDepth, i64ThresholdStride, i64WeightsStride, i64InnerWeightsNum, i64OutChannels, i64Rows, i64Cols);
        }
      }
    }

    vGrads[0] = inDataGrad;
  }

  if (bInThresholdsGrad) {
    torch::Tensor inThresholdsGrad = torch::zeros_like(inThresholds);
    RealType * const d_inThresholdsGrad = inThresholdsGrad.data_ptr<RealType>();

    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t g = 0; g < i64Groups; ++g) {
        for (int64_t c = 0; c < i64InChannels; ++c) {
          clImageToMatrix.ExtractMatrixGPU(d_featureMatrix, d_inData + ((i*i64Groups + g)*i64InChannels + c)*i64InChannelSize, d_indexMatrix, a_i64ImageSize);

          BackwardThresholdsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(d_featureMatrix,
            d_inThresholds + ((g*i64OutChannels + 0)*i64InChannels + c)*i64NumDecisionsPerTree,
            d_inOrdinals + ((g*i64OutChannels + 0)*i64InChannels + c)*i64NumDecisionsPerTree,
            d_inWeights + (((g*i64OutChannels + 0)*i64InChannels + c)*i64NumLeavesPerTree + 0)*i64InnerWeightsNum,
            d_outDataGrad + (i*i64Groups + g)*i64OutChannels*i64Rows*i64InnerWeightsNum,
            d_inThresholdsGrad + ((g*i64OutChannels + 0)*i64InChannels +c)*i64NumDecisionsPerTree,
            i64TreeDepth, i64ThresholdStride, i64WeightsStride, i64InnerWeightsNum, i64OutChannels, i64Rows, i64Cols);

        }
      }
    }

    vGrads[1] = inThresholdsGrad;
  }

  if (bInWeightsGrad) {
    torch::Tensor inWeightsGrad = torch::zeros_like(inWeights);
    RealType * const d_inWeightsGrad = inWeightsGrad.data_ptr<RealType>();

    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t g = 0; g < i64Groups; ++g) {
        for (int64_t c = 0; c < i64InChannels; ++c) {
          clImageToMatrix.ExtractMatrixGPU(d_featureMatrix, d_inData + ((i*i64Groups + g)*i64InChannels + c)*i64InChannelSize, d_indexMatrix, a_i64ImageSize);

          BackwardWeightsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(d_featureMatrix,
            d_inThresholds + ((g*i64OutChannels + 0)*i64InChannels + c)*i64NumDecisionsPerTree,
            d_inOrdinals + ((g*i64OutChannels + 0)*i64InChannels + c)*i64NumDecisionsPerTree,
            /*d_inWeights + (((g*i64OutChannels + 0)*i64InChannels + c)*i64NumLeavesPerTree + 0)*i64InnerWeightsNum,*/
            d_outDataGrad + (i*i64Groups + g)*i64OutChannels*i64Rows*i64InnerWeightsNum,
            d_inWeightsGrad + (((g*i64OutChannels + 0)*i64InChannels + c)*i64NumLeavesPerTree + 0)*i64InnerWeightsNum,
            i64TreeDepth, i64ThresholdStride, i64WeightsStride, i64InnerWeightsNum, i64OutChannels, i64Rows, i64Cols);
            
        }
      }
    }

    vGrads[3] = inWeightsGrad;
  }

  return vGrads;
}

// 1D
template torch::Tensor hingetree_conv_gpu_forward<float, 1, bleak::HingeTreeCommon<float>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_gpu_forward<double, 1, bleak::HingeTreeCommon<double>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_gpu_forward<float, 1, bleak::HingeFernCommon<float>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_gpu_forward<double, 1, bleak::HingeFernCommon<double>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);

template std::vector<torch::Tensor> hingetree_conv_gpu_backward<float, 1, bleak::HingeTreeCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_gpu_backward<double, 1, bleak::HingeTreeCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_gpu_backward<float, 1, bleak::HingeFernCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_gpu_backward<double, 1, bleak::HingeFernCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);

// 2D
template torch::Tensor hingetree_conv_gpu_forward<float, 2, bleak::HingeTreeCommon<float>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_gpu_forward<double, 2, bleak::HingeTreeCommon<double>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_gpu_forward<float, 2, bleak::HingeFernCommon<float>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_gpu_forward<double, 2, bleak::HingeFernCommon<double>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);

template std::vector<torch::Tensor> hingetree_conv_gpu_backward<float, 2, bleak::HingeTreeCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_gpu_backward<double, 2, bleak::HingeTreeCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_gpu_backward<float, 2, bleak::HingeFernCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_gpu_backward<double, 2, bleak::HingeFernCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);

// 3D
template torch::Tensor hingetree_conv_gpu_forward<float, 3, bleak::HingeTreeCommon<float>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_gpu_forward<double, 3, bleak::HingeTreeCommon<double>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_gpu_forward<float, 3, bleak::HingeFernCommon<float>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_gpu_forward<double, 3, bleak::HingeFernCommon<double>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);

template std::vector<torch::Tensor> hingetree_conv_gpu_backward<float, 3, bleak::HingeTreeCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_gpu_backward<double, 3, bleak::HingeTreeCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_gpu_backward<float, 3, bleak::HingeFernCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_gpu_backward<double, 3, bleak::HingeFernCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);

