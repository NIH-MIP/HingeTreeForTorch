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
#include <iostream>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <utility>
#include <functional>

#include "torch/extension.h"
#include "HingeTreeCommon.h"
#include "ImageToMatrix.h"

#include "hingetree_error.h"

typedef c10::IntArrayRef IntArrayRef;

#ifndef WITH_CUDA
template<typename RealType, unsigned int Dimension, typename TreeTraitsType>
torch::Tensor hingetree_conv_gpu_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef) {
  RunTimeErrorStream("Error: GPU support not compiled in.").raise();
  return torch::Tensor();
}

template<typename RealType, unsigned int Dimension, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_conv_gpu_backward(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef) {
  RunTimeErrorStream("Error: GPU support not compiled in.").raise();
  return std::vector<torch::Tensor>();
}
#endif // WITH_CUDA

// Returns convolution result and small index matrix (for backward)
template<typename RealType, unsigned int Dimension, typename TreeTraitsType>
torch::Tensor hingetree_conv_cpu_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, 
  IntArrayRef kernelSize, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {

  typedef typename TreeTraitsType::KeyType KeyType;
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

  if (inOrdinals.min().item<int64_t>() < 0 || inOrdinals.max().item<int64_t>() >= i64KernelCount) {
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

  int64_t * const p_indexMatrix = indexMatrix.data_ptr<int64_t>();

  {
    auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());
    featureMatrix = torch::empty({ i64Rows, i64Cols }, clOptions);
  }

  RealType * const p_featureMatrix = featureMatrix.data_ptr<RealType>();

  clImageToMatrix.ExtractIndexMatrix(p_indexMatrix, a_i64ImageSize);

  torch::Tensor outData;

  {
    auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());
    outData = torch::zeros(IntArrayRef(vSizes.data(), vSizes.size()),clOptions);
  }

  RealType * const p_outData = outData.data_ptr<RealType>();

  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();

  for (int64_t i = 0; i < i64BatchSize; ++i) {
    for (int64_t g = 0; g < i64Groups; ++g) {
      for (int64_t c = 0; c < i64InChannels; ++c) {
        clImageToMatrix.ExtractMatrix(p_featureMatrix, p_inData + ((i*i64Groups + g)*i64InChannels + c)*i64InChannelSize, p_indexMatrix, a_i64ImageSize);

        for (int64_t j = 0; j < i64OutChannels; ++j) {
          const RealType * const p_thresholds = p_inThresholds + ((g*i64OutChannels + j)*i64InChannels + c)*i64NumDecisionsPerTree;
          const int64_t * const p_ordinals = p_inOrdinals + ((g*i64OutChannels + j)*i64InChannels + c)*i64NumDecisionsPerTree;

          for (int64_t k = 0; k < i64Rows; ++k) {
            const RealType * const p_row = p_featureMatrix + k*i64Cols;

            const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_row, p_thresholds, p_ordinals, i64TreeDepth, 1);

            const KeyType key = std::get<0>(clKeyMarginTuple);
            const RealType signedMargin = std::get<1>(clKeyMarginTuple);
            const RealType margin = std::abs(signedMargin);

            const RealType * const p_leafWeights = p_inWeights + (((g*i64OutChannels + j)*i64InChannels + c)*i64NumLeavesPerTree + key)*i64InnerWeightsNum;

            for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
              p_outData[(((i*i64Groups + g)*i64OutChannels + j)*i64OutDataImageSize + k)*i64InnerWeightsNum + l] += p_leafWeights[l]*margin;
          }
        }
      }
    }
  }

  return outData;
}

template<typename RealType, unsigned int Dimension, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_conv_cpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad, IntArrayRef kernelSize, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  typedef typename TreeTraitsType::KeyType KeyType;
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

  if (inOrdinals.min().item<int64_t>() < 0 || inOrdinals.max().item<int64_t>() >= i64KernelCount) {
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

  int64_t * const p_indexMatrix = indexMatrix.data_ptr<int64_t>();

  {
    auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());
    featureMatrix = torch::empty({ i64Rows, i64Cols }, clOptions);
  }

  RealType * const p_featureMatrix = featureMatrix.data_ptr<RealType>();

  clImageToMatrix.ExtractIndexMatrix(p_indexMatrix, a_i64ImageSize);

  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();

  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();

  //auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());
  std::vector<torch::Tensor> vGrads(4);

  if (bInDataGrad) {
    torch::Tensor inDataGrad = torch::zeros_like(inData);
    RealType * const p_inDataGrad = inDataGrad.data_ptr<RealType>();

    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t g = 0; g < i64Groups; ++g) {
        for (int64_t c = 0; c < i64InChannels; ++c) {
          clImageToMatrix.ExtractMatrix(p_featureMatrix, p_inData + ((i*i64Groups + g)*i64InChannels + c)*i64InChannelSize, p_indexMatrix, a_i64ImageSize);

          for (int64_t j = 0; j < i64OutChannels; ++j) {
            const RealType * const p_thresholds = p_inThresholds + ((g*i64OutChannels + j)*i64InChannels + c)*i64NumDecisionsPerTree;
            const int64_t * const p_ordinals = p_inOrdinals + ((g*i64OutChannels + j)*i64InChannels + c)*i64NumDecisionsPerTree;

            for (int64_t k = 0; k < i64Rows; ++k) {
              const int64_t * const p_indexRow = p_indexMatrix + k*i64Cols;
              const RealType * const p_row = p_featureMatrix + k*i64Cols;

              const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_row, p_thresholds, p_ordinals, i64TreeDepth, 1);

              const KeyType key = std::get<0>(clKeyMarginTuple);
              const RealType signedMargin = std::get<1>(clKeyMarginTuple);
              const KeyType thresholdIndex = std::get<2>(clKeyMarginTuple);
              const int64_t i64FeatureIndex = p_ordinals[thresholdIndex];
              const int64_t i64ImageIndex = p_indexRow[i64FeatureIndex];

              const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));

              const RealType * const p_leafWeights = p_inWeights + (((g*i64OutChannels + j)*i64InChannels + c)*i64NumLeavesPerTree + key)*i64InnerWeightsNum;

              // Is it padding?
              if (i64ImageIndex >= 0) {
                for (int64_t l = 0; l < i64InnerWeightsNum; ++l) {
                  p_inDataGrad[((i*i64Groups + g)*i64InChannels + c)*i64InChannelSize + i64ImageIndex] += sign * p_leafWeights[l] * p_outDataGrad[(((i*i64Groups + g)*i64OutChannels + j)*i64OutDataImageSize + k)*i64InnerWeightsNum + l];
                }
              }
            }
          }
        }
      }
    }

    vGrads[0] = inDataGrad;
  }

  if (bInThresholdsGrad) {
    torch::Tensor inThresholdsGrad = torch::zeros_like(inThresholds);
    RealType * const p_inThresholdsGrad = inThresholdsGrad.data_ptr<RealType>();

    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t g = 0; g < i64Groups; ++g) {
        for (int64_t c = 0; c < i64InChannels; ++c) {
          clImageToMatrix.ExtractMatrix(p_featureMatrix, p_inData + ((i*i64Groups + g)*i64InChannels + c)*i64InChannelSize, p_indexMatrix, a_i64ImageSize);

          for (int64_t j = 0; j < i64OutChannels; ++j) {
            const RealType * const p_thresholds = p_inThresholds + ((g*i64OutChannels + j)*i64InChannels + c)*i64NumDecisionsPerTree;
            const int64_t * const p_ordinals = p_inOrdinals + ((g*i64OutChannels + j)*i64InChannels + c)*i64NumDecisionsPerTree;

            for (int64_t k = 0; k < i64Rows; ++k) {
              const RealType * const p_row = p_featureMatrix + k*i64Cols;

              const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_row, p_thresholds, p_ordinals, i64TreeDepth, 1);

              const KeyType key = std::get<0>(clKeyMarginTuple);
              const RealType signedMargin = std::get<1>(clKeyMarginTuple);
              const KeyType thresholdIndex = std::get<2>(clKeyMarginTuple);

              const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));

              const RealType * const p_leafWeights = p_inWeights + (((g*i64OutChannels + j)*i64InChannels + c)*i64NumLeavesPerTree + key)*i64InnerWeightsNum;

              for (int64_t l = 0; l < i64InnerWeightsNum; ++l) {
                p_inThresholdsGrad[((g*i64OutChannels + j)*i64InChannels + c)*i64NumDecisionsPerTree + thresholdIndex] += -sign * p_leafWeights[l] * p_outDataGrad[(((i*i64Groups + g)*i64OutChannels + j)*i64OutDataImageSize + k)*i64InnerWeightsNum + l];
              }
            }
          }
        }
      }
    }

    vGrads[1] = inThresholdsGrad;
  }

  if (bInWeightsGrad) {
    torch::Tensor inWeightsGrad = torch::zeros_like(inWeights);
    RealType * const p_inWeightsGrad = inWeightsGrad.data_ptr<RealType>();

    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t g = 0; g < i64Groups; ++g) {
        for (int64_t c = 0; c < i64InChannels; ++c) {
          clImageToMatrix.ExtractMatrix(p_featureMatrix, p_inData + ((i*i64Groups + g)*i64InChannels + c)*i64InChannelSize, p_indexMatrix, a_i64ImageSize);

          for (int64_t j = 0; j < i64OutChannels; ++j) {
            const RealType * const p_thresholds = p_inThresholds + ((g*i64OutChannels + j)*i64InChannels + c)*i64NumDecisionsPerTree;
            const int64_t * const p_ordinals = p_inOrdinals + ((g*i64OutChannels + j)*i64InChannels + c)*i64NumDecisionsPerTree;

            for (int64_t k = 0; k < i64Rows; ++k) {
              const RealType * const p_row = p_featureMatrix + k*i64Cols;

              const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_row, p_thresholds, p_ordinals, i64TreeDepth, 1);

              const KeyType key = std::get<0>(clKeyMarginTuple);
              const RealType signedMargin = std::get<1>(clKeyMarginTuple);
              //const KeyType thresholdIndex = std::get<2>(clKeyMarginTuple);

              const RealType margin = std::abs(signedMargin);

              for (int64_t l = 0; l < i64InnerWeightsNum; ++l) {
                p_inWeightsGrad[(((g*i64OutChannels + j)*i64InChannels + c)*i64NumLeavesPerTree + key)*i64InnerWeightsNum + l] += margin * p_outDataGrad[(((i*i64Groups + g)*i64OutChannels + j)*i64OutDataImageSize + k)*i64InnerWeightsNum + l];
              }
            }
          }
        }
      }
    }
    
    vGrads[3] = inWeightsGrad;
  }

  return vGrads;
}

// 1D
template torch::Tensor hingetree_conv_cpu_forward<float, 1, bleak::HingeTreeCommon<float>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_cpu_forward<double, 1, bleak::HingeTreeCommon<double>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_cpu_forward<float, 1, bleak::HingeFernCommon<float>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_cpu_forward<double, 1, bleak::HingeFernCommon<double>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);

template std::vector<torch::Tensor> hingetree_conv_cpu_backward<float, 1, bleak::HingeTreeCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_cpu_backward<double, 1, bleak::HingeTreeCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_cpu_backward<float, 1, bleak::HingeFernCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_cpu_backward<double, 1, bleak::HingeFernCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);

// 2D
template torch::Tensor hingetree_conv_cpu_forward<float, 2, bleak::HingeTreeCommon<float>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_cpu_forward<double, 2, bleak::HingeTreeCommon<double>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_cpu_forward<float, 2, bleak::HingeFernCommon<float>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_cpu_forward<double, 2, bleak::HingeFernCommon<double>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);

template std::vector<torch::Tensor> hingetree_conv_cpu_backward<float, 2, bleak::HingeTreeCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_cpu_backward<double, 2, bleak::HingeTreeCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_cpu_backward<float, 2, bleak::HingeFernCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_cpu_backward<double, 2, bleak::HingeFernCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);


// 3D
template torch::Tensor hingetree_conv_cpu_forward<float, 3, bleak::HingeTreeCommon<float>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_cpu_forward<double, 3, bleak::HingeTreeCommon<double>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_cpu_forward<float, 3, bleak::HingeFernCommon<float>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template torch::Tensor hingetree_conv_cpu_forward<double, 3, bleak::HingeFernCommon<double>>(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);

template std::vector<torch::Tensor> hingetree_conv_cpu_backward<float, 3, bleak::HingeTreeCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_cpu_backward<double, 3, bleak::HingeTreeCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_cpu_backward<float, 3, bleak::HingeFernCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);
template std::vector<torch::Tensor> hingetree_conv_cpu_backward<double, 3, bleak::HingeFernCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);

#ifndef WITH_CUDA
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

#endif // !WITH_CUDA

