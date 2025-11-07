/*-
 * Nathan Lay
 * AI Resource at National Cancer Institute
 * National Institutes of Health
 * October 2023
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

#include "hingetree_error.h"

typedef c10::IntArrayRef IntArrayRef;

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_fusion_gpu_forward(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_fusion_gpu_backward(torch::Tensor inImg, bool inImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad);

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_fusion_cpu_forward(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  typedef typename TreeTraitsType::KeyType KeyType;
  
  if (inImg.dim() < 2 || inVec.dim() != 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2) {
    InvalidArgumentStream("Error: inImg, inVec, inWeights must have at least 2 dimensions. inThresholds and inOrdinals must have 2 dimensions.").raise();
    return torch::Tensor();
  }

  if (inImg.sizes()[0] != inVec.sizes()[0] || inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0]) {
    InvalidArgumentStream("Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0]. inImg.shape[0] must be the same as inVec.shape[0].").raise();
    return torch::Tensor();
  }
  
  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth)) {
    RunTimeErrorStream ss;
    ss << "Error: The tree depth exceeds compile-time constraints or the number of thresholds is incorrect for the tree depth (tree depth = " << i64TreeDepth << ").";
    ss.raise();
    return torch::Tensor();
  }

  const int64_t i64BatchSize = inImg.sizes()[0];
  const int64_t i64ImgChannels = inImg.sizes()[1];
  const int64_t i64VecChannels = inVec.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];

  if (inOrdinals.min().item<int64_t>() < 0 || inOrdinals.max().item<int64_t>() >= i64ImgChannels + i64VecChannels) {
    RunTimeErrorStream ss;
    ss << "Error: An ordinal value is either negative or larger than or equal to the number of image channels + feature vector channels (" << i64ImgChannels + i64VecChannels << ").";
    ss.raise();
    return torch::Tensor();
  }

  const RealType * const p_inImg = inImg.data_ptr<RealType>();
  const RealType * const p_inVec = inVec.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  
  std::vector<IntArrayRef::value_type> vSizes;
  
  vSizes.resize(2);
  vSizes[0] = i64BatchSize; // batch size
  vSizes[1] = inWeights.sizes()[0]; // Number of trees
  
  auto clOptions = torch::TensorOptions().dtype(inImg.dtype()).device(inImg.device());
  
  {
    auto inImgSlice = inImg.sizes().slice(2);
    vSizes.insert(vSizes.end(), inImgSlice.begin(), inImgSlice.end());
  }

  if (inWeights.sizes().size() > 2) {
    auto inWeightsSlice = inWeights.sizes().slice(2);
    vSizes.insert(vSizes.end(), inWeightsSlice.begin(), inWeightsSlice.end());
  }
  
  torch::Tensor outData = torch::empty(IntArrayRef(vSizes.data(), vSizes.size()), clOptions);

  RealType * const p_outData = outData.data_ptr<RealType>();

  int64_t i64InnerDataNum = 1;
  
  {
    auto inImgSlice = inImg.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inImgSlice.begin(), inImgSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }
  
  int64_t i64InnerWeightsNum = 1;
  
  {
    auto inWeightsSlice = inWeights.sizes().slice(2);
    i64InnerWeightsNum = std::accumulate(inWeightsSlice.begin(), inWeightsSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }

  for (int64_t i = 0; i < i64BatchSize; ++i) {
    for (int64_t j = 0; j < i64NumTrees; ++j) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k), p_inVec + (i*i64VecChannels + 0),
          p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64ImgChannels, i64InnerDataNum);
		  
        const KeyType leafKey = std::get<0>(clKeyMarginTuple);
        const RealType margin = std::get<1>(clKeyMarginTuple);
		
        for (int64_t m = 0; m < i64InnerWeightsNum; ++m)
          p_outData[((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum + m] = std::abs(margin) * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m];
      }
    }
  }
  
  return outData;
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_fusion_cpu_backward(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  typedef typename TreeTraitsType::KeyType KeyType;
  
  if (bInOrdinalsGrad) { // Not differentiable, ever!
    RunTimeErrorStream("Error: Gradient on inOrdinals is requested, but inOrdinals never has a gradient.").raise();
    return std::vector<torch::Tensor>();
  }
  
  if (inImg.dim() < 2 || inVec.dim() != 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2 || outDataGrad.dim() < 2) {
    InvalidArgumentStream("Error: inImg, inVec, inWeights must have at least 2 dimensions. inThresholds and inOrdinals must have 2 dimensions.").raise();
    return std::vector<torch::Tensor>();
  }

  if (inImg.sizes()[0] != inVec.sizes()[0] || inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0]) {
    InvalidArgumentStream("Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0]. inImg.shape[0] must be the same as inVec.shape[0].").raise();
    return std::vector<torch::Tensor>();
  }
  
  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth)) {
    RunTimeErrorStream ss;
    ss << "Error: The tree depth exceeds compile-time constraints or the number of thresholds is incorrect for the tree depth (tree depth = " << i64TreeDepth << ").";
    ss.raise();
    return std::vector<torch::Tensor>();
  }
  
  const int64_t i64BatchSize = inImg.sizes()[0];
  const int64_t i64ImgChannels = inImg.sizes()[1];
  const int64_t i64VecChannels = inVec.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];

  if (inOrdinals.min().item<int64_t>() < 0 || inOrdinals.max().item<int64_t>() >= i64ImgChannels + i64VecChannels) {
    RunTimeErrorStream ss;
    ss << "Error: An ordinal value is either negative or larger than or equal to the number of image channels + feature vector channels (" << i64ImgChannels + i64VecChannels << ").";
    ss.raise();
    return std::vector<torch::Tensor>();
  }

  std::vector<IntArrayRef::value_type> vSizes;
  
  vSizes.resize(2);
  vSizes[0] = i64BatchSize; // batch size
  vSizes[1] = inWeights.sizes()[0]; // Number of trees

  int64_t i64InnerDataNum = 1;

  {
    auto inImgSlice = inImg.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inImgSlice.begin(), inImgSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
    vSizes.insert(vSizes.end(), inImgSlice.begin(), inImgSlice.end());
  }
  
  int64_t i64InnerWeightsNum = 1;
  
  {
    auto inWeightsSlice = inWeights.sizes().slice(2);
    i64InnerWeightsNum = std::accumulate(inWeightsSlice.begin(), inWeightsSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
    vSizes.insert(vSizes.end(), inWeightsSlice.begin(), inWeightsSlice.end());
  }

  // Sanity check on outDataGrad
  if (outDataGrad.sizes() != IntArrayRef(vSizes.data(), vSizes.size())) {
    InvalidArgumentStream ss;
    ss << "Error: outDataGrad.shape does not match the expected shape (" << outDataGrad.sizes() << " != " << IntArrayRef(vSizes.data(), vSizes.size()) << ").";
    ss.raise();
    return std::vector<torch::Tensor>();
  }
  
  const RealType * const p_inImg = inImg.data_ptr<RealType>();
  const RealType * const p_inVec = inVec.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();
  
  std::vector<torch::Tensor> vGradTensors(5);

  if (bInImgGrad || bInVecGrad) {
    torch::Tensor inImgGrad = torch::zeros_like(inImg);
    torch::Tensor inVecGrad = torch::zeros_like(inVec);
    RealType * const p_inImgGrad = inImgGrad.data_ptr<RealType>();
    RealType * const p_inVecGrad = inVecGrad.data_ptr<RealType>();

    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k), p_inVec + (i*i64VecChannels + 0),
            p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64ImgChannels, i64InnerDataNum);
          
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
          const KeyType treeIndex = std::get<2>(clKeyMarginTuple);
          
          const int64_t i64InputIndex = p_inOrdinals[j*i64NumDecisionsPerTree + treeIndex];
          const RealType sign = RealType((RealType(0) < margin) - (margin < RealType(0)));

          if (i64InputIndex < i64ImgChannels) {
            for (int64_t m = 0; m < i64InnerWeightsNum; ++m)
              p_inImgGrad[(i*i64ImgChannels + i64InputIndex)*i64InnerDataNum + k] += sign * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
          }
          else {
            for (int64_t m = 0; m < i64InnerWeightsNum; ++m)
              p_inVecGrad[i*i64VecChannels + i64InputIndex - i64ImgChannels] += sign * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
          }
        }
      }
    }

    vGradTensors[0] = inImgGrad;
    vGradTensors[1] = inVecGrad;
  }
  
  if (bInThresholdsGrad) {
    torch::Tensor inThresholdsGrad = torch::zeros_like(inThresholds);
    RealType * const p_inThresholdsGrad = inThresholdsGrad.data_ptr<RealType>();
    
    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k), p_inVec + (i*i64VecChannels + 0),
            p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64ImgChannels, i64InnerDataNum);
  
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
          const KeyType treeIndex = std::get<2>(clKeyMarginTuple);
  
          const RealType sign = RealType((RealType(0) < margin) - (margin < RealType(0)));
  
          for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
            p_inThresholdsGrad[j*i64NumDecisionsPerTree + treeIndex] += -sign * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
          }
        }
      }
    }

    vGradTensors[2] = inThresholdsGrad;
  }
  
  if (bInWeightsGrad) {
    torch::Tensor inWeightsGrad = torch::zeros_like(inWeights);
    RealType * const p_inWeightsGrad = inWeightsGrad.data_ptr<RealType>();
    
    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k), p_inVec + (i*i64VecChannels + 0),
            p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64ImgChannels, i64InnerDataNum);
  
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
  
          for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
            p_inWeightsGrad[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] += std::abs(margin) * p_outDataGrad[((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
          }
        }
      }
    }

    vGradTensors[4] = inWeightsGrad;
  }

  return vGradTensors;
}

#ifndef WITH_CUDA
template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_fusion_gpu_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor) {
  RunTimeErrorStream("Error: GPU support not compiled in.").raise();
  return torch::Tensor();
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_fusion_gpu_backward(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor) {
  RunTimeErrorStream("Error: GPU support not compiled in.").raise();
  return std::vector<torch::Tensor>();
}
#endif // !WITH_CUDA

torch::Tensor hingetree_fusion_forward(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inImg.dtype() != inVec.dtype() || inImg.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inImg.dtype() != inWeights.dtype()) {
    InvalidArgumentStream("Error: inImg, inVec, inThresholds, inWeights are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64.").raise();
    return torch::Tensor();
  }
  
  if (inImg.device() != inVec.device() || inImg.device() != inThresholds.device() || inImg.device() != inOrdinals.device() || inImg.device() != inWeights.device()) {
    InvalidArgumentStream("Error: All tensors are expected to be on the same device.").raise();
    return torch::Tensor();
  }

  if (!inImg.is_contiguous() || !inVec.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous()) {
    InvalidArgumentStream("Error: All tensors are expected to be contiguous.").raise();
    return torch::Tensor();
  }
  
  c10::DeviceGuard clGuard(inImg.device());

  switch (inImg.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_gpu_forward<float, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_fusion_cpu_forward<float, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_gpu_forward<double, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_fusion_cpu_forward<double, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    InvalidArgumentStream("Error: Unsupported data type. Only torch.float32 and torch.float64 are supported.").raise();
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

std::vector<torch::Tensor> hingetree_fusion_backward(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  if (inImg.dtype() != inVec.dtype() || inImg.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inImg.dtype() != inWeights.dtype() || inImg.dtype() != outDataGrad.dtype()) {
    InvalidArgumentStream("Error: inImg, inVec, inThresholds, inWeights, outDataGrad are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64.").raise();
    return std::vector<torch::Tensor>();
  }
  
  if (inImg.device() != inVec.device() || inImg.device() != inThresholds.device() || inImg.device() != inOrdinals.device() || inImg.device() != inWeights.device() || inImg.device() != outDataGrad.device()) {
    InvalidArgumentStream("Error: All tensors are expected to be on the same device.").raise();
    return std::vector<torch::Tensor>();
  }

  if (!inImg.is_contiguous() || !inVec.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !outDataGrad.is_contiguous()) {
    InvalidArgumentStream("Error: All tensors are expected to be contiguous.").raise();
    return std::vector<torch::Tensor>();
  }

  c10::DeviceGuard clGuard(inImg.device());

  switch (inImg.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_gpu_backward<float, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_fusion_cpu_backward<float, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_gpu_backward<double, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_fusion_cpu_backward<double, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  default:
    InvalidArgumentStream("Error: Unsupported data type. Only torch.float32 and torch.float64 are supported.").raise();
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

torch::Tensor hingefern_fusion_forward(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inImg.dtype() != inVec.dtype() || inImg.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inImg.dtype() != inWeights.dtype()) {
    InvalidArgumentStream("Error: inImg, inVec, inThresholds, inWeights are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64.").raise();
    return torch::Tensor();
  }
  
  if (inImg.device() != inVec.device() || inImg.device() != inThresholds.device() || inImg.device() != inOrdinals.device() || inImg.device() != inWeights.device()) {
    InvalidArgumentStream("Error: All tensors are expected to be on the same device.").raise();
    return torch::Tensor();
  }

  if (!inImg.is_contiguous() || !inVec.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous()) {
    InvalidArgumentStream("Error: All tensors are expected to be contiguous.").raise();
    return torch::Tensor();
  }

  c10::DeviceGuard clGuard(inImg.device());

  switch (inImg.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_gpu_forward<float, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_fusion_cpu_forward<float, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_gpu_forward<double, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_fusion_cpu_forward<double, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    InvalidArgumentStream("Error: Unsupported data type. Only torch.float32 and torch.float64 are supported.").raise();
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

std::vector<torch::Tensor> hingefern_fusion_backward(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  if (inImg.dtype() != inVec.dtype() || inImg.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inImg.dtype() != inWeights.dtype() || inImg.dtype() != outDataGrad.dtype()) {
    InvalidArgumentStream("Error: inImg, inVec, inThresholds, inWeights, outDataGrad are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64.").raise();
    return std::vector<torch::Tensor>();
  }
  
  if (inImg.device() != inVec.device() || inImg.device() != inThresholds.device() || inImg.device() != inOrdinals.device() || inImg.device() != inWeights.device() || inImg.device() != outDataGrad.device()) {
    InvalidArgumentStream("Error: All tensors are expected to be on the same device.").raise();
    return std::vector<torch::Tensor>();
  }

  if (!inImg.is_contiguous() || !inVec.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !outDataGrad.is_contiguous()) {
    InvalidArgumentStream("Error: All tensors are expected to be contiguous.").raise();
    return std::vector<torch::Tensor>();
  }

  c10::DeviceGuard clGuard(inImg.device());

  switch (inImg.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_gpu_backward<float, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_fusion_cpu_backward<float, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_gpu_backward<double, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_fusion_cpu_backward<double, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  default:
    InvalidArgumentStream("Error: Unsupported data type. Only torch.float32 and torch.float64 are supported.").raise();
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

