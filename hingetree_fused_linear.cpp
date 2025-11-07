/*-
 * Nathan Lay
 * AI Resource at National Cancer Institute
 * National Institutes of Health
 * March 2022
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
#include "MedianInit.h"
#include "GreedyInit.h"

#include "hingetree_error.h"

typedef c10::IntArrayRef IntArrayRef;

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_fused_linear_gpu_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias);

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_fused_linear_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_fusion_fused_linear_gpu_forward(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias);

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_fusion_fused_linear_gpu_backward(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

#ifndef WITH_CUDA
template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_fused_linear_gpu_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor) {
  RunTimeErrorStream("Error: GPU support not compiled in.").raise();
  return torch::Tensor();
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_fused_linear_gpu_backward(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor) {
  RunTimeErrorStream("Error: GPU support not compiled in.").raise();
  return std::vector<torch::Tensor>();
}

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_fusion_fused_linear_gpu_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor) {
  RunTimeErrorStream("Error: GPU support not compiled in.").raise();
  return torch::Tensor();
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_fusion_fused_linear_gpu_backward(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor) {
  RunTimeErrorStream("Error: GPU support not compiled in.").raise();
  return std::vector<torch::Tensor>();
}
#endif // !WITH_CUDA

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_fused_linear_cpu_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias) {
  typedef typename TreeTraitsType::KeyType KeyType;
  
  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2 || inLinearWeights.dim() != 2 || inLinearBias.dim() != 1) {
    InvalidArgumentStream("Error: inData, inWeights must have at least 2 dimensions. inThresholds, inOrdinals, inLinearWeights must have 2 dimensions. inLinearBias must have 1 dimension.").raise();
    return torch::Tensor();
  }

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0] || inWeights.sizes()[0] != inLinearWeights.sizes()[1] || inLinearWeights.sizes()[0] != inLinearBias.sizes()[0]) {
    InvalidArgumentStream("Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0]. inLinearWeightrs.shape[0] must be the same as inLinearBias.shape[0].").raise();
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

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];
  const int64_t i64OutChannels = inLinearWeights.sizes()[0];

  if (inOrdinals.min().item<int64_t>() < 0 || inOrdinals.max().item<int64_t>() >= i64NumChannels) {
    RunTimeErrorStream("Error: An ordinal value is either negative or larger than or equal to the number of channels.").raise();
    return torch::Tensor();
  }

  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_inLinearWeights = inLinearWeights.data_ptr<RealType>();
  const RealType * const p_inLinearBias = inLinearBias.data_ptr<RealType>();
  
  std::vector<IntArrayRef::value_type> vSizes;
  
  vSizes.resize(2);
  vSizes[0] = inData.sizes()[0]; // batch size
  //vSizes[1] = inWeights.sizes()[0]; // Number of trees
  vSizes[1] = inLinearWeights.sizes()[0]; // Number of linear outputs
  
  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    vSizes.insert(vSizes.end(), inDataSlice.begin(), inDataSlice.end());
  }

  if (inWeights.sizes().size() > 2) {
    auto inWeightsSlice = inWeights.sizes().slice(2);
    vSizes.insert(vSizes.end(), inWeightsSlice.begin(), inWeightsSlice.end());
  }
  
  torch::Tensor outData = torch::empty(IntArrayRef(vSizes.data(), vSizes.size()), clOptions);

  RealType * const p_outData = outData.data_ptr<RealType>();

  int64_t i64InnerDataNum = 1;
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }
  
  int64_t i64InnerWeightsNum = 1;
  
  {
    auto inWeightsSlice = inWeights.sizes().slice(2);
    i64InnerWeightsNum = std::accumulate(inWeightsSlice.begin(), inWeightsSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }

  for (int64_t o = 0; o < i64OutChannels; ++o) {
    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
          p_outData[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m] = p_inLinearBias[o];
        }
      }
    }
  }

  for (int64_t i = 0; i < i64BatchSize; ++i) {
    for (int64_t j = 0; j < i64NumTrees; ++j) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k),
          p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64InnerDataNum);
		  
        const KeyType leafKey = std::get<0>(clKeyMarginTuple);
        const RealType margin = std::abs(std::get<1>(clKeyMarginTuple));
		
        for (int64_t o = 0; o < i64OutChannels; ++o) {
          const RealType scale = margin * p_inLinearWeights[o*i64NumTrees + j];

          for (int64_t m = 0; m < i64InnerWeightsNum; ++m)
            p_outData[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m] += scale * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m];
        }
      }
    }
  }
  
  return outData;
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_fused_linear_cpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad) {
  typedef typename TreeTraitsType::KeyType KeyType;
  
  if (bInOrdinalsGrad) { // Not differentiable, ever!
    RunTimeErrorStream("Error: Gradient on inOrdinals is requested, but inOrdinals never has a gradient.").raise();
    return std::vector<torch::Tensor>();
  }
  
  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2 || inLinearWeights.dim() != 2 || inLinearBias.dim() != 1 || outDataGrad.dim() < 2) {
    InvalidArgumentStream("Error: inData, inWeights must have at least 2 dimensions. inThresholds, inOrdinals, inLinearWeights must have 2 dimensions. inLinearBias must have 1 dimension.").raise();
    return std::vector<torch::Tensor>();
  }

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0] || inWeights.sizes()[0] != inLinearWeights.sizes()[1] || inLinearWeights.sizes()[0] != inLinearBias.sizes()[0]) {
    InvalidArgumentStream("Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0]. inLinearWeightrs.shape[0] must be the same as inLinearBias.shape[0].").raise();
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
  
  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];
  const int64_t i64OutChannels = inLinearWeights.sizes()[0];

  if (inOrdinals.min().item<int64_t>() < 0 || inOrdinals.max().item<int64_t>() >= i64NumChannels) {
    RunTimeErrorStream("Error: An ordinal value is either negative or larger than or equal to the number of channels.").raise();
    return std::vector<torch::Tensor>();
  }

  std::vector<IntArrayRef::value_type> vSizes;
  
  vSizes.resize(2);
  vSizes[0] = inData.sizes()[0]; // batch size
  //vSizes[1] = inWeights.sizes()[0]; // Number of trees
  vSizes[1] = inLinearWeights.sizes()[0]; // Number of linear outputs

  int64_t i64InnerDataNum = 1;

  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
    vSizes.insert(vSizes.end(), inDataSlice.begin(), inDataSlice.end());
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
  
  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_inLinearWeights = inLinearWeights.data_ptr<RealType>();
  //const RealType * const p_inLinearBias = inLinearBias.data_ptr<RealType>();
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();
  
  std::vector<torch::Tensor> vGradTensors(6);

  if (bInDataGrad) {
    torch::Tensor inDataGrad = torch::zeros_like(inData);
    RealType * const p_inDataGrad = inDataGrad.data_ptr<RealType>();

    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k), 
            p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64InnerDataNum);
          
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
          const KeyType treeIndex = std::get<2>(clKeyMarginTuple);
          
          const int64_t i64InputIndex = p_inOrdinals[j*i64NumDecisionsPerTree + treeIndex];
          const RealType sign = RealType((RealType(0) < margin) - (margin < RealType(0)));

          for (int64_t o = 0; o < i64OutChannels; ++o) {
            const RealType scale = sign * p_inLinearWeights[o*i64NumTrees + j];

            for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
              p_inDataGrad[(i*i64NumChannels + i64InputIndex)*i64InnerDataNum + k] += scale * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
          }
        }
      }
    }

    vGradTensors[0] = inDataGrad;
  }
  
  if (bInThresholdsGrad) {
    torch::Tensor inThresholdsGrad = torch::zeros_like(inThresholds);
    RealType * const p_inThresholdsGrad = inThresholdsGrad.data_ptr<RealType>();
    
    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          // p_inData[(i*iNumChannels + l)*iInnerNum + k]
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k), 
            p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64InnerDataNum);
  
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
          const KeyType treeIndex = std::get<2>(clKeyMarginTuple);
  
          const RealType sign = RealType((RealType(0) < margin) - (margin < RealType(0)));
  
          for (int64_t o = 0; o < i64OutChannels; ++o) {
            const RealType scale = -sign * p_inLinearWeights[o*i64NumTrees + j];

            for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
              p_inThresholdsGrad[j*i64NumDecisionsPerTree + treeIndex] += scale * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
          }
        }
      }
    }

    vGradTensors[1] = inThresholdsGrad;
  }
  
  if (bInWeightsGrad) {
    torch::Tensor inWeightsGrad = torch::zeros_like(inWeights);
    RealType * const p_inWeightsGrad = inWeightsGrad.data_ptr<RealType>();
    
    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          // p_inData[(i*iNumChannels + l)*iInnerNum + k]
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k), 
            p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64InnerDataNum);
  
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::abs(std::get<1>(clKeyMarginTuple)); // Signed margin
  
          for (int64_t o = 0; o < i64OutChannels; ++o) {
            const RealType scale = margin * p_inLinearWeights[o*i64NumTrees + j];

            for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
              p_inWeightsGrad[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] += scale * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
          }
        }
      }
    }

    vGradTensors[3] = inWeightsGrad;
  }

  if (bInLinearWeightsGrad) {
    torch::Tensor inLinearWeightsGrad = torch::zeros_like(inLinearWeights);
    RealType * const p_inLinearWeightsGrad = inLinearWeightsGrad.data_ptr<RealType>();

    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          // p_inData[(i*iNumChannels + l)*iInnerNum + k]
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k), 
            p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64InnerDataNum);
  
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::abs(std::get<1>(clKeyMarginTuple)); // Signed margin
  
          for (int64_t o = 0; o < i64OutChannels; ++o) {
            for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
              p_inLinearWeightsGrad[o*i64NumTrees + j] += margin * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
          }
        }
      }
    }

    vGradTensors[4] = inLinearWeightsGrad;
  }

  if (bInLinearBiasGrad) {
    std::vector<IntArrayRef::value_type> vSumOver(vSizes.size()-1);
    vSumOver[0] = 0;
    std::iota(vSumOver.begin()+1, vSumOver.end(), 2);

    torch::Tensor inLinearBiasGrad = outDataGrad.sum(IntArrayRef(vSumOver.data(), vSumOver.size()));

#if 0
    torch::Tensor inLinearBiasGrad = torch::zeros_like(inLinearBias);
    RealType * const p_inLinearBiasGrad = inLinearBiasGrad.data_ptr<RealType>();

    
    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          for (int64_t o = 0; o < i64OutChannels; ++o) {
            for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
              p_inLinearBiasGrad[o] += p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeighytsNum + m]
            }
          }
        }
      }
    }
#endif

    vGradTensors[5] = inLinearBiasGrad;
  }

  return vGradTensors;
}

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_fusion_fused_linear_cpu_forward(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias) {
  typedef typename TreeTraitsType::KeyType KeyType;
  
  if (inImg.dim() < 2 || inVec.dim() != 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2 || inLinearWeights.dim() != 2 || inLinearBias.dim() != 1) {
   InvalidArgumentStream("Error: inImg, inVec, inWeights must have at least 2 dimensions. inThresholds, inOrdinals, inLinearWeights must have 2 dimensions. inLinearBias must have 1 dimension.").raise();
    return torch::Tensor();
  }

  if (inImg.sizes()[0] != inVec.sizes()[0] || inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0] || inWeights.sizes()[0] != inLinearWeights.sizes()[1] || inLinearWeights.sizes()[0] != inLinearBias.sizes()[0]) {
    InvalidArgumentStream("Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0]. inImg.shape[0] must be the same as inVec.shape[0]. inLinearWeights.sha[e0] must be the same as inLinearBias.shape[0].").raise();
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
  const int64_t i64OutChannels = inLinearWeights.sizes()[0];

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
  const RealType * const p_inLinearWeights = inLinearWeights.data_ptr<RealType>();
  const RealType * const p_inLinearBias = inLinearBias.data_ptr<RealType>();
  
  std::vector<IntArrayRef::value_type> vSizes;
  
  vSizes.resize(2);
  vSizes[0] = i64BatchSize; // batch size
  //vSizes[1] = inWeights.sizes()[0]; // Number of trees
  vSizes[1] = inLinearWeights.sizes()[0]; // Number of linear outputs
  
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

  for (int64_t o = 0; o < i64OutChannels; ++o) {
    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
          p_outData[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m] = p_inLinearBias[o];
        }
      }
    }
  }

  for (int64_t i = 0; i < i64BatchSize; ++i) {
    for (int64_t j = 0; j < i64NumTrees; ++j) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k), p_inVec + (i*i64VecChannels + 0),
          p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64ImgChannels, i64InnerDataNum);
		  
        const KeyType leafKey = std::get<0>(clKeyMarginTuple);
        const RealType margin = std::abs(std::get<1>(clKeyMarginTuple));
		
        for (int64_t o = 0; o < i64OutChannels; ++o) {
          const RealType scale = margin * p_inLinearWeights[o*i64NumTrees + j];

          for (int64_t m = 0; m < i64InnerWeightsNum; ++m)
            p_outData[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m] += scale * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m];
        }
      }
    }
  }
  
  return outData;
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_fusion_fused_linear_cpu_backward(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad) {
  typedef typename TreeTraitsType::KeyType KeyType;
  
  if (bInOrdinalsGrad) { // Not differentiable, ever!
    RunTimeErrorStream("Error: Gradient on inOrdinals is requested, but inOrdinals never has a gradient.").raise();
    return std::vector<torch::Tensor>();
  }
  
  if (inImg.dim() < 2 || inVec.dim() != 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2 || inLinearWeights.dim() != 2 || inLinearBias.dim() != 1 || outDataGrad.dim() < 2) {
    InvalidArgumentStream("Error: inImg, inVec, inWeights must have at least 2 dimensions. inThresholds, inOrdinals, inLinearWeights must have 2 dimensions. inLinearBias must have 1 dimension.").raise();
    return std::vector<torch::Tensor>();
  }

  if (inImg.sizes()[0] != inVec.sizes()[0] || inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0] || inWeights.sizes()[0] != inLinearWeights.sizes()[1] || inLinearWeights.sizes()[0] != inLinearBias.sizes()[0]) {
    InvalidArgumentStream("Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0]. inImg.shape[0] must be the same as inVec.shape[0]. inLinearWeights.sha[e0] must be the same as inLinearBias.shape[0].").raise();
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
  const int64_t i64OutChannels = inLinearWeights.sizes()[0];

  if (inOrdinals.min().item<int64_t>() < 0 || inOrdinals.max().item<int64_t>() >= i64ImgChannels + i64VecChannels) {
    RunTimeErrorStream ss;
    ss << "Error: An ordinal value is either negative or larger than or equal to the number of image channels + feature vector channels (" << i64ImgChannels + i64VecChannels << ").";
    ss.raise();
    return std::vector<torch::Tensor>();
  }

  std::vector<IntArrayRef::value_type> vSizes;
  
  vSizes.resize(2);
  vSizes[0] = i64BatchSize; // batch size
  //vSizes[1] = inWeights.sizes()[0]; // Number of trees
  vSizes[1] = inLinearWeights.sizes()[0]; // Number of linear outputs

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
  const RealType * const p_inLinearWeights = inLinearWeights.data_ptr<RealType>();
  //const RealType * const p_inLinearBias = inLinearBias.data_ptr<RealType>();
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();
  
  std::vector<torch::Tensor> vGradTensors(7);

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

          for (int64_t o = 0; o < i64OutChannels; ++o) {
            const RealType scale = sign * p_inLinearWeights[o*i64NumTrees + j];

            if (i64InputIndex < i64ImgChannels) {
              for (int64_t m = 0; m < i64InnerWeightsNum; ++m)
                p_inImgGrad[(i*i64ImgChannels + i64InputIndex)*i64InnerDataNum + k] += scale * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
            else {
              for (int64_t m = 0; m < i64InnerWeightsNum; ++m)
                p_inVecGrad[i*i64VecChannels + i64InputIndex - i64ImgChannels] += scale * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
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
          // p_inData[(i*iNumChannels + l)*iInnerNum + k]
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k), p_inVec + (i*i64VecChannels + 0),
            p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64ImgChannels, i64InnerDataNum);
  
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
          const KeyType treeIndex = std::get<2>(clKeyMarginTuple);
  
          const RealType sign = RealType((RealType(0) < margin) - (margin < RealType(0)));
  
          for (int64_t o = 0; o < i64OutChannels; ++o) {
            const RealType scale = -sign * p_inLinearWeights[o*i64NumTrees + j];

            for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
              p_inThresholdsGrad[j*i64NumDecisionsPerTree + treeIndex] += scale * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
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
          // p_inData[(i*iNumChannels + l)*iInnerNum + k]
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k), p_inVec + (i*i64VecChannels + 0),
            p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64ImgChannels, i64InnerDataNum);
  
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::abs(std::get<1>(clKeyMarginTuple)); // Signed margin
  
          for (int64_t o = 0; o < i64OutChannels; ++o) {
            const RealType scale = margin * p_inLinearWeights[o*i64NumTrees + j];

            for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
              p_inWeightsGrad[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] += scale * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
          }
        }
      }
    }

    vGradTensors[4] = inWeightsGrad;
  }

  if (bInLinearWeightsGrad) {
    torch::Tensor inLinearWeightsGrad = torch::zeros_like(inLinearWeights);
    RealType * const p_inLinearWeightsGrad = inLinearWeightsGrad.data_ptr<RealType>();

    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          // p_inData[(i*iNumChannels + l)*iInnerNum + k]
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k), p_inVec + (i*i64VecChannels + 0),
            p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64ImgChannels, i64InnerDataNum);
  
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::abs(std::get<1>(clKeyMarginTuple)); // Signed margin
  
          for (int64_t o = 0; o < i64OutChannels; ++o) {
            for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
              p_inLinearWeightsGrad[o*i64NumTrees + j] += margin * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
          }
        }
      }
    }

    vGradTensors[5] = inLinearWeightsGrad;
  }

  if (bInLinearBiasGrad) {
    std::vector<IntArrayRef::value_type> vSumOver(vSizes.size()-1);
    vSumOver[0] = 0;
    std::iota(vSumOver.begin()+1, vSumOver.end(), 2);

    torch::Tensor inLinearBiasGrad = outDataGrad.sum(IntArrayRef(vSumOver.data(), vSumOver.size()));

#if 0
    torch::Tensor inLinearBiasGrad = torch::zeros_like(inLinearBias);
    RealType * const p_inLinearBiasGrad = inLinearBiasGrad.data_ptr<RealType>();

    
    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          for (int64_t o = 0; o < i64OutChannels; ++o) {
            for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
              p_inLinearBiasGrad[o] += p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeighytsNum + m]
            }
          }
        }
      }
    }
#endif

    vGradTensors[6] = inLinearBiasGrad;
  }

  return vGradTensors;
}

torch::Tensor hingetree_fused_linear_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype() || inData.dtype() != inLinearWeights.dtype() || inData.dtype() != inLinearBias.dtype()) {
    InvalidArgumentStream("Error: inData, inThresholds, inWeights, inLinearWeights, inLinearBias are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64.").raise();
    return torch::Tensor();
  }
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != inLinearWeights.device() || inData.device() != inLinearBias.device()) {
    InvalidArgumentStream("Error: All tensors are expected to be on the same device.").raise();
    return torch::Tensor();
  }

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !inLinearWeights.is_contiguous() || !inLinearBias.is_contiguous()) {
    InvalidArgumentStream("Error: All tensors are expected to be contiguous.").raise();
    return torch::Tensor();
  }
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_fused_linear_gpu_forward<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
      else
        return hingetree_fused_linear_cpu_forward<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_fused_linear_gpu_forward<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
      else
        return hingetree_fused_linear_cpu_forward<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
    }
    break;
  default:
    InvalidArgumentStream("Error: Unsupported data type. Only torch.float32 and torch.float64 are supported.").raise();
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

std::vector<torch::Tensor> hingetree_fused_linear_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype() || inData.dtype() != inLinearWeights.dtype() || inData.dtype() != inLinearBias.dtype() || inData.dtype() != outDataGrad.dtype()) {
    InvalidArgumentStream("Error: inData, inThresholds, inWeights, inLinearWeights, inLinearBias, outDataGrad are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64.").raise();
    return std::vector<torch::Tensor>();
  }
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != inLinearWeights.device() || inData.device() != inLinearBias.device() || inData.device() != outDataGrad.device()) {
    InvalidArgumentStream("Error: All tensors are expected to be on the same device.").raise();
    return std::vector<torch::Tensor>();
  }

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !inLinearWeights.is_contiguous() || !inLinearBias.is_contiguous() || !outDataGrad.is_contiguous()) {
    InvalidArgumentStream("Error: All tensors are expected to be contiguous.").raise();
    return std::vector<torch::Tensor>();
  }

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_fused_linear_gpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
      else
        return hingetree_fused_linear_cpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_fused_linear_gpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
      else
        return hingetree_fused_linear_cpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
    }
    break;
  default:
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

torch::Tensor hingefern_fused_linear_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype() || inData.dtype() != inLinearWeights.dtype() || inData.dtype() != inLinearBias.dtype()) {
    InvalidArgumentStream("Error: inData, inThresholds, inWeights, inLinearWeights, inLinearBias are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64.").raise();
    return torch::Tensor();
  }
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != inLinearWeights.device() || inData.device() != inLinearBias.device()) {
    InvalidArgumentStream("Error: All tensors are expected to be on the same device.").raise();
    return torch::Tensor();
  }

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !inLinearWeights.is_contiguous() || !inLinearBias.is_contiguous()) {
    InvalidArgumentStream("Error: All tensors are expected to be contiguous.").raise();
    return torch::Tensor();
  }
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_fused_linear_gpu_forward<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
      else
        return hingetree_fused_linear_cpu_forward<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_fused_linear_gpu_forward<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
      else
        return hingetree_fused_linear_cpu_forward<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
    }
    break;
  default:
    InvalidArgumentStream("Error: Unsupported data type. Only torch.float32 and torch.float64 are supported.").raise();
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

std::vector<torch::Tensor> hingefern_fused_linear_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype() || inData.dtype() != inLinearWeights.dtype() || inData.dtype() != inLinearBias.dtype() || inData.dtype() != outDataGrad.dtype()) {
    InvalidArgumentStream("Error: inData, inThresholds, inWeights, inLinearWeights, inLinearBias, outDataGrad are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64.").raise();
    return std::vector<torch::Tensor>();
  }
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != inLinearWeights.device() || inData.device() != inLinearBias.device() || inData.device() != outDataGrad.device()) {
    InvalidArgumentStream("Error: All tensors are expected to be on the same device.").raise();
    return std::vector<torch::Tensor>();
  }

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !inLinearWeights.is_contiguous() || !inLinearBias.is_contiguous() || !outDataGrad.is_contiguous()) {
    InvalidArgumentStream("Error: All tensors are expected to be contiguous.").raise();
    return std::vector<torch::Tensor>();
  }

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_fused_linear_gpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
      else
        return hingetree_fused_linear_cpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_fused_linear_gpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
      else
        return hingetree_fused_linear_cpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
    }
    break;
  default:
    InvalidArgumentStream("Error: Unsupported data type. Only torch.float32 and torch.float64 are supported.").raise();
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

torch::Tensor hingetree_fusion_fused_linear_forward(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias) {
  if (inImg.dtype() != inVec.dtype() || inImg.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inImg.dtype() != inWeights.dtype() || inImg.dtype() != inLinearWeights.dtype() || inImg.dtype() != inLinearBias.dtype()) {
    InvalidArgumentStream("Error: inImg, inVec, inThresholds, inWeights, inLinearWeights, inLinearBias are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64.").raise();
    return torch::Tensor();
  }
  
  if (inImg.device() != inVec.device() || inImg.device() != inThresholds.device() || inImg.device() != inOrdinals.device() || inImg.device() != inWeights.device() || inImg.device() != inLinearWeights.device() || inImg.device() != inLinearBias.device()) {
    InvalidArgumentStream("Error: All tensors are expected to be on the same device.").raise();
    return torch::Tensor();
  }

  if (!inImg.is_contiguous() || !inVec.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !inLinearWeights.is_contiguous() || !inLinearBias.is_contiguous()) {
    InvalidArgumentStream("Error: All tensors are expected to be contiguous.").raise();
    return torch::Tensor();
  }
  
  c10::DeviceGuard clGuard(inImg.device());

  switch (inImg.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_fused_linear_gpu_forward<float, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
      else
        return hingetree_fusion_fused_linear_cpu_forward<float, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_fused_linear_gpu_forward<double, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
      else
        return hingetree_fusion_fused_linear_cpu_forward<double, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
    }
    break;
  default:
    InvalidArgumentStream("Error: Unsupported data type. Only torch.float32 and torch.float64 are supported.").raise();
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

std::vector<torch::Tensor> hingetree_fusion_fused_linear_backward(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad) {
  if (inImg.dtype() != inVec.dtype() || inImg.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inImg.dtype() != inWeights.dtype() || inImg.dtype() != inLinearWeights.dtype() || inImg.dtype() != inLinearBias.dtype() || inImg.dtype() != outDataGrad.dtype()) {
    InvalidArgumentStream("Error: inImg, inVec, inThresholds, inWeights, inLinearWeights, inLinearBias, outDataGrad are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64.").raise();
    return std::vector<torch::Tensor>();
  }
  
  if (inImg.device() != inVec.device() || inImg.device() != inThresholds.device() || inImg.device() != inOrdinals.device() || inImg.device() != inWeights.device() || inImg.device() != inLinearWeights.device() || inImg.device() != inLinearBias.device() || inImg.device() != outDataGrad.device()) {
    InvalidArgumentStream("Error: All tensors are expected to be on the same device.").raise();
    return std::vector<torch::Tensor>();
  }

  if (!inImg.is_contiguous() || !inVec.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !inLinearWeights.is_contiguous() || !inLinearBias.is_contiguous() || !outDataGrad.is_contiguous()) {
    InvalidArgumentStream("Error: All tensors are expected to be contiguous.").raise();
    return std::vector<torch::Tensor>();
  }

  c10::DeviceGuard clGuard(inImg.device());

  switch (inImg.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_fused_linear_gpu_backward<float, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
      else
        return hingetree_fusion_fused_linear_cpu_backward<float, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_fused_linear_gpu_backward<double, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
      else
        return hingetree_fusion_fused_linear_cpu_backward<double, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
    }
    break;
  default:
    InvalidArgumentStream("Error: Unsupported data type. Only torch.float32 and torch.float64 are supported.").raise();
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

torch::Tensor hingefern_fusion_fused_linear_forward(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias) {
  if (inImg.dtype() != inVec.dtype() || inImg.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inImg.dtype() != inWeights.dtype() || inImg.dtype() != inLinearWeights.dtype() || inImg.dtype() != inLinearBias.dtype()) {
    InvalidArgumentStream("Error: inImg, inVec, inThresholds, inWeights, inLinearWeights, inLinearBias are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64.").raise();
    return torch::Tensor();
  }
  
  if (inImg.device() != inVec.device() || inImg.device() != inThresholds.device() || inImg.device() != inOrdinals.device() || inImg.device() != inWeights.device() || inImg.device() != inLinearWeights.device() || inImg.device() != inLinearBias.device()) {
    InvalidArgumentStream("Error: All tensors are expected to be on the same device.").raise();
    return torch::Tensor();
  }

  if (!inImg.is_contiguous() || !inVec.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !inLinearWeights.is_contiguous() || !inLinearBias.is_contiguous()) {
    InvalidArgumentStream("Error: All tensors are expected to be contiguous.").raise();
    return torch::Tensor();
  }
  
  c10::DeviceGuard clGuard(inImg.device());

  switch (inImg.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_fused_linear_gpu_forward<float, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
      else
        return hingetree_fusion_fused_linear_cpu_forward<float, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_fused_linear_gpu_forward<double, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
      else
        return hingetree_fusion_fused_linear_cpu_forward<double, TreeTraitsType>(inImg, inVec, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias);
    }
    break;
  default:
    InvalidArgumentStream("Error: Unsupported data type. Only torch.float32 and torch.float64 are supported.").raise();
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

std::vector<torch::Tensor> hingefern_fusion_fused_linear_backward(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad) {
  if (inImg.dtype() != inVec.dtype() || inImg.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inImg.dtype() != inWeights.dtype() || inImg.dtype() != inLinearWeights.dtype() || inImg.dtype() != inLinearBias.dtype() || inImg.dtype() != outDataGrad.dtype()) {
    InvalidArgumentStream("Error: inImg, inVec, inThresholds, inWeights, inLinearWeights, inLinearBias, outDataGrad are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64.").raise();
    return std::vector<torch::Tensor>();
  }
  
  if (inImg.device() != inVec.device() || inImg.device() != inThresholds.device() || inImg.device() != inOrdinals.device() || inImg.device() != inWeights.device() || inImg.device() != inLinearWeights.device() || inImg.device() != inLinearBias.device() || inImg.device() != outDataGrad.device()) {
    InvalidArgumentStream("Error: All tensors are expected to be on the same device.").raise();
    return std::vector<torch::Tensor>();
  }

  if (!inImg.is_contiguous() || !inVec.is_contiguous() ||  !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !inLinearWeights.is_contiguous() || !inLinearBias.is_contiguous() || !outDataGrad.is_contiguous()) {
    InvalidArgumentStream("Error: All tensors are expected to be contiguous.").raise();
    return std::vector<torch::Tensor>();
  }

  c10::DeviceGuard clGuard(inImg.device());

  switch (inImg.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_fused_linear_gpu_backward<float, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
      else
        return hingetree_fusion_fused_linear_cpu_backward<float, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inImg.is_cuda())
        return hingetree_fusion_fused_linear_gpu_backward<double, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
      else
        return hingetree_fusion_fused_linear_cpu_backward<double, TreeTraitsType>(inImg, bInImgGrad, inVec, bInVecGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
    }
    break;
  default:
    InvalidArgumentStream("Error: Unsupported data type. Only torch.float32 and torch.float64 are supported.").raise();
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

