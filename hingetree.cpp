/*-
 * Nathan Lay
 * AI Resource at National Cancer Institute
 * National Institutes of Health
 * November 2020
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
// XXX: No longer comes with PyTorch?
//#include "caffe2/core/timer.h"
#include "Timer.h"
#include "HingeTreeCommon.h"
#include "MedianInit.h"
#include "GreedyInit.h"

typedef c10::IntArrayRef IntArrayRef;

// Broken in PyTorch 1.13
//typedef caffe2::Timer TimerType;
typedef bleak::Timer TimerType;

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_gpu_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad);

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_gpu_backward_deterministic(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad);

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_gpu_reachability(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_gpu_leafmap(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_gpu_marginmap(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_cpu_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  typedef typename TreeTraitsType::KeyType KeyType;
  
  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2)
    return torch::Tensor();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0])
    return torch::Tensor();
  
  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth))
    return torch::Tensor();

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];

  if (inOrdinals.min().item<int64_t>() < 0 || inOrdinals.max().item<int64_t>() >= i64NumChannels)
    return torch::Tensor();

  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  
  std::vector<IntArrayRef::value_type> vSizes;
  
  vSizes.resize(2);
  vSizes[0] = inData.sizes()[0]; // batch size
  vSizes[1] = inWeights.sizes()[0]; // Number of trees
  
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

  for (int64_t i = 0; i < i64BatchSize; ++i) {
    for (int64_t j = 0; j < i64NumTrees; ++j) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k),
          p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64InnerDataNum);
		  
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
std::vector<torch::Tensor> hingetree_cpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  typedef typename TreeTraitsType::KeyType KeyType;
  
  if (bInOrdinalsGrad) // Not differentiable, ever!
    return std::vector<torch::Tensor>();
  
  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2 || outDataGrad.dim() < 2)
    return std::vector<torch::Tensor>();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0])
    return std::vector<torch::Tensor>();
  
  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth))
    return std::vector<torch::Tensor>();
  
  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];

  if (inOrdinals.min().item<int64_t>() < 0 || inOrdinals.max().item<int64_t>() >= i64NumChannels)
    return std::vector<torch::Tensor>();

  std::vector<IntArrayRef::value_type> vSizes;
  
  vSizes.resize(2);
  vSizes[0] = inData.sizes()[0]; // batch size
  vSizes[1] = inWeights.sizes()[0]; // Number of trees

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
  if (outDataGrad.sizes() != IntArrayRef(vSizes.data(), vSizes.size()))
    return std::vector<torch::Tensor>();
  
  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();
  
  std::vector<torch::Tensor> vGradTensors(4);

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

          for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
            p_inDataGrad[(i*i64NumChannels + i64InputIndex)*i64InnerDataNum + k] += sign * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
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
  
          for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
            p_inThresholdsGrad[j*i64NumDecisionsPerTree + treeIndex] += -sign * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
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
          const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
  
          for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
            p_inWeightsGrad[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] += std::abs(margin) * p_outDataGrad[((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
          }
        }
      }
    }

    vGradTensors[3] = inWeightsGrad;
  }

  return vGradTensors;
}

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_cpu_leafmap(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  typedef typename TreeTraitsType::KeyType KeyType;
  
  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2)
    return torch::Tensor();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0])
    return torch::Tensor();
  
  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth))
    return torch::Tensor();

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];

  if (inOrdinals.min().item<int64_t>() < 0 || inOrdinals.max().item<int64_t>() >= i64NumChannels)
    return torch::Tensor();

  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  
  std::vector<IntArrayRef::value_type> vSizes;
  
  vSizes.resize(2);
  vSizes[0] = inData.sizes()[0]; // batch size
  vSizes[1] = inWeights.sizes()[0]; // Number of trees
  
  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    vSizes.insert(vSizes.end(), inDataSlice.begin(), inDataSlice.end());
  }

  torch::Tensor outData = torch::empty(IntArrayRef(vSizes.data(), vSizes.size()), clOptions);

  RealType * const p_outData = outData.data_ptr<RealType>();

  int64_t i64InnerDataNum = 1;
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }
  
  for (int64_t i = 0; i < i64BatchSize; ++i) {
    for (int64_t j = 0; j < i64NumTrees; ++j) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k),
          p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64InnerDataNum);
		  
        const KeyType leafKey = std::get<0>(clKeyMarginTuple);
		
        p_outData[(i*i64NumTrees + j)*i64InnerDataNum + k] = RealType(leafKey);
      }
    }
  }
  
  return outData;
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_cpu_marginmap(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  typedef typename TreeTraitsType::KeyType KeyType;
  
  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2)
    return std::vector<torch::Tensor>();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0])
    return std::vector<torch::Tensor>();
  
  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth))
    return std::vector<torch::Tensor>();

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];

  if (inOrdinals.min().item<int64_t>() < 0 || inOrdinals.max().item<int64_t>() >= i64NumChannels)
    return std::vector<torch::Tensor>();

  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  
  std::vector<IntArrayRef::value_type> vSizes;
  
  vSizes.resize(2);
  vSizes[0] = inData.sizes()[0]; // batch size
  vSizes[1] = inWeights.sizes()[0]; // Number of trees
  
  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    vSizes.insert(vSizes.end(), inDataSlice.begin(), inDataSlice.end());
  }

  torch::Tensor outMargins = torch::empty(IntArrayRef(vSizes.data(), vSizes.size()), clOptions);
  torch::Tensor outOrdinals = torch::empty(IntArrayRef(vSizes.data(), vSizes.size()), clOptions.dtype(torch::kInt64));

  RealType * const p_outMargins = outMargins.data_ptr<RealType>();
  int64_t *p_outOrdinals = outOrdinals.data_ptr<int64_t>();

  int64_t i64InnerDataNum = 1;
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }
  
  for (int64_t i = 0; i < i64BatchSize; ++i) {
    for (int64_t j = 0; j < i64NumTrees; ++j) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k),
          p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64InnerDataNum);
		  
        //const KeyType leafKey = std::get<0>(clKeyMarginTuple);
        const RealType margin = std::get<1>(clKeyMarginTuple);
        const KeyType treeIndex = std::get<2>(clKeyMarginTuple);
		
        p_outMargins[(i*i64NumTrees + j)*i64InnerDataNum + k] = margin;
        p_outOrdinals[(i*i64NumTrees + j)*i64InnerDataNum + k] = p_inOrdinals[j*i64NumDecisionsPerTree + treeIndex];
      }
    }
  }
  
  return { outMargins, outOrdinals };
}

#ifndef WITH_CUDA
template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_gpu_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor) { return torch::Tensor(); }

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_gpu_backward(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor) { return std::vector<torch::Tensor>(); }

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_gpu_backward_deterministic(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor) { return std::vector<torch::Tensor>(); }

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_gpu_reachability(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor) { return torch::Tensor(); }

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_gpu_leafmap(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor) { return torch::Tensor(); }

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_gpu_marginmap(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor) { return std::vector<torch::Tensor>(); }
#endif // !WITH_CUDA

template<typename RealType, typename TreeTraitsType>
std::vector<bool> hingetree_cpu_check_thresholds(torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2)
    return std::vector<bool>();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0])
    return std::vector<bool>();

  const int64_t i64NumTrees = inThresholds.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);

  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth))
    return std::vector<bool>();
  
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();

  std::vector<bool> vGood(i64NumTrees);

  for (int64_t j = 0; j < i64NumTrees; ++j)
   vGood[j] = TreeTraitsType::CheckThresholds(p_inThresholds + j*i64NumDecisionsPerTree, p_inOrdinals + j*i64NumDecisionsPerTree, i64TreeDepth);

  return vGood;
}

template<typename RealType, typename TreeTraitsType>
std::vector<bool> hingetree_cpu_fix_thresholds(torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2)
    return std::vector<bool>();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0])
    return std::vector<bool>();

  const int64_t i64NumTrees = inThresholds.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);

  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth))
    return std::vector<bool>();
  
  RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();

  std::vector<bool> vChangesMade(i64NumTrees);

  for (int64_t j = 0; j < i64NumTrees; ++j)
   vChangesMade[j] = TreeTraitsType::FixThresholds(p_inThresholds + j*i64NumDecisionsPerTree, p_inOrdinals + j*i64NumDecisionsPerTree, i64TreeDepth);

  return vChangesMade;
}

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_cpu_reachability(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  typedef typename TreeTraitsType::KeyType KeyType;

  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2)
    return torch::Tensor();

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0])
    return torch::Tensor();

  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth))
    return torch::Tensor();

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];
 
  if (inOrdinals.min().item<int64_t>() < 0 || inOrdinals.max().item<int64_t>() >= i64NumChannels)
    return torch::Tensor();

  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());
  torch::Tensor outCounts = torch::zeros(inWeights.sizes().slice(0,2), clOptions);

  RealType * const p_outCounts = outCounts.data_ptr<RealType>();

  int64_t i64InnerDataNum = 1;
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }
  
  for (int64_t i = 0; i < i64BatchSize; ++i) {
    for (int64_t j = 0; j < i64NumTrees; ++j) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k),
          p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64InnerDataNum);
		  
        const KeyType leafKey = std::get<0>(clKeyMarginTuple);
        p_outCounts[j*i64NumLeavesPerTree + leafKey] += RealType(1);
      }
    }
  }
  
  return outCounts;
}

torch::Tensor hingetree_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype())
    return torch::Tensor();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return torch::Tensor();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return torch::Tensor();
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_forward<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_forward<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_forward<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_forward<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

std::vector<torch::Tensor> hingetree_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype() || inData.dtype() != outDataGrad.dtype())
    return std::vector<torch::Tensor>();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != outDataGrad.device())
    return std::vector<torch::Tensor>();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !outDataGrad.is_contiguous())
    return std::vector<torch::Tensor>();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  default:
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

std::vector<torch::Tensor> hingetree_backward_deterministic(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype() || inData.dtype() != outDataGrad.dtype())
    return std::vector<torch::Tensor>();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != outDataGrad.device())
    return std::vector<torch::Tensor>();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !outDataGrad.is_contiguous())
    return std::vector<torch::Tensor>();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward_deterministic<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward_deterministic<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  default:
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

torch::Tensor hingefern_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype())
    return torch::Tensor();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return torch::Tensor();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return torch::Tensor();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_forward<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_forward<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_forward<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_forward<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

std::vector<torch::Tensor> hingefern_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype() || inData.dtype() != outDataGrad.dtype())
    return std::vector<torch::Tensor>();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != outDataGrad.device())
    return std::vector<torch::Tensor>();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !outDataGrad.is_contiguous())
    return std::vector<torch::Tensor>();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  default:
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

std::vector<torch::Tensor> hingefern_backward_deterministic(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype() || inData.dtype() != outDataGrad.dtype())
    return std::vector<torch::Tensor>();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != outDataGrad.device())
    return std::vector<torch::Tensor>();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !outDataGrad.is_contiguous())
    return std::vector<torch::Tensor>();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward_deterministic<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_backward_deterministic<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return hingetree_cpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  default:
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

std::vector<bool> hingetree_check_thresholds(torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (torch::kInt64 != inOrdinals.scalar_type() || inThresholds.dtype() != inWeights.dtype())
    return std::vector<bool>();

  if (inThresholds.device() != torch::kCPU || inThresholds.device() != inOrdinals.device() || inThresholds.device() != inWeights.device())
    return std::vector<bool>();

  if (!inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return std::vector<bool>();

  switch (inThresholds.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      return hingetree_cpu_check_thresholds<float, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      return hingetree_cpu_check_thresholds<double, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return std::vector<bool>();
  }

  return std::vector<bool>();
}

std::vector<bool> hingefern_check_thresholds(torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (torch::kInt64 != inOrdinals.scalar_type() || inThresholds.dtype() != inWeights.dtype())
    return std::vector<bool>();

  if (inThresholds.device() != torch::kCPU || inThresholds.device() != inOrdinals.device() || inThresholds.device() != inWeights.device())
    return std::vector<bool>();

  if (!inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return std::vector<bool>();

  switch (inThresholds.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      return hingetree_cpu_check_thresholds<float, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      return hingetree_cpu_check_thresholds<double, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return std::vector<bool>();
  }

  return std::vector<bool>();
}

std::vector<bool> hingetree_fix_thresholds(torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (torch::kInt64 != inOrdinals.scalar_type() || inThresholds.dtype() != inWeights.dtype())
    return std::vector<bool>();

  if (inThresholds.device() != torch::kCPU || inThresholds.device() != inOrdinals.device() || inThresholds.device() != inWeights.device())
    return std::vector<bool>();

  if (!inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return std::vector<bool>();

  switch (inThresholds.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      return hingetree_cpu_fix_thresholds<float, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      return hingetree_cpu_fix_thresholds<double, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return std::vector<bool>();
  }

  return std::vector<bool>();
}

std::vector<bool> hingefern_fix_thresholds(torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (torch::kInt64 != inOrdinals.scalar_type() || inThresholds.dtype() != inWeights.dtype())
    return std::vector<bool>();

  if (inThresholds.device() != torch::kCPU || inThresholds.device() != inOrdinals.device() || inThresholds.device() != inWeights.device())
    return std::vector<bool>();

  if (!inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return std::vector<bool>();

  switch (inThresholds.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      return hingetree_cpu_fix_thresholds<float, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      return hingetree_cpu_fix_thresholds<double, TreeTraitsType>(inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return std::vector<bool>();
  }

  return std::vector<bool>();
}

torch::Tensor hingetree_reachability(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype())
    return torch::Tensor();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return torch::Tensor();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return torch::Tensor();
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_reachability<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_reachability<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_reachability<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_reachability<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

torch::Tensor hingetree_leafmap(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype())
    return torch::Tensor();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return torch::Tensor();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return torch::Tensor();
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_leafmap<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_leafmap<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_leafmap<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_leafmap<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

std::vector<torch::Tensor> hingetree_marginmap(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype())
    return std::vector<torch::Tensor>();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return std::vector<torch::Tensor>();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return std::vector<torch::Tensor>();
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_marginmap<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_marginmap<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_marginmap<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_marginmap<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

torch::Tensor hingefern_reachability(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype())
    return torch::Tensor();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return torch::Tensor();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return torch::Tensor();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_reachability<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_reachability<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_reachability<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_reachability<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

torch::Tensor hingefern_leafmap(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype())
    return torch::Tensor();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return torch::Tensor();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return torch::Tensor();
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_leafmap<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_leafmap<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_leafmap<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_leafmap<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

std::vector<torch::Tensor> hingefern_marginmap(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype())
    return std::vector<torch::Tensor>();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return std::vector<torch::Tensor>();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return std::vector<torch::Tensor>();
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_marginmap<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_marginmap<float, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_gpu_marginmap<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
      else
        return hingetree_cpu_marginmap<double, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights);
    }
    break;
  default:
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

// inData is the size of minibatch to test... and a hint of which device to test!
torch::Tensor hingetree_speedtest(torch::Tensor inData, bool bDeterministic) {
  constexpr int64_t i64NumBatches = 100;
  constexpr int64_t i64NumTreeSteps = 10;
  constexpr int64_t i64MaxDepth = 12;
  typedef bleak::HingeTreeCommon<float> TreeTraitsType; // Type doesn't matter... just used to deduce dimensions of tensors

  if (inData.dim() < 2)
    return torch::Tensor();

  torch::Tensor clTimings = torch::zeros({4, i64NumTreeSteps*i64MaxDepth}, torch::TensorOptions().dtype(torch::kFloat32)); // numTrees, depth, fotward timing, backward timing

  float * const p_fNumTrees = clTimings.data_ptr<float>();
  float * const p_fDepths = p_fNumTrees + (i64NumTreeSteps*i64MaxDepth);
  float * const p_fForwardTimings = p_fDepths + (i64NumTreeSteps*i64MaxDepth);
  float * const p_fBackwardTimings = p_fForwardTimings + (i64NumTreeSteps*i64MaxDepth);

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  int64_t t = 0; // Timings index
  for (int64_t s = 0; s < i64NumTreeSteps; ++s) {
    const int64_t i64NumTrees = (((int64_t)1) << s);
    
    for (int64_t i64Depth = 1; i64Depth <= i64MaxDepth; ++i64Depth, ++t) {
      torch::Tensor inThresholds = torch::rand( { i64NumTrees, TreeTraitsType::GetThresholdCount(i64Depth) }, clOptions);
      inThresholds *= 6.0f;
      inThresholds -= 3.0f;

      torch::Tensor inWeights = torch::randn( { i64NumTrees, TreeTraitsType::GetLeafCount(i64Depth) }, clOptions);
      torch::Tensor inOrdinals = torch::randint(0, inData.sizes()[1], { i64NumTrees, TreeTraitsType::GetThresholdCount(i64Depth) }, clOptions.dtype(torch::kInt64));

      torch::Tensor outData;

      p_fNumTrees[t] = (float)i64NumTrees;
      p_fDepths[t] = (float)i64Depth;

      {
        TimerType clForwardTimer;

        for (int64_t i64Batch = 0; i64Batch < i64NumBatches; ++i64Batch)
          outData = hingetree_forward(inData, inThresholds, inOrdinals, inWeights);

        const float fAverage = clForwardTimer.MilliSeconds() / i64NumBatches;
        p_fForwardTimings[t] = fAverage;

        std::cout << "hingetree_forward: numBatches = " << i64NumBatches << ", numTrees = " << i64NumTrees << ", depth = " << i64Depth << ": " << fAverage << " ms per batch." << std::endl;
      }

      torch::Tensor outDataGrad = torch::ones_like(outData);

      if (bDeterministic) {
        TimerType clBackwardTimer;

        for (int64_t i64Batch = 0; i64Batch < i64NumBatches; ++i64Batch)
          hingetree_backward_deterministic(inData, true, inThresholds, true, inOrdinals, false, inWeights, true, outDataGrad);

        const float fAverage = clBackwardTimer.MilliSeconds() / i64NumBatches;
        p_fBackwardTimings[t] = fAverage;

        std::cout << "hingetree_backward_deterministic: numBatches = " << i64NumBatches << ", numTrees = " << i64NumTrees << ", depth = " << i64Depth << ": " << fAverage << " ms per batch." << std::endl;
      }
      else {
        TimerType clBackwardTimer;

        for (int64_t i64Batch = 0; i64Batch < i64NumBatches; ++i64Batch)
          hingetree_backward(inData, true, inThresholds, true, inOrdinals, false, inWeights, true, outDataGrad);

        const float fAverage = clBackwardTimer.MilliSeconds() / i64NumBatches;
        p_fBackwardTimings[t] = fAverage;

        std::cout << "hingetree_backward: numBatches = " << i64NumBatches << ", numTrees = " << i64NumTrees << ", depth = " << i64Depth << ": " << fAverage << " ms per batch." << std::endl;
      }
    }
  }

  return clTimings;
}

torch::Tensor hingefern_speedtest(torch::Tensor inData, bool bDeterministic) {
  constexpr int64_t i64NumBatches = 100;
  constexpr int64_t i64NumTreeSteps = 10;
  constexpr int64_t i64MaxDepth = 12;
  typedef bleak::HingeFernCommon<float> TreeTraitsType; // Type doesn't matter... just used to deduce dimensions of tensors

  if (inData.dim() < 2)
    return torch::Tensor();

  torch::Tensor clTimings = torch::zeros({4, i64NumTreeSteps*i64MaxDepth}, torch::TensorOptions().dtype(torch::kFloat32)); // numTrees, depth, fotward timing, backward timing

  float * const p_fNumTrees = clTimings.data_ptr<float>();
  float * const p_fDepths = p_fNumTrees + (i64NumTreeSteps*i64MaxDepth);
  float * const p_fForwardTimings = p_fDepths + (i64NumTreeSteps*i64MaxDepth);
  float * const p_fBackwardTimings = p_fForwardTimings + (i64NumTreeSteps*i64MaxDepth);

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  int64_t t = 0; // Timings index
  for (int64_t s = 0; s < i64NumTreeSteps; ++s) {
    const int64_t i64NumTrees = (((int64_t)1) << s);
    
    for (int64_t i64Depth = 1; i64Depth <= i64MaxDepth; ++i64Depth, ++t) {
      torch::Tensor inThresholds = torch::rand( { i64NumTrees, TreeTraitsType::GetThresholdCount(i64Depth) }, clOptions);
      inThresholds *= 6.0f;
      inThresholds -= 3.0f;

      torch::Tensor inWeights = torch::randn( { i64NumTrees, TreeTraitsType::GetLeafCount(i64Depth) }, clOptions);
      torch::Tensor inOrdinals = torch::randint(0, inData.sizes()[1], { i64NumTrees, TreeTraitsType::GetThresholdCount(i64Depth) }, clOptions.dtype(torch::kInt64));

      torch::Tensor outData;

      p_fNumTrees[t] = (float)i64NumTrees;
      p_fDepths[t] = (float)i64Depth;

      {
        TimerType clForwardTimer;

        for (int64_t i64Batch = 0; i64Batch < i64NumBatches; ++i64Batch)
          outData = hingefern_forward(inData, inThresholds, inOrdinals, inWeights);

        const float fAverage = clForwardTimer.MilliSeconds() / i64NumBatches;
        p_fForwardTimings[t] = fAverage;

        std::cout << "hingefern_forward: numBatches = " << i64NumBatches << ", numTrees = " << i64NumTrees << ", depth = " << i64Depth << ": " << fAverage << " ms per batch." << std::endl;
      }

      torch::Tensor outDataGrad = torch::ones_like(outData);

      if (bDeterministic) {
        TimerType clBackwardTimer;

        for (int64_t i64Batch = 0; i64Batch < i64NumBatches; ++i64Batch)
          hingefern_backward_deterministic(inData, true, inThresholds, true, inOrdinals, false, inWeights, true, outDataGrad);

        const float fAverage = clBackwardTimer.MilliSeconds() / i64NumBatches;
        p_fBackwardTimings[t] = fAverage;

        std::cout << "hingefern_backward_deterministic: numBatches = " << i64NumBatches << ", numTrees = " << i64NumTrees << ", depth = " << i64Depth << ": " << fAverage << " ms per batch." << std::endl;
      }
      else {
        TimerType clBackwardTimer;

        for (int64_t i64Batch = 0; i64Batch < i64NumBatches; ++i64Batch)
          hingefern_backward(inData, true, inThresholds, true, inOrdinals, false, inWeights, true, outDataGrad);

        const float fAverage = clBackwardTimer.MilliSeconds() / i64NumBatches;
        p_fBackwardTimings[t] = fAverage;

        std::cout << "hingefern_backward: numBatches = " << i64NumBatches << ", numTrees = " << i64NumTrees << ", depth = " << i64Depth << ": " << fAverage << " ms per batch." << std::endl;
      }
    }
  }

  return clTimings;
}

bool hingetree_init_medians(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype())
    return false;
  
  if (inData.device() != torch::kCPU || inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return false;

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return false;
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;

      auto vTrees = FromPyTorch<float, TreeTraitsType>(inThresholds, inOrdinals, inWeights);

      if (vTrees.empty())
        return false;

      if (!InitMedianSplits<float, TreeTraitsType>(vTrees, inData, inWeights))
        return false;

      if (!ToPyTorch<float, TreeTraitsType>(vTrees, inThresholds, inOrdinals, inWeights))
        return false;

      return true;
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;

      auto vTrees = FromPyTorch<double, TreeTraitsType>(inThresholds, inOrdinals, inWeights);

      if (vTrees.empty())
        return false;

      if (!InitMedianSplits<double, TreeTraitsType>(vTrees, inData, inWeights))
        return false;

      if (!ToPyTorch<double, TreeTraitsType>(vTrees, inThresholds, inOrdinals, inWeights))
        return false;
     
      return true;
    }
    break;
  default:
    return false;
  }
  
  return false; // Not reached
}

bool hingefern_init_medians(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype())
    return false;
  
  if (inData.device() != torch::kCPU || inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return false;

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return false;
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;

      auto vTrees = FromPyTorch<float, TreeTraitsType>(inThresholds, inOrdinals, inWeights);

      if (vTrees.empty())
        return false;

      if (!InitMedianSplits<float, TreeTraitsType>(vTrees, inData, inWeights))
        return false;

      if (!ToPyTorch<float, TreeTraitsType>(vTrees, inThresholds, inOrdinals, inWeights))
        return false;

      return true;
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;

      auto vTrees = FromPyTorch<double, TreeTraitsType>(inThresholds, inOrdinals, inWeights);

      if (vTrees.empty())
        return false;

      if (!InitMedianSplits<double, TreeTraitsType>(vTrees, inData, inWeights))
        return false;

      if (!ToPyTorch<double, TreeTraitsType>(vTrees, inThresholds, inOrdinals, inWeights))
        return false;
     
      return true;
    }
    break;
  default:
    return false;
  }
  
  return false; // Not reached
}

bool hingetree_init_greedy(torch::Tensor inData, torch::Tensor inLabels, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype())
    return false;

  if (inLabels.scalar_type() != torch::kInt64 && inLabels.dtype() != inData.dtype()) // NOTE: The latter check is for regression
    return false;
  
  if (inData.device() != torch::kCPU || inData.device() != inLabels.device() || inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return false;

  if (!inData.is_contiguous() || !inLabels.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return false;
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;

      auto vTrees = FromPyTorch<float, TreeTraitsType>(inThresholds, inOrdinals, inWeights);

      if (vTrees.empty())
        return false;

      if (inLabels.scalar_type() == torch::kInt64) {
        if (!InitGreedySplitsClassification<float, TreeTraitsType>(vTrees, inData, inLabels, inWeights))
          return false;
      }
      else {
        return false; // TODO: Implement greedy regression
      }

      if (!ToPyTorch<float, TreeTraitsType>(vTrees, inThresholds, inOrdinals, inWeights))
        return false;

      return true;
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;

      auto vTrees = FromPyTorch<double, TreeTraitsType>(inThresholds, inOrdinals, inWeights);

      if (vTrees.empty())
        return false;

      if (inLabels.scalar_type() == torch::kInt64) {
        if (!InitGreedySplitsClassification<double, TreeTraitsType>(vTrees, inData, inLabels, inWeights))
            return false;
      }
      else {
        return false; // TODO: Implement greedy regression
      }

      if (!ToPyTorch<double, TreeTraitsType>(vTrees, inThresholds, inOrdinals, inWeights))
        return false;
     
      return true;
    }
    break;
  default:
    return false;
  }
  
  return false; // Not reached
}

// Convolution operations below
template<typename RealType, unsigned int Dimension, typename TreeTraitsType>
torch::Tensor hingetree_conv_cpu_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef kernelSize, IntArrayRef, IntArrayRef, IntArrayRef);

template<typename RealType, unsigned int Dimension, typename TreeTraitsType>
torch::Tensor hingetree_conv_gpu_forward(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, IntArrayRef kernelSize, IntArrayRef, IntArrayRef, IntArrayRef);

template<typename RealType, unsigned int Dimension, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_conv_cpu_backward(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);

template<typename RealType, unsigned int Dimension, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_conv_gpu_backward(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, IntArrayRef, IntArrayRef, IntArrayRef, IntArrayRef);

template<unsigned int Dimension>
torch::Tensor hingetree_conv_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, IntArrayRef kernelSize, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype())
    return torch::Tensor();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return torch::Tensor();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return torch::Tensor();
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_conv_gpu_forward<float, Dimension, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation);
      else
        return hingetree_conv_cpu_forward<float, Dimension, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_conv_gpu_forward<double, Dimension, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation);
      else
        return hingetree_conv_cpu_forward<double, Dimension, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation);
    }
    break;
  default:
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

template<unsigned int Dimension>
std::vector<torch::Tensor> hingetree_conv_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad, IntArrayRef kernelSize, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype() || inData.dtype() != outDataGrad.dtype())
    return std::vector<torch::Tensor>();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != outDataGrad.device())
    return std::vector<torch::Tensor>();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !outDataGrad.is_contiguous())
    return std::vector<torch::Tensor>();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_conv_gpu_backward<float, Dimension, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad, kernelSize, stride, padding, dilation);
      else
        return hingetree_conv_cpu_backward<float, Dimension, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad, kernelSize, stride, padding, dilation);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_conv_gpu_backward<double, Dimension, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad, kernelSize, stride, padding, dilation);
      else
        return hingetree_conv_cpu_backward<double, Dimension, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad, kernelSize, stride, padding, dilation);
    }
    break;
  default:
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

template<unsigned int Dimension>
torch::Tensor hingefern_conv_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, IntArrayRef kernelSize, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype())
    return torch::Tensor();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device())
    return torch::Tensor();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous())
    return torch::Tensor();
  
  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_conv_gpu_forward<float, Dimension, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation);
      else
        return hingetree_conv_cpu_forward<float, Dimension, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_conv_gpu_forward<double, Dimension, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation);
      else
        return hingetree_conv_cpu_forward<double, Dimension, TreeTraitsType>(inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation);
    }
    break;
  default:
    return torch::Tensor();
  }
  
  return torch::Tensor(); // Not reached
}

template<unsigned int Dimension>
std::vector<torch::Tensor> hingefern_conv_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad, IntArrayRef kernelSize, IntArrayRef stride, IntArrayRef padding, IntArrayRef dilation) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype() || inData.dtype() != outDataGrad.dtype())
    return std::vector<torch::Tensor>();
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != outDataGrad.device())
    return std::vector<torch::Tensor>();

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !outDataGrad.is_contiguous())
    return std::vector<torch::Tensor>();

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_conv_gpu_backward<float, Dimension, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad, kernelSize, stride, padding, dilation);
      else
        return hingetree_conv_cpu_backward<float, Dimension, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad, kernelSize, stride, padding, dilation);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return hingetree_conv_gpu_backward<double, Dimension, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad, kernelSize, stride, padding, dilation);
      else
        return hingetree_conv_cpu_backward<double, Dimension, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad, kernelSize, stride, padding, dilation);
    }
    break;
  default:
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

torch::Tensor hingetrie_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
std::vector<torch::Tensor> hingetrie_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad); 
bool hingetrie_init_medians(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

torch::Tensor hingetree_fused_linear_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias);
std::vector<torch::Tensor> hingetree_fused_linear_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

torch::Tensor hingefern_fused_linear_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias);
std::vector<torch::Tensor> hingefern_fused_linear_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

torch::Tensor hingetree_fusion_forward(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
std::vector<torch::Tensor> hingetree_fusion_backward(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad);

torch::Tensor hingefern_fusion_forward(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
std::vector<torch::Tensor> hingefern_fusion_backward(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad);

torch::Tensor hingetree_fusion_fused_linear_forward(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias);
std::vector<torch::Tensor> hingetree_fusion_fused_linear_backward(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

torch::Tensor hingefern_fusion_fused_linear_forward(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias);
std::vector<torch::Tensor> hingefern_fusion_fused_linear_backward(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

torch::Tensor contract(torch::Tensor inData, IntArrayRef window, IntArrayRef padding);
torch::Tensor expand(torch::Tensor inData, IntArrayRef padding);

// Sparse backward operations
std::vector<torch::Tensor> sparse_hingetree_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad);

std::vector<torch::Tensor> sparse_hingefern_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad);

std::vector<torch::Tensor> sparse_hingetree_fused_linear_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

std::vector<torch::Tensor> sparse_hingefern_fused_linear_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tree_forward", &hingetree_forward, "Hinge tree forward.");
  m.def("tree_backward", &hingetree_backward, "Hinge tree backward.");
  m.def("tree_backward_deterministic", &hingetree_backward_deterministic, "Deterministic hinge tree backward.");
  m.def("sparse_tree_backward", &sparse_hingetree_backward, "Hinge tree backward with sparse gradients.");
  m.def("tree_check_thresholds", &hingetree_check_thresholds, "Check logical consistency of thresholds and ordinals in trees. Returns true if logically consistent.");
  m.def("tree_fix_thresholds", &hingetree_fix_thresholds, "Fix logical consistency of thresholds and ordinals in trees. Returns true if changes were made.");
  m.def("tree_reachability", &hingetree_reachability, "Compute leaf visit counts. Returns per-leaf visit counts.");
  m.def("tree_leafmap", &hingetree_leafmap, "Hinge tree leaf map.");
  m.def("tree_marginmap", &hingetree_marginmap, "Hinge tree margin map with ordinal.");
  m.def("tree_speedtest", &hingetree_speedtest, "Compute forward/backward timings on a given mini-batch. Returns 2D tensor with rows: numTrees, depth, forward timings, backward timings.");
  m.def("tree_init_medians", &hingetree_init_medians, "Initialize decision thresholds to be medians of input data (CPU only).");
  m.def("tree_init_greedy", &hingetree_init_greedy, "Initialize decision thresholds using random decision tree learning (CPU only).");
  
  m.def("fern_forward", &hingefern_forward, "Hinge fern forward.");
  m.def("fern_backward", &hingefern_backward, "Hinge fern backward.");
  m.def("fern_backward_deterministic", &hingefern_backward_deterministic, "Deterministic hinge fern backward.");
  m.def("sparse_fern_backward", &sparse_hingefern_backward, "Hinge fern backward with sparse gradients.");
  m.def("fern_check_thresholds", &hingefern_check_thresholds, "Check logical consistency of thresholds and ordinals in ferns. Returns true if logically consistent.");
  m.def("fern_fix_thresholds", &hingefern_fix_thresholds, "Fix logical consistency of thresholds and ordinals in ferns. Returns true if changes were made.");
  m.def("fern_reachability", &hingefern_reachability, "Compute leaf visit counts. Returns per-leaf visit counts.");
  m.def("fern_leafmap", &hingefern_leafmap, "Hinge fern leaf map.");
  m.def("fern_marginmap", &hingefern_marginmap, "Hinge fern margin map with ordinal.");
  m.def("fern_speedtest", &hingefern_speedtest, "Compute forward/backward timings on a given mini-batch. Returns 2D tensor with rows: numTrees, depth, forward timings, backward timings.");
  m.def("fern_init_medians", &hingefern_init_medians, "Initialize decision thresholds to be medians of input data (CPU only).");

  // Convolution operations
  m.def("tree_conv1d_forward", &hingetree_conv_forward<1>, "Hinge tree conv1d forward.");
  m.def("tree_conv2d_forward", &hingetree_conv_forward<2>, "Hinge tree conv2d forward.");
  m.def("tree_conv3d_forward", &hingetree_conv_forward<3>, "Hinge tree conv3d forward.");

  m.def("tree_conv1d_backward", &hingetree_conv_backward<1>, "Hinge tree conv1d backward.");
  m.def("tree_conv2d_backward", &hingetree_conv_backward<2>, "Hinge tree conv2d backward.");
  m.def("tree_conv3d_backward", &hingetree_conv_backward<3>, "Hinge tree conv3d backward.");

  m.def("fern_conv1d_forward", &hingefern_conv_forward<1>, "Hinge fern conv1d forward.");
  m.def("fern_conv2d_forward", &hingefern_conv_forward<2>, "Hinge fern conv2d forward.");
  m.def("fern_conv3d_forward", &hingefern_conv_forward<3>, "Hinge fern conv3d forward.");

  m.def("fern_conv1d_backward", &hingefern_conv_backward<1>, "Hinge fern conv1d backward.");
  m.def("fern_conv2d_backward", &hingefern_conv_backward<2>, "Hinge fern conv2d backward.");
  m.def("fern_conv3d_backward", &hingefern_conv_backward<3>, "Hinge fern conv3d backward.");

  m.def("trie_forward", &hingetrie_forward, "Hinge trie forward.");
  m.def("trie_backward", &hingetrie_backward, "Hinge trie backward.");
  m.def("trie_init_medians", &hingetrie_init_medians, "Initialize decision thresholds to be medians of input data (CPU only).");

  m.def("tree_fused_linear_forward", &hingetree_fused_linear_forward, "Hinge tree + linear fused forward.");
  m.def("tree_fused_linear_backward", &hingetree_fused_linear_backward, "Hinge tree + linear fused backward.");
  m.def("sparse_tree_fused_linear_backward", &sparse_hingetree_fused_linear_backward, "Sparse hinge tree + linear fused backward.");

  m.def("fern_fused_linear_forward", &hingefern_fused_linear_forward, "Hinge fern + linear fused forward.");
  m.def("fern_fused_linear_backward", &hingefern_fused_linear_backward, "Hinge fern + linear fused backward.");
  m.def("sparse_fern_fused_linear_backward", &sparse_hingefern_fused_linear_backward, "Sparse hinge fern + linear fused backward.");

  m.def("tree_fusion_forward", &hingetree_fusion_forward, "Hinge tree image + feature vector fusion.");
  m.def("tree_fusion_backward", &hingetree_fusion_backward, "Hinge tree image + feature vector fusion.");

  m.def("fern_fusion_forward", &hingefern_fusion_forward, "Hinge fern image + feature vector fusion.");
  m.def("fern_fusion_backward", &hingefern_fusion_backward, "Hinge fern image + feature vector fusion.");

  m.def("tree_fusion_fused_linear_forward", &hingetree_fusion_fused_linear_forward, "Hinge tree + feature vector fusion + linear fused forward.");
  m.def("tree_fusion_fused_linear_backward", &hingetree_fusion_fused_linear_backward, "Hinge tree + feature vector fusion + linear fused backward.");

  m.def("fern_fusion_fused_linear_forward", &hingefern_fusion_fused_linear_forward, "Hinge fern + feature vector fusion + linear fused forward.");
  m.def("fern_fusion_fused_linear_backward", &hingefern_fusion_fused_linear_backward, "Hinge fern + feature vector fusion + linear fused backward.");

  m.def("contract", &contract, "Collapse batched 2D or 3D image to 2D or 3D images of patches.");
  m.def("expand", &expand, "Expand batched 2D or 3D images of patches to 2D or 3D images.");
}

