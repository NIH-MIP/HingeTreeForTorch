/*-
 * Nathan Lay
 * AI Resource at National Cancer Institute
 * National Institutes of Health
 * April 2024
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
#include <unordered_map>
#include <tuple>

#include "torch/extension.h"
// XXX: No longer comes with PyTorch?
//#include "caffe2/core/timer.h"
#include "Timer.h"
#include "HingeTreeCommon.h"
#include "MedianInit.h"
#include "GreedyInit.h"

typedef c10::IntArrayRef IntArrayRef;

namespace {

template<typename RealType>
torch::Tensor unordered_map_to_sparse(const std::unordered_map<int64_t, RealType> &values, IntArrayRef shape) {
  if (shape.size() <= 0)
  	return torch::Tensor();
  
  if (*std::min_element(shape.cbegin(), shape.cend()) <= 0)
  	return torch::Tensor();

  const int64_t i64MaxIndex = std::accumulate(shape.cbegin(), shape.cend(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());

  const auto minMaxKey = std::minmax_element(values.begin(), values.end(),
    [](const auto &a, const auto &b) -> bool {
	  return a.first < b.first;
	});
	
  if (minMaxKey.first->first < 0 || minMaxKey.second->first >= i64MaxIndex) {
	std::cerr << "Error: Invalid linear index encountered." << std::endl;  
    return torch::Tensor();
  }
  
  const int64_t i64Dimension = shape.size();
  const int64_t i64NumIndices = values.size();
  
  auto clOptions = torch::TensorOptions();
  torch::Tensor cooIndices = torch::empty({i64Dimension, i64NumIndices}, clOptions.dtype(torch::kInt64));
  torch::Tensor cooValues = torch::empty({i64NumIndices}, clOptions.dtype(torch::CppTypeToScalarType<RealType>()));
  
  int64_t * const p_i64CooIndices = cooIndices.data_ptr<int64_t>();
  RealType * const p_cooValues = cooValues.data_ptr<RealType>();
  
  int64_t i = 0;
  for (const auto &p : values) {
    int64_t n = p.first;
	
	//std::cout << "n = " << n << std::endl;
	
	for (int64_t d = i64Dimension-1; d > 0; --d) {
      const int64_t q = n / shape[d];
	  const int64_t r = n - q * shape[d];
	  
	  //std::cout << "d: " << d << ": q = " << q << ", r = " << r << ", shape[d] = " << shape[d] << std::endl;
	  
	  p_i64CooIndices[i64NumIndices*d + i] = r;
	  
	  n = q;
	}
	
	//std::cout << "d: 0: " << r = " << n << std::endl;
	
	p_i64CooIndices[i64NumIndices*0 + i] = n;
	p_cooValues[i] = p.second;
	
	++i;
  }
  
  return torch::sparse_coo_tensor(cooIndices, cooValues, shape).coalesce();
}

} // end anonymous namespace

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> sparse_hingetree_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad);

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> sparse_hingetree_fused_linear_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

#ifndef WITH_CUDA
template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> sparse_hingetree_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  return std::vector<torch::Tensor>();
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> sparse_hingetree_fused_linear_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad) {
  return std::vector<torch::Tensor>();
}
#endif // !WITH_CUDA

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> sparse_hingetree_cpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  typedef typename TreeTraitsType::KeyType KeyType;
  if (bInOrdinalsGrad) { // Not differentiable, ever!
    std::cerr << "Error: Gradient on inOrdinals is requested, but inOrdinals never has a gradient." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2 || outDataGrad.dim() < 2) {
    std::cerr << "Error: inData, inWeights must have at least 2 dimensions. inThresholds and inOrdinals must have 2 dimensions." << std::endl;
    return std::vector<torch::Tensor>();
  }

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0]) {
    std::cerr << "Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0]." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth)) {
    std::cerr << "Error: The tree depth exceeds compile-time constraints or the number of thresholds is incorrect for the tree depth (tree depth = " << i64TreeDepth << ")." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];

  if (inOrdinals.min().item<int64_t>() < 0 || inOrdinals.max().item<int64_t>() >= i64NumChannels) {
    std::cerr << "Error: An ordinal value is either negative or larger than or equal to the number of channels." << std::endl;
    return std::vector<torch::Tensor>();
  }

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
  if (outDataGrad.sizes() != IntArrayRef(vSizes.data(), vSizes.size())) {
    std::cerr << "Error: outDataGrad.shape does not match the expected shape (" << outDataGrad.sizes() << " != " << IntArrayRef(vSizes.data(), vSizes.size()) << ")." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();
  
  std::vector<torch::Tensor> vGradTensors(4);
  
  std::unordered_map<int64_t, RealType> valueMap;

  if (bInDataGrad) {
    //torch::Tensor inDataGrad = torch::zeros_like(inData);
    //RealType * const p_inDataGrad = inDataGrad.data_ptr<RealType>();
	
	valueMap.clear();

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
			const int64_t linearIndex = (i*i64NumChannels + i64InputIndex)*i64InnerDataNum + k;
			valueMap[linearIndex] += sign * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            //p_inDataGrad[(i*i64NumChannels + i64InputIndex)*i64InnerDataNum + k] += sign * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
          }
        }
      }
    }

    //vGradTensors[0] = inDataGrad;
	vGradTensors[0] = unordered_map_to_sparse(valueMap, inData.sizes()).to(inData.device());
  }
  
  if (bInThresholdsGrad) {
    //torch::Tensor inThresholdsGrad = torch::zeros_like(inThresholds);
    //RealType * const p_inThresholdsGrad = inThresholdsGrad.data_ptr<RealType>();
	
	valueMap.clear();
    
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
			const int64_t linearIndex = j*i64NumDecisionsPerTree + treeIndex;
			valueMap[linearIndex] += -sign * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            //p_inThresholdsGrad[j*i64NumDecisionsPerTree + treeIndex] += -sign * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
          }
        }
      }
    }

    //vGradTensors[1] = inThresholdsGrad;
	vGradTensors[1] = unordered_map_to_sparse(valueMap, inThresholds.sizes()).to(inThresholds.device());
  }
  
  if (bInWeightsGrad) {
    //torch::Tensor inWeightsGrad = torch::zeros_like(inWeights);
    //RealType * const p_inWeightsGrad = inWeightsGrad.data_ptr<RealType>();

	valueMap.clear();
    
    for (int64_t i = 0; i < i64BatchSize; ++i) {
      for (int64_t j = 0; j < i64NumTrees; ++j) {
        for (int64_t k = 0; k < i64InnerDataNum; ++k) {
          // p_inData[(i*iNumChannels + l)*iInnerNum + k]
          const auto clKeyMarginTuple = TreeTraitsType::ComputeKeyAndSignedMargin(p_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k), 
            p_inThresholds + (j*i64NumDecisionsPerTree + 0), p_inOrdinals + (j*i64NumDecisionsPerTree + 0), i64TreeDepth, i64InnerDataNum);
  
          const KeyType leafKey = std::get<0>(clKeyMarginTuple);
          const RealType margin = std::get<1>(clKeyMarginTuple); // Signed margin
  
          for (int64_t m = 0; m < i64InnerWeightsNum; ++m) {
			const int64_t linearIndex = (j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m;
			valueMap[linearIndex] += std::abs(margin) * p_outDataGrad[((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            //p_inWeightsGrad[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] += std::abs(margin) * p_outDataGrad[((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
          }
        }
      }
    }

    //vGradTensors[3] = inWeightsGrad;
	vGradTensors[3] = unordered_map_to_sparse(valueMap, inWeights.sizes()).to(inWeights.device());
  }

  return vGradTensors;
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> sparse_hingetree_fused_linear_cpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad) {
  typedef typename TreeTraitsType::KeyType KeyType;
  
  if (bInOrdinalsGrad) { // Not differentiable, ever!
    std::cerr << "Error: Gradient on inOrdinals is requested, but inOrdinals never has a gradient." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2 || inLinearWeights.dim() != 2 || inLinearBias.dim() != 1 || outDataGrad.dim() < 2) {
    std::cerr << "Error: inData, inWeights must have at least 2 dimensions. inThresholds, inOrdinals, inLinearWeights must have 2 dimensions. inLinearBias must have 1 dimension." << std::endl;
    return std::vector<torch::Tensor>();
  }

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0] || inWeights.sizes()[0] != inLinearWeights.sizes()[1] || inLinearWeights.sizes()[0] != inLinearBias.sizes()[0]) {
    std::cerr << "Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0]. inLinearWeightrs.shape[0] must be the same as inLinearBias.shape[0]." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth)) {
    std::cerr << "Error: The tree depth exceeds compile-time constraints or the number of thresholds is incorrect for the tree depth (tree depth = " << i64TreeDepth << ")." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];
  const int64_t i64OutChannels = inLinearWeights.sizes()[0];

  if (inOrdinals.min().item<int64_t>() < 0 || inOrdinals.max().item<int64_t>() >= i64NumChannels) {
    std::cerr << "Error: An ordinal value is either negative or larger than or equal to the number of channels." << std::endl;
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
    std::cerr << "Error: outDataGrad.shape does not match the expected shape (" << outDataGrad.sizes() << " != " << IntArrayRef(vSizes.data(), vSizes.size()) << ")." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_inLinearWeights = inLinearWeights.data_ptr<RealType>();
  //const RealType * const p_inLinearBias = inLinearBias.data_ptr<RealType>();
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();
  
  std::unordered_map<int64_t, RealType> valueMap;

  std::vector<torch::Tensor> vGradTensors(6);

  if (bInDataGrad) {
    valueMap.clear();

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
              const int64_t linearIndex = (i*i64NumChannels + i64InputIndex)*i64InnerDataNum + k;
              valueMap[linearIndex] += scale * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
              //p_inDataGrad[(i*i64NumChannels + i64InputIndex)*i64InnerDataNum + k] += scale * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
          }
        }
      }
    }

	vGradTensors[0] = unordered_map_to_sparse(valueMap, inData.sizes()).to(inData.device());
  }
  
  if (bInThresholdsGrad) {
    valueMap.clear();
    
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
              const int64_t linearIndex = j*i64NumDecisionsPerTree + treeIndex;
              valueMap[linearIndex] += scale * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
              //p_inThresholdsGrad[j*i64NumDecisionsPerTree + treeIndex] += scale * p_inWeights[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
          }
        }
      }
    }

	vGradTensors[1] = unordered_map_to_sparse(valueMap, inThresholds.sizes()).to(inThresholds.device());
  }
  
  if (bInWeightsGrad) {
    valueMap.clear();
    
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
              const int64_t linearIndex = (j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m;
              valueMap[linearIndex] += scale * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
              //p_inWeightsGrad[(j*i64NumLeavesPerTree + leafKey)*i64InnerWeightsNum + m] += scale * p_outDataGrad[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + m];
            }
          }
        }
      }
    }

    vGradTensors[3] = unordered_map_to_sparse(valueMap, inWeights.sizes()).to(inWeights.device());
  }

  // NOTE: The below are never sparse

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

std::vector<torch::Tensor> sparse_hingetree_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype() || inData.dtype() != outDataGrad.dtype()) {
    std::cerr << "Error: inData, inThresholds, inWeights, outDataGrad are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != outDataGrad.device()) {
    std::cerr << "Error: All tensors are expected to be on the same device." << std::endl;
    return std::vector<torch::Tensor>();
  }

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !outDataGrad.is_contiguous()) {
    std::cerr << "Error: All tensors are expected to be contiguous." << std::endl;
    return std::vector<torch::Tensor>();
  }

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return sparse_hingetree_gpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return sparse_hingetree_cpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return sparse_hingetree_gpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return sparse_hingetree_cpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  default:
    std::cerr << "Error: Unsupported data type. Only torch.float32 and torch.float64 are supported." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

std::vector<torch::Tensor> sparse_hingefern_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype() || inData.dtype() != outDataGrad.dtype()) {
    std::cerr << "Error: inData, inThresholds, inWeights, outDataGrad are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != outDataGrad.device()) {
    std::cerr << "Error: All tensors are expected to be on the same device." << std::endl;
    return std::vector<torch::Tensor>();
  }

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !outDataGrad.is_contiguous()) {
    std::cerr << "Error: All tensors are expected to be contiguous." << std::endl;
    return std::vector<torch::Tensor>();
  }

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return sparse_hingetree_gpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return sparse_hingetree_cpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return sparse_hingetree_gpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
      else
        return sparse_hingetree_cpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, outDataGrad);
    }
    break;
  default:
    std::cerr << "Error: Unsupported data type. Only torch.float32 and torch.float64 are supported." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

std::vector<torch::Tensor> sparse_hingetree_fused_linear_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype() || inData.dtype() != inLinearWeights.dtype() || inData.dtype() != inLinearBias.dtype() || inData.dtype() != outDataGrad.dtype()) {
    std::cerr << "Error: inData, inThresholds, inWeights, inLinearWeights, inLinearBias, outDataGrad are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != inLinearWeights.device() || inData.device() != inLinearBias.device() || inData.device() != outDataGrad.device()) {
    std::cerr << "Error: All tensors are expected to be on the same device." << std::endl;
    return std::vector<torch::Tensor>();
  }

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !inLinearWeights.is_contiguous() || !inLinearBias.is_contiguous() || !outDataGrad.is_contiguous()) {
    std::cerr << "Error: All tensors are expected to be contiguous." << std::endl;
    return std::vector<torch::Tensor>();
  }

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeTreeCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return sparse_hingetree_fused_linear_gpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
      else
        return sparse_hingetree_fused_linear_cpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeTreeCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return sparse_hingetree_fused_linear_gpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
      else
        return sparse_hingetree_fused_linear_cpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
    }
    break;
  default:
    std::cerr << "Error: Unsupported data type. Only torch.float32 and torch.float64 are supported." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

std::vector<torch::Tensor> sparse_hingefern_fused_linear_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad) {
  if (inData.dtype() != inThresholds.dtype() || torch::kInt64 != inOrdinals.scalar_type() || inData.dtype() != inWeights.dtype() || inData.dtype() != inLinearWeights.dtype() || inData.dtype() != inLinearBias.dtype() || inData.dtype() != outDataGrad.dtype()) {
    std::cerr << "Error: inData, inThresholds, inWeights, inLinearWeights, inLinearBias, outDataGrad are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  if (inData.device() != inThresholds.device() || inData.device() != inOrdinals.device() || inData.device() != inWeights.device() || inData.device() != inLinearWeights.device() || inData.device() != inLinearBias.device() || inData.device() != outDataGrad.device()) {
    std::cerr << "Error: All tensors are expected to be on the same device." << std::endl;
    return std::vector<torch::Tensor>();
  }

  if (!inData.is_contiguous() || !inThresholds.is_contiguous() || !inOrdinals.is_contiguous() || !inWeights.is_contiguous() || !inLinearWeights.is_contiguous() || !inLinearBias.is_contiguous() || !outDataGrad.is_contiguous()) {
    std::cerr << "Error: All tensors are expected to be contiguous." << std::endl;
    return std::vector<torch::Tensor>();
  }

  c10::DeviceGuard clGuard(inData.device());

  switch (inData.scalar_type()) {
  case torch::kFloat32:
    {
      typedef bleak::HingeFernCommon<float> TreeTraitsType;
      
      if (inData.is_cuda())
        return sparse_hingetree_fused_linear_gpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
      else
        return sparse_hingetree_fused_linear_cpu_backward<float, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
    }
    break;
  case torch::kFloat64:
    {
      typedef bleak::HingeFernCommon<double> TreeTraitsType;
      
      if (inData.is_cuda())
        return sparse_hingetree_fused_linear_gpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
      else
        return sparse_hingetree_fused_linear_cpu_backward<double, TreeTraitsType>(inData, bInDataGrad, inThresholds, bInThresholdsGrad, inOrdinals, bInOrdinalsGrad, inWeights, bInWeightsGrad, inLinearWeights, bInLinearWeightsGrad, inLinearBias, bInLinearBiasGrad, outDataGrad);
    }
    break;
  default:
    std::cerr << "Error: Unsupported data type. Only torch.float32 and torch.float64 are supported." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  return std::vector<torch::Tensor>(); // Not reached
}

