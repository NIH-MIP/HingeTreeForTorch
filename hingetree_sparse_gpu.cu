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

#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <type_traits>

#include "torch/extension.h"
#include "HingeTreeCommon.cuh"

#include "hingetree_error.h"

#include <cuda.h>

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

template<typename RealType>
class IndexMap {
public:
  using KeyType = int64_t;
  using ValueType = RealType;
  
  __host__ __device__ constexpr static KeyType UnoccupiedKey() { return KeyType((((uint64_t)1) << 63) - 1); }

  // d_keys assumed to be initialized to UnoccupiedKey()
  __host__ IndexMap(KeyType *d_keys, ValueType *d_values, size_t length, bool *d_fail)
  : m_d_keys(d_keys), m_d_values(d_values), m_length(length), m_d_fail(d_fail) { }

  __device__ ValueType & operator[](const KeyType &key) {
    if (key == UnoccupiedKey()) {
      *m_d_fail = true;
      return m_d_values[0];
    }

    constexpr unsigned long long unoccupied = UnoccupiedKey();
    const unsigned long long key_as_ull = (unsigned long long)key;
    const size_t hash = (size_t)key_as_ull;
    
    for (size_t i = 0; i < m_length; ++i) {
        const size_t index = ((hash + i) % m_length);
        unsigned long long * const d_address = ((unsigned long long *)m_d_keys) + index;
        const unsigned long long old = atomicCAS(d_address, unoccupied, key_as_ull);
        if (old == unoccupied || old == key_as_ull)
            return m_d_values[index];
    }
    
    *m_d_fail = true;
    
    return m_d_values[0];
  }

private:
   KeyType * const m_d_keys;
   ValueType * const m_d_values;
   const size_t m_length;
   bool * const m_d_fail;
};

template class IndexMap<float>;
template class IndexMap<double>;

static_assert(std::is_trivially_copyable_v<IndexMap<float>>);
static_assert(std::is_trivially_copyable_v<IndexMap<double>>);

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardThresholdsKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, IndexMap<RealType> inThresholdsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i64OuterNum && j < i64NumTrees && k < i64InnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;
    //RealType * const d_thresholdsGradient = d_inThresholdsGradient + j*i64ThresholdStride;
    const int64_t thresholdsOffset = j*i64ThresholdStride;

    const RealType * const d_row = d_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, i64TreeDepth, i64InnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const KeyType thresholdIndex = keyMarginTuple.thresholdIndex;

    const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));

    const RealType * const d_leafWeights = d_inWeights + (j*i64WeightsStride + key)*i64InnerWeightsNum;
    const RealType * const d_outGradient = d_outDataGradient + ((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum;

    RealType tmpSum = RealType(0);
    for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
      tmpSum += d_leafWeights[l] * d_outGradient[l];

    tmpSum *= -sign;

    atomicAdd(&inThresholdsGradient[thresholdsOffset + thresholdIndex], tmpSum); // Do this just once
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardWeightsKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, /*const RealType *d_inWeights,*/
    const RealType *d_outDataGradient, IndexMap<RealType> inWeightsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i64OuterNum && j < i64NumTrees && k < i64InnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

    const RealType * const d_row = d_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, i64TreeDepth, i64InnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const RealType margin = std::abs(signedMargin);

    const RealType * const d_outGradient = d_outDataGradient + ((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum;
    const int64_t leafWeightsOffset = (j*i64WeightsStride + key)*i64InnerWeightsNum;

    for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
      atomicAdd(&inWeightsGradient[leafWeightsOffset + l], margin * d_outGradient[l]); // Really bad!
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardDataKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, IndexMap<RealType> inDataGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i64OuterNum && j < i64NumTrees && k < i64InnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

    const RealType * const d_row = d_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, i64TreeDepth, i64InnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const KeyType thresholdIndex = keyMarginTuple.thresholdIndex;
    const int64_t i64InputIndex = d_ordinals[thresholdIndex];

    const RealType * const d_leafWeights = d_inWeights + (j*i64WeightsStride + key)*i64InnerWeightsNum;
    const RealType * const d_outGradient = d_outDataGradient + ((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum;

    const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));
    RealType tmpSum = RealType(0);

    for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
      tmpSum += d_leafWeights[l] * d_outGradient[l];

    tmpSum *= sign;

    atomicAdd(&inDataGradient[((i*i64NumChannels + i64InputIndex)*i64InnerDataNum + k)], tmpSum); // Do this just once
  }
}


template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void FusedBackwardThresholdsKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, const RealType *d_inLinearWeights,
    const RealType *d_outDataGradient, IndexMap<RealType> inThresholdsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum, int64_t i64OutChannels) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i64OuterNum && j < i64NumTrees && k < i64InnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;
    //RealType * const d_thresholdsGradient = d_inThresholdsGradient + j*i64ThresholdStride;
    const int64_t thresholdsOffset = j*i64ThresholdStride;

    const RealType * const d_row = d_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, i64TreeDepth, i64InnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const KeyType thresholdIndex = keyMarginTuple.thresholdIndex;

    const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));

    const RealType * const d_leafWeights = d_inWeights + (j*i64WeightsStride + key)*i64InnerWeightsNum;

    RealType tmpSum = RealType(0);
    for (int64_t o = 0; o < i64OutChannels; ++o) {
      const RealType * const d_outGradient = d_outDataGradient + ((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum;
      RealType tmpSum2 = RealType(0);

      for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
        tmpSum2 += d_leafWeights[l] * d_outGradient[l];

      tmpSum += tmpSum2 * d_inLinearWeights[o*i64NumTrees + j];
    }

    tmpSum *= -sign;

    atomicAdd(&inThresholdsGradient[thresholdsOffset + thresholdIndex], tmpSum);
    //atomicAdd(d_thresholdsGradient + thresholdIndex, tmpSum); // Do this just once
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void FusedBackwardWeightsKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inLinearWeights,
    const RealType *d_outDataGradient, IndexMap<RealType> inWeightsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum, int64_t i64OutChannels) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i64OuterNum && j < i64NumTrees && k < i64InnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

    const RealType * const d_row = d_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, i64TreeDepth, i64InnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const RealType margin = std::abs(signedMargin);

    //RealType * const d_leafWeightsGradient = d_inWeightsGradient + (j*i64WeightsStride + key)*i64InnerWeightsNum;
    const int64_t leafWeightsOffset = (j*i64WeightsStride + key)*i64InnerWeightsNum;

    for (int64_t l = 0; l < i64InnerWeightsNum; ++l) {
      RealType tmpSum = RealType(0);

      for (int64_t o = 0; o < i64OutChannels; ++o) {
        const RealType * const d_outGradient = d_outDataGradient + ((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum;
        tmpSum += d_inLinearWeights[o*i64NumTrees + j] * d_outGradient[l];
      }

      tmpSum *= margin;

      atomicAdd(&inWeightsGradient[leafWeightsOffset + l], tmpSum);
      //atomicAdd(d_leafWeightsGradient + l, tmpSum); // Really bad!
    }
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void FusedBackwardDataKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, const RealType *d_inLinearWeights,
    const RealType *d_outDataGradient, IndexMap<RealType> inDataGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum, int64_t i64OutChannels) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i64OuterNum && j < i64NumTrees && k < i64InnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

    const RealType * const d_row = d_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, i64TreeDepth, i64InnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const KeyType thresholdIndex = keyMarginTuple.thresholdIndex;
    const int64_t i64InputIndex = d_ordinals[thresholdIndex];

    const RealType * const d_leafWeights = d_inWeights + (j*i64WeightsStride + key)*i64InnerWeightsNum;

    const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));
    RealType tmpSum = RealType(0);

    for (int64_t o = 0; o < i64OutChannels; ++o) {
      const RealType * const d_outGradient = d_outDataGradient + ((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum;

      RealType tmpSum2 = RealType(0);

      for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
        tmpSum2 += d_leafWeights[l] * d_outGradient[l];

      tmpSum += d_inLinearWeights[o*i64NumTrees + j] * tmpSum2;
    }

    tmpSum *= sign;

    atomicAdd(&inDataGradient[(i*i64NumChannels + i64InputIndex)*i64InnerDataNum + k], tmpSum);
    //atomicAdd(d_inDataGradient + ((i*i64NumChannels + i64InputIndex)*i64InnerDataNum + k), tmpSum); // Do this just once
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void FusedBackwardLinearWeightsKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights,
    const RealType *d_outDataGradient, RealType *d_inLinearWeightsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum, int64_t i64OutChannels) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i64OuterNum && j < i64NumTrees && k < i64InnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

    const RealType * const d_row = d_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, i64TreeDepth, i64InnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType margin = std::abs(keyMarginTuple.signedMargin);

    const RealType * const d_leafWeights = d_inWeights + (j*i64WeightsStride + key)*i64InnerWeightsNum;


    for (int64_t o = 0; o < i64OutChannels; ++o) {
      const RealType * const d_outGradient = d_outDataGradient + ((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum;

      RealType tmpSum = RealType(0);

      for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
        tmpSum += d_leafWeights[l] * d_outGradient[l];

      tmpSum *= margin;

      atomicAdd(d_inLinearWeightsGradient + (o*i64NumTrees + j), tmpSum); // This is bad!
    }
  }
}

__global__ void ConvertLinearToCoo(const int64_t *d_i64Linear, int64_t *d_i64Coo, const int64_t *d_i64Shape, int64_t i64Dimension, int64_t i64NumIndices) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i64NumIndices) {
    int64_t n = d_i64Linear[i];

    for (int64_t d = i64Dimension-1; d > 0; --d) {
      const int64_t q = n / d_i64Shape[d];
      const int64_t r = n - q * d_i64Shape[d];

      d_i64Coo[i64NumIndices*d + i] = r;

      n = q;
    }

    d_i64Coo[i64NumIndices*0 + i] = n;
  }
}

torch::Tensor make_sparse(torch::Tensor linearIndices, torch::Tensor values, IntArrayRef shape) {
  if (linearIndices.scalar_type() != torch::kInt64 || linearIndices.dim() != 1 || linearIndices.device() != values.device() || linearIndices.sizes() != values.sizes())
    return torch::Tensor();

  const int64_t i64Dimension = shape.size();

  { 
    auto mask = (linearIndices != IndexMap<float>::UnoccupiedKey());
    values = values.masked_select(mask);
    linearIndices = linearIndices.masked_select(mask);
  }

  const int64_t i64MaxIndex = std::accumulate(shape.begin(), shape.end(), (int64_t)1, std::multiplies<int64_t>());

  if (linearIndices.min().to(torch::kCPU).item<int64_t>() < 0 || linearIndices.max().to(torch::kCPU).item<int64_t>() >= i64MaxIndex) {
    OutOfRangeStream("Error: Invalid linear index encountered during conversion to sparse.").raise();
    return torch::Tensor();
  }

  const int64_t i64NumIndices = linearIndices.sizes()[0];

  auto clOptions = torch::TensorOptions().dtype(torch::kInt64);

  torch::Tensor shapeAsTensor = torch::empty({ i64Dimension }, clOptions);
  std::copy(shape.begin(), shape.end(), shapeAsTensor.data_ptr<int64_t>());
  shapeAsTensor = shapeAsTensor.to(linearIndices.device());

  torch::Tensor cooIndices = torch::empty({ i64Dimension, i64NumIndices }, clOptions.device(linearIndices.device()));

  constexpr unsigned int threadsPerBlock = 1024;
  const unsigned int numBlocks = (unsigned int)((i64NumIndices + threadsPerBlock-1)/threadsPerBlock);

  ConvertLinearToCoo<<<numBlocks, threadsPerBlock>>>(linearIndices.data_ptr<int64_t>(), cooIndices.data_ptr<int64_t>(), shapeAsTensor.data_ptr<int64_t>(), i64Dimension, i64NumIndices);

  return torch::sparse_coo_tensor(cooIndices, values, shape).coalesce();
}

} // end anonymous namespace

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> sparse_hingetree_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

  if (bInOrdinalsGrad) { // Not differentiable, ever!
    RunTimeErrorStream("Error: Gradient on inOrdinals is requested, but inOrdinals never has a gradient.").raise();
    return std::vector<torch::Tensor>();
  }
  
  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2) {
    InvalidArgumentStream("Error: inData, inWeights must have at least 2 dimensions. inThresholds and inOrdinals must have 2 dimensions.").raise();
    return std::vector<torch::Tensor>();
  }

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0]) {
    InvalidArgumentStream("Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0].").raise();
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

  if (inOrdinals.min().to(torch::kCPU).item<int64_t>() < 0 || inOrdinals.max().to(torch::kCPU).item<int64_t>() >= i64NumChannels) {
    RunTimeErrorStream("Error: An ordinal value is either negative or larger than or equal to the number of channels.").raise();
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
    InvalidArgumentStream ss;
    ss << "Error: outDataGrad.shape does not match the expected shape (" << outDataGrad.sizes() << " != " << IntArrayRef(vSizes.data(), vSizes.size()) << ").";
    ss.raise();
    return std::vector<torch::Tensor>();
  }

  const int64_t i64DataSize = std::min((int64_t)inData.numel(), i64BatchSize*i64NumTrees*i64InnerDataNum);
  const int64_t i64WeightsSize = std::min((int64_t)inWeights.numel(), i64BatchSize*i64NumTrees*i64InnerDataNum*i64InnerWeightsNum);
  const int64_t i64ThresholdsSize = std::min((int64_t)inThresholds.numel(), i64BatchSize*i64NumTrees*i64InnerDataNum);

  const int64_t i64WorkSize = std::max(std::max(i64DataSize, i64WeightsSize), i64ThresholdsSize);

  auto clOptions = torch::TensorOptions().device(inData.device());
  torch::Tensor indices = torch::empty({ i64WorkSize }, clOptions.dtype(torch::kInt64));
  torch::Tensor values = torch::empty({ i64WorkSize }, clOptions.dtype(inData.dtype()));
  torch::Tensor fail = torch::scalar_tensor(false, clOptions.dtype(torch::kBool));

  IndexMap<RealType> gradientMap(indices.data_ptr<int64_t>(), values.data_ptr<RealType>(), i64WorkSize, fail.data_ptr<bool>());

  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();

  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1)/threadsPerBlock.y, (i64NumTrees + threadsPerBlock.z-1)/threadsPerBlock.z);

  std::vector<torch::Tensor> vGradTensors(4);

  if (bInDataGrad) {
    values.zero_();
    indices.fill_(gradientMap.UnoccupiedKey());

    BackwardDataKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_outDataGrad, gradientMap, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum);

    if (fail.to(torch::kCPU).item<bool>()) {
      OverflowErrorStream("Error: IndexMap overflowed!").raise();
      return std::vector<torch::Tensor>();
    }

    vGradTensors[0] = make_sparse(indices, values, inData.sizes());
  }
  
  if (bInThresholdsGrad) {
    values.zero_();
    indices.fill_(gradientMap.UnoccupiedKey());

    BackwardThresholdsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_outDataGrad, gradientMap, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum);

    if (fail.to(torch::kCPU).item<bool>()) {
      OverflowErrorStream("Error: IndexMap overflowed!").raise();
      return std::vector<torch::Tensor>();
    }

    vGradTensors[1] = make_sparse(indices, values, inThresholds.sizes());
  }
  
  if (bInWeightsGrad) {
    values.zero_();
    indices.fill_(gradientMap.UnoccupiedKey());

    BackwardWeightsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_outDataGrad, gradientMap, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum);

    if (fail.to(torch::kCPU).item<bool>()) {
      OverflowErrorStream("Error: IndexMap overflowed!").raise();
      return std::vector<torch::Tensor>();
    }

    vGradTensors[3] = make_sparse(indices, values, inWeights.sizes());
  }

  return vGradTensors;
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> sparse_hingetree_fused_linear_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

  if (bInOrdinalsGrad) { // Not differentiable, ever!
    RunTimeErrorStream("Error: Gradient on inOrdinals is requested, but inOrdinals never has a gradient.").raise();
    return std::vector<torch::Tensor>();
  }
  
  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2 || inLinearWeights.dim() != 2 || inLinearBias.dim() != 1 || outDataGrad.dim() < 2) {
    InvalidArgumentStream("Error: inData, inWeights must have at least 2 dimensions. inThresholds, inLinearWeights, inOrdinals must have 2 dimensions. inLinearBias must have 1 dimension.").raise();
    return std::vector<torch::Tensor>();
  }

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0] || inWeights.sizes()[0] != inLinearWeights.sizes()[1] || inLinearWeights.sizes()[0] != inLinearBias.sizes()[0]) {
    InvalidArgumentStream("Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0].").raise();
    return std::vector<torch::Tensor>();
  }  

  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  const int64_t i64OutChannels = inLinearWeights.sizes()[0];
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth)) {
    RunTimeErrorStream ss;
    ss << "Error: The tree depth exceeds compile-time constraints or the number of thresholds is incorrect for the tree depth (tree depth = " << i64TreeDepth << ").";
    ss.raise();
    return std::vector<torch::Tensor>();
  }
  
  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];

  if (inOrdinals.min().to(torch::kCPU).item<int64_t>() < 0 || inOrdinals.max().to(torch::kCPU).item<int64_t>() >= i64NumChannels) {
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

  const int64_t i64DataSize = std::min((int64_t)inData.numel(), i64BatchSize*i64NumTrees*i64InnerDataNum);
  const int64_t i64WeightsSize = std::min((int64_t)inWeights.numel(), i64BatchSize*i64NumTrees*i64InnerDataNum*i64InnerWeightsNum);
  const int64_t i64ThresholdsSize = std::min((int64_t)inThresholds.numel(), i64BatchSize*i64NumTrees*i64InnerDataNum);

  const int64_t i64WorkSize = std::max(std::max(i64DataSize, i64WeightsSize), i64ThresholdsSize);

  auto clOptions = torch::TensorOptions().device(inData.device());
  torch::Tensor indices = torch::empty({ i64WorkSize }, clOptions.dtype(torch::kInt64));
  torch::Tensor values = torch::empty({ i64WorkSize }, clOptions.dtype(inData.dtype()));
  torch::Tensor fail = torch::scalar_tensor(false, clOptions.dtype(torch::kBool));

  IndexMap<RealType> gradientMap(indices.data_ptr<int64_t>(), values.data_ptr<RealType>(), i64WorkSize, fail.data_ptr<bool>());

  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_inLinearWeights = inLinearWeights.data_ptr<RealType>();
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();

  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1)/threadsPerBlock.y, (i64NumTrees + threadsPerBlock.z-1)/threadsPerBlock.z);

  std::vector<torch::Tensor> vGradTensors(6);

  if (bInDataGrad) {
    values.zero_();
    indices.fill_(gradientMap.UnoccupiedKey());

    FusedBackwardDataKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_inLinearWeights, p_outDataGrad, gradientMap, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum, i64OutChannels);

    if (fail.to(torch::kCPU).item<bool>()) {
      OverflowErrorStream("Error: IndexMap overflowed!").raise();
      return std::vector<torch::Tensor>();
    }

    vGradTensors[0] = make_sparse(indices, values, inData.sizes());
  }
  
  if (bInThresholdsGrad) {
    values.zero_();
    indices.fill_(gradientMap.UnoccupiedKey());
   
    FusedBackwardThresholdsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_inLinearWeights, p_outDataGrad, gradientMap,
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum, i64OutChannels);

    if (fail.to(torch::kCPU).item<bool>()) {
      OverflowErrorStream("Error: IndexMap overflowed!").raise();
      return std::vector<torch::Tensor>();
    }

    vGradTensors[1] = make_sparse(indices, values, inThresholds.sizes());
  }
  
  if (bInWeightsGrad) {
    values.zero_();
    indices.fill_(gradientMap.UnoccupiedKey());
   
    FusedBackwardWeightsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inLinearWeights, p_outDataGrad, gradientMap, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum, i64OutChannels);

    if (fail.to(torch::kCPU).item<bool>()) {
      OverflowErrorStream("Error: IndexMap overflowed!").raise();
      return std::vector<torch::Tensor>();
    }

    vGradTensors[3] = make_sparse(indices, values, inWeights.sizes());
  }

  if (bInLinearWeightsGrad) {
    torch::Tensor inLinearWeightsGrad = torch::zeros_like(inLinearWeights);
    RealType * const p_inLinearWeightsGrad = inLinearWeightsGrad.data_ptr<RealType>();

    FusedBackwardLinearWeightsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_outDataGrad, p_inLinearWeightsGrad, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum, i64OutChannels);

    vGradTensors[4] = inLinearWeightsGrad;
  }

  if (bInLinearBiasGrad) {
    std::vector<IntArrayRef::value_type> vSumOver(vSizes.size()-1);
    vSumOver[0] = 0;
    std::iota(vSumOver.begin()+1, vSumOver.end(), 2);

    torch::Tensor inLinearBiasGrad = outDataGrad.sum(IntArrayRef(vSumOver.data(), vSumOver.size()));

    vGradTensors[5] = inLinearBiasGrad;
  }

  return vGradTensors;
}

template std::vector<torch::Tensor> sparse_hingetree_gpu_backward<float, bleak::HingeTreeCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);
template std::vector<torch::Tensor> sparse_hingetree_gpu_backward<double, bleak::HingeTreeCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);

template std::vector<torch::Tensor> sparse_hingetree_gpu_backward<float, bleak::HingeFernCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);
template std::vector<torch::Tensor> sparse_hingetree_gpu_backward<double, bleak::HingeFernCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);

template std::vector<torch::Tensor> sparse_hingetree_fused_linear_gpu_backward<float, bleak::HingeTreeCommon<float>>(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

template std::vector<torch::Tensor> sparse_hingetree_fused_linear_gpu_backward<double, bleak::HingeTreeCommon<double>>(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

template std::vector<torch::Tensor> sparse_hingetree_fused_linear_gpu_backward<float, bleak::HingeFernCommon<float>>(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

template std::vector<torch::Tensor> sparse_hingetree_fused_linear_gpu_backward<double, bleak::HingeFernCommon<double>>(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

