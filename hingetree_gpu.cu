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

#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>

#include "torch/extension.h"
#include "HingeTreeCommon.cuh"

#include "hingetree_error.h"

#include <cuda.h>

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
__global__ void ForwardKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, RealType *d_outData, 
    int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum) {

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

    const RealType * const d_leafWeights = d_inWeights + (j*i64WeightsStride + key)*i64InnerWeightsNum;
    RealType * const d_out = d_outData + ((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum;

    for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
      d_out[l] = d_leafWeights[l] * margin;
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void ReachabilityKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, int64_t *d_outCounts, 
    int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64NumTrees, int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (j < i64NumTrees) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;
    int64_t * const d_counts = d_outCounts + j*i64WeightsStride;

    for (int64_t i = 0; i < i64OuterNum; ++i) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        const RealType * const d_row = d_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k);

        // leaf key, margin, ordinal index
        const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, i64TreeDepth, i64InnerDataNum);

        const KeyType key = keyMarginTuple.leafKey;
        d_counts[key] += 1;
      }
    }
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void LeafMapKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, int64_t *d_outData, 
    int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64NumTrees, int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum) {

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

    d_outData[(i*i64NumTrees + j)*i64InnerDataNum + k] = key;
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void MarginMapKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, RealType *d_outMargins, 
    int64_t *d_outOrdinals, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64NumTrees, int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum) {

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

    const RealType signedMargin = keyMarginTuple.signedMargin;
    const KeyType thresholdIndex = keyMarginTuple.thresholdIndex;

    d_outMargins[(i*i64NumTrees + j)*i64InnerDataNum + k] = signedMargin;
    d_outOrdinals[(i*i64NumTrees + j)*i64InnerDataNum + k] = d_ordinals[thresholdIndex];
  }
}

// Execute each example on one tree per thread for deterministic gradients
// This is potentially *really* slow
template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void DeterministicBackwardThresholdsKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inThresholdsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  // Tree index
  const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (j < i64NumTrees) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;
    RealType * const d_thresholdsGradient = d_inThresholdsGradient + j*i64ThresholdStride;

    for (int64_t i = 0; i < i64OuterNum; ++i) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        const RealType * const d_row = d_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k);

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

        d_thresholdsGradient[thresholdIndex] += tmpSum; // Do this just once
      }
    }
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardThresholdsKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inThresholdsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i64OuterNum && j < i64NumTrees && k < i64InnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;
    RealType * const d_thresholdsGradient = d_inThresholdsGradient + j*i64ThresholdStride;

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

    atomicAdd(d_thresholdsGradient + thresholdIndex, tmpSum); // Do this just once

    //for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
      //d_thresholdsGradient[thresholdIndex] += -sign * d_leafWeights[l] * d_outGradient[l];
  }
}

// Execute each example on one tree per thread for deterministic gradients
// This is potentially *really* slow
template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void DeterministicBackwardWeightsKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, /*const RealType *d_inWeights,*/
    const RealType *d_outDataGradient, RealType *d_inWeightsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  // Tree index
  const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (j < i64NumTrees) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

    for (int64_t i = 0; i < i64OuterNum; ++i) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        const RealType * const d_row = d_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k);

        // leaf key, margin, ordinal index
        const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_row, d_thresholds, d_ordinals, i64TreeDepth, i64InnerDataNum);

        const KeyType key = keyMarginTuple.leafKey;
        const RealType signedMargin = keyMarginTuple.signedMargin;
        const RealType margin = std::abs(signedMargin);

        const RealType * const d_outGradient = d_outDataGradient + ((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum;
        RealType * const d_leafWeightsGradient = d_inWeightsGradient + (j*i64WeightsStride + key)*i64InnerWeightsNum;

        for (int64_t l = 0; l < i64InnerWeightsNum; ++l) {
          d_leafWeightsGradient[l] += margin * d_outGradient[l]; 
        }
      }
    }
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardWeightsKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, /*const RealType *d_inWeights,*/
    const RealType *d_outDataGradient, RealType *d_inWeightsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
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
    RealType * const d_leafWeightsGradient = d_inWeightsGradient + (j*i64WeightsStride + key)*i64InnerWeightsNum;

    for (int64_t l = 0; l < i64InnerWeightsNum; ++l) {
      atomicAdd(d_leafWeightsGradient + l, margin * d_outGradient[l]); // Really bad!
      //d_leafWeightsGradient[l] += margin * d_outGradient[l];
    }
  }
}
// Execute all trees on one example per thread for deterministic gradients
// This is potentially *really* slow
template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void DeterministicBackwardDataKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inDataGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  // Batch and inner indices
  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i64OuterNum && k < i64InnerDataNum) {
    const RealType * const d_row = d_inData + ((i*i64NumChannels + 0)*i64InnerDataNum + k);

    for (int64_t j = 0; j < i64NumTrees; ++j) {
      const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
      const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

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

      d_inDataGradient[(i*i64NumChannels + i64InputIndex)*i64InnerDataNum + k] += tmpSum; 
    }
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardDataKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inDataGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
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

    atomicAdd(d_inDataGradient + ((i*i64NumChannels + i64InputIndex)*i64InnerDataNum + k), tmpSum); // Do this just once

    //d_inDataGradient[(i*iNumChannels + iInputIndex)*iInnerDataNum + k] += tmpSum; // Do this just once
  }
}

} // end anonymous namespace

typedef c10::IntArrayRef IntArrayRef;

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_gpu_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2) {
    InvalidArgumentStream("Error: inData, inWeights must have at least 2 dimensions. inThresholds and inOrdinals must have 2 dimensions.").raise();
    return torch::Tensor();
  }

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0]) {
    InvalidArgumentStream("Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0].").raise();
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

  if (inOrdinals.min().to(torch::kCPU).item<int64_t>() < 0 || inOrdinals.max().to(torch::kCPU).item<int64_t>() >= i64NumChannels) {
    RunTimeErrorStream("Error: An ordinal value is either negative or larger than or equal to the number of channels.").raise();
    return torch::Tensor();
  }
 
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
  
  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1)/threadsPerBlock.y, (i64NumTrees + threadsPerBlock.z-1)/threadsPerBlock.z);

  ForwardKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_outData, 
    i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum);

  return outData;
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_gpu_backward_deterministic(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
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

  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();

  std::vector<torch::Tensor> vGradTensors(4);

  if (bInDataGrad) {
    torch::Tensor inDataGrad = torch::zeros_like(inData);
    RealType * const p_inDataGrad = inDataGrad.data_ptr<RealType>();

    const dim3 threadsPerBlock(32,32);
    const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1)/threadsPerBlock.y);

    DeterministicBackwardDataKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_outDataGrad, p_inDataGrad, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum);

    vGradTensors[0] = inDataGrad;
  }
  
  if (bInThresholdsGrad) {
    torch::Tensor inThresholdsGrad = torch::zeros_like(inThresholds);
    RealType * const p_inThresholdsGrad = inThresholdsGrad.data_ptr<RealType>();
    
    const dim3 threadsPerBlock(1024);
    const dim3 numBlocks((i64NumTrees + threadsPerBlock.x-1)/threadsPerBlock.x);

    DeterministicBackwardThresholdsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_outDataGrad, p_inThresholdsGrad, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum);

    vGradTensors[1] = inThresholdsGrad;
  }
  
  if (bInWeightsGrad) {
    torch::Tensor inWeightsGrad = torch::zeros_like(inWeights);
    RealType * const p_inWeightsGrad = inWeightsGrad.data_ptr<RealType>();
    
    const dim3 threadsPerBlock(1024);
    const dim3 numBlocks((i64NumTrees + threadsPerBlock.x-1)/threadsPerBlock.x);

    DeterministicBackwardWeightsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_outDataGrad, p_inWeightsGrad, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum);

    vGradTensors[3] = inWeightsGrad;
  }

  return vGradTensors;
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
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

  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();

  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1)/threadsPerBlock.y, (i64NumTrees + threadsPerBlock.z-1)/threadsPerBlock.z);

  std::vector<torch::Tensor> vGradTensors(4);

  if (bInDataGrad) {
    torch::Tensor inDataGrad = torch::zeros_like(inData);
    RealType * const p_inDataGrad = inDataGrad.data_ptr<RealType>();
    
    BackwardDataKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_outDataGrad, p_inDataGrad, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum);

    vGradTensors[0] = inDataGrad;
  }
  
  if (bInThresholdsGrad) {
    torch::Tensor inThresholdsGrad = torch::zeros_like(inThresholds);
    RealType * const p_inThresholdsGrad = inThresholdsGrad.data_ptr<RealType>();
    
    BackwardThresholdsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_outDataGrad, p_inThresholdsGrad, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum);

    vGradTensors[1] = inThresholdsGrad;
  }
  
  if (bInWeightsGrad) {
    torch::Tensor inWeightsGrad = torch::zeros_like(inWeights);
    RealType * const p_inWeightsGrad = inWeightsGrad.data_ptr<RealType>();
    
    BackwardWeightsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_outDataGrad, p_inWeightsGrad, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum);

    vGradTensors[3] = inWeightsGrad;
  }

  return vGradTensors;
}

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_gpu_reachability(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2) {
    InvalidArgumentStream("Error: inData, inWeights must have at least 2 dimensions. inThresholds and inOrdinals must have 2 dimensions.").raise();
    return torch::Tensor();
  }

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0]) {
    InvalidArgumentStream("Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0].").raise();
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

  if (inOrdinals.min().to(torch::kCPU).item<int64_t>() < 0 || inOrdinals.max().to(torch::kCPU).item<int64_t>() >= i64NumChannels) {
    RunTimeErrorStream("Error: An ordinal value is either negative or larger than or equal to the number of channels.").raise();
    return torch::Tensor();
  }
 
  const RealType * const p_inData = inData.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  
  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());
  torch::Tensor outCounts = torch::zeros(inWeights.sizes().slice(0,2), clOptions.dtype(torch::kInt64));
  
  int64_t * const p_outCounts = outCounts.data_ptr<int64_t>();
  
  int64_t i64InnerDataNum = 1;
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }
  
  const dim3 threadsPerBlock(1024);
  const dim3 numBlocks((i64NumTrees + threadsPerBlock.x-1)/threadsPerBlock.x);

  ReachabilityKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_outCounts,
    i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum);

  return outCounts;
}

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_gpu_leafmap(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2) {
    InvalidArgumentStream("Error: inData, inWeights must have at least 2 dimensions. inThresholds and inOrdinals must have 2 dimensions.").raise();
    return torch::Tensor();
  }

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0]) {
    InvalidArgumentStream("Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0].").raise();
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

  if (inOrdinals.min().to(torch::kCPU).item<int64_t>() < 0 || inOrdinals.max().to(torch::kCPU).item<int64_t>() >= i64NumChannels) {
    RunTimeErrorStream("Error: An ordinal value is either negative or larger than or equal to the number of channels.").raise();
    return torch::Tensor();
  }
 
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

  torch::Tensor outData = torch::empty(IntArrayRef(vSizes.data(), vSizes.size()), clOptions.dtype(torch::kInt64));
  
  int64_t * const p_outData = outData.data_ptr<int64_t>();
  
  int64_t i64InnerDataNum = 1;
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }
  
  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1)/threadsPerBlock.y, (i64NumTrees + threadsPerBlock.z-1)/threadsPerBlock.z);

  LeafMapKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_outData, 
    i64TreeDepth, i64NumDecisionsPerTree, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum);

  return outData;
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_gpu_marginmap(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

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
  int64_t * const p_outOrdinals = outOrdinals.data_ptr<int64_t>();
  
  int64_t i64InnerDataNum = 1;
  
  {
    auto inDataSlice = inData.sizes().slice(2);
    i64InnerDataNum = std::accumulate(inDataSlice.begin(), inDataSlice.end(), (int64_t)1, std::multiplies<IntArrayRef::value_type>());
  }
  
  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1)/threadsPerBlock.y, (i64NumTrees + threadsPerBlock.z-1)/threadsPerBlock.z);

  MarginMapKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_outMargins, p_outOrdinals, 
    i64TreeDepth, i64NumDecisionsPerTree, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum);

  return { outMargins, outOrdinals };
}

template torch::Tensor hingetree_gpu_forward<float, bleak::HingeTreeCommon<float>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
template torch::Tensor hingetree_gpu_forward<double, bleak::HingeTreeCommon<double>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template torch::Tensor hingetree_gpu_forward<float, bleak::HingeFernCommon<float>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
template torch::Tensor hingetree_gpu_forward<double, bleak::HingeFernCommon<double>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template std::vector<torch::Tensor> hingetree_gpu_backward<float, bleak::HingeTreeCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);
template std::vector<torch::Tensor> hingetree_gpu_backward<double, bleak::HingeTreeCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);

template std::vector<torch::Tensor> hingetree_gpu_backward_deterministic<float, bleak::HingeTreeCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);
template std::vector<torch::Tensor> hingetree_gpu_backward_deterministic<double, bleak::HingeTreeCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);

template std::vector<torch::Tensor> hingetree_gpu_backward<float, bleak::HingeFernCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);
template std::vector<torch::Tensor> hingetree_gpu_backward<double, bleak::HingeFernCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);

template std::vector<torch::Tensor> hingetree_gpu_backward_deterministic<float, bleak::HingeFernCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);
template std::vector<torch::Tensor> hingetree_gpu_backward_deterministic<double, bleak::HingeFernCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);

template torch::Tensor hingetree_gpu_reachability<float, bleak::HingeTreeCommon<float>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
template torch::Tensor hingetree_gpu_reachability<double, bleak::HingeTreeCommon<double>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template torch::Tensor hingetree_gpu_reachability<float, bleak::HingeFernCommon<float>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
template torch::Tensor hingetree_gpu_reachability<double, bleak::HingeFernCommon<double>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template torch::Tensor hingetree_gpu_leafmap<float, bleak::HingeTreeCommon<float>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
template torch::Tensor hingetree_gpu_leafmap<double, bleak::HingeTreeCommon<double>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template torch::Tensor hingetree_gpu_leafmap<float, bleak::HingeFernCommon<float>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
template torch::Tensor hingetree_gpu_leafmap<double, bleak::HingeFernCommon<double>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template std::vector<torch::Tensor> hingetree_gpu_marginmap<float, bleak::HingeTreeCommon<float>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
template std::vector<torch::Tensor> hingetree_gpu_marginmap<double, bleak::HingeTreeCommon<double>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template std::vector<torch::Tensor> hingetree_gpu_marginmap<float, bleak::HingeFernCommon<float>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
template std::vector<torch::Tensor> hingetree_gpu_marginmap<double, bleak::HingeFernCommon<double>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

