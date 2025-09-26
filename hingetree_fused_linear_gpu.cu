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

#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>

#include "torch/extension.h"
#include "HingeTreeCommon.cuh"

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

template<typename RealType>
__global__ void InitForwardKernel(const RealType *d_inLinearBias, RealType *d_outData,
    int64_t i64InnerWeightsNum, int64_t i64OuterNum, int64_t i64InnerDataNum, int64_t i64OutChannels) {

  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t l = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;

  if (i < i64OuterNum && k < i64InnerDataNum && l < i64InnerWeightsNum) {
    for (int64_t o = 0; o < i64OutChannels; ++o) {
      d_outData[((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum + l] = d_inLinearBias[o];
    }
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void ForwardKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, const RealType *d_inLinearWeights, RealType *d_outData, 
    int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum, int64_t i64OutChannels) {

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

    for (int64_t o = 0; o < i64OutChannels; ++o) {
      const RealType scale = margin * d_inLinearWeights[o*i64NumTrees + j];

      RealType * const d_out = d_outData + ((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum;

      // TODO: Initialize d_out with bias
      for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
        atomicAdd(d_out + l, scale*d_leafWeights[l]); // XXX: Slow! Not deterministic!!!
    }
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void FusionForwardKernel(const RealType *d_inImg, const RealType *d_inVec, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, const RealType *d_inLinearWeights, RealType *d_outData, 
    int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, int64_t i64OuterNum, int64_t i64ImgChannels, int64_t i64VecChannels, int64_t i64InnerDataNum, int64_t i64OutChannels) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i64OuterNum && j < i64NumTrees && k < i64InnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

    const RealType * const d_imgRow = d_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k);
    const RealType * const d_vecRow = d_inVec + (i*i64VecChannels + 0);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_imgRow, d_vecRow, d_thresholds, d_ordinals, i64TreeDepth, i64ImgChannels, i64InnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const RealType margin = std::abs(signedMargin);

    const RealType * const d_leafWeights = d_inWeights + (j*i64WeightsStride + key)*i64InnerWeightsNum;

    for (int64_t o = 0; o < i64OutChannels; ++o) {
      const RealType scale = margin * d_inLinearWeights[o*i64NumTrees + j];

      RealType * const d_out = d_outData + ((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum;

      // TODO: Initialize d_out with bias
      for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
        atomicAdd(d_out + l, scale*d_leafWeights[l]); // XXX: Slow! Not deterministic!!!
    }
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardThresholdsKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, const RealType *d_inLinearWeights,
    const RealType *d_outDataGradient, RealType *d_inThresholdsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64NumChannels, int64_t i64InnerDataNum, int64_t i64OutChannels) {

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

    RealType tmpSum = RealType(0);
    for (int64_t o = 0; o < i64OutChannels; ++o) {
      const RealType * const d_outGradient = d_outDataGradient + ((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum;
      RealType tmpSum2 = RealType(0);

      for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
        tmpSum2 += d_leafWeights[l] * d_outGradient[l];

      tmpSum += tmpSum2 * d_inLinearWeights[o*i64NumTrees + j];
    }

    tmpSum *= -sign;

    atomicAdd(d_thresholdsGradient + thresholdIndex, tmpSum); // Do this just once
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void FusionBackwardThresholdsKernel(const RealType *d_inImg, const RealType *d_inVec, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, const RealType *d_inLinearWeights,
    const RealType *d_outDataGradient, RealType *d_inThresholdsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64ImgChannels, int64_t i64VecChannels, int64_t i64InnerDataNum, int64_t i64OutChannels) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i64OuterNum && j < i64NumTrees && k < i64InnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;
    RealType * const d_thresholdsGradient = d_inThresholdsGradient + j*i64ThresholdStride;

    const RealType * const d_imgRow = d_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k);
    const RealType * const d_vecRow = d_inVec + (i*i64VecChannels + 0);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_imgRow, d_vecRow, d_thresholds, d_ordinals, i64TreeDepth, i64ImgChannels, i64InnerDataNum);

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

    atomicAdd(d_thresholdsGradient + thresholdIndex, tmpSum); // Do this just once
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardWeightsKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inLinearWeights,
    const RealType *d_outDataGradient, RealType *d_inWeightsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
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

    RealType * const d_leafWeightsGradient = d_inWeightsGradient + (j*i64WeightsStride + key)*i64InnerWeightsNum;

    for (int64_t l = 0; l < i64InnerWeightsNum; ++l) {
      RealType tmpSum = RealType(0);

      for (int64_t o = 0; o < i64OutChannels; ++o) {
        const RealType * const d_outGradient = d_outDataGradient + ((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum;
        tmpSum += d_inLinearWeights[o*i64NumTrees + j] * d_outGradient[l];
      }

      tmpSum *= margin;

      atomicAdd(d_leafWeightsGradient + l, tmpSum); // Really bad!
    }
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void FusionBackwardWeightsKernel(const RealType *d_inImg, const RealType *d_inVec, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inLinearWeights,
    const RealType *d_outDataGradient, RealType *d_inWeightsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64ImgChannels, int64_t i64VecChannels, int64_t i64InnerDataNum, int64_t i64OutChannels) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i64OuterNum && j < i64NumTrees && k < i64InnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

    const RealType * const d_imgRow = d_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k);
    const RealType * const d_vecRow = d_inVec + (i*i64VecChannels + 0);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_imgRow, d_vecRow, d_thresholds, d_ordinals, i64TreeDepth, i64ImgChannels, i64InnerDataNum);

    const KeyType key = keyMarginTuple.leafKey;
    const RealType signedMargin = keyMarginTuple.signedMargin;
    const RealType margin = std::abs(signedMargin);

    RealType * const d_leafWeightsGradient = d_inWeightsGradient + (j*i64WeightsStride + key)*i64InnerWeightsNum;

    for (int64_t l = 0; l < i64InnerWeightsNum; ++l) {
      RealType tmpSum = RealType(0);

      for (int64_t o = 0; o < i64OutChannels; ++o) {
        const RealType * const d_outGradient = d_outDataGradient + ((i*i64OutChannels + o)*i64InnerDataNum + k)*i64InnerWeightsNum;
        tmpSum += d_inLinearWeights[o*i64NumTrees + j] * d_outGradient[l];
      }

      tmpSum *= margin;

      atomicAdd(d_leafWeightsGradient + l, tmpSum); // Really bad!
    }
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardDataKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, const RealType *d_inLinearWeights,
    const RealType *d_outDataGradient, RealType *d_inDataGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
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

    atomicAdd(d_inDataGradient + ((i*i64NumChannels + i64InputIndex)*i64InnerDataNum + k), tmpSum); // Do this just once
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void FusionBackwardDataKernel(const RealType *d_inImg, const RealType *d_inVec, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, const RealType *d_inLinearWeights,
    const RealType *d_outDataGradient, RealType *d_inImgGradient, RealType *d_inVecGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64ImgChannels, int64_t i64VecChannels, int64_t i64InnerDataNum, int64_t i64OutChannels) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i64OuterNum && j < i64NumTrees && k < i64InnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

    const RealType * const d_imgRow = d_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k);
    const RealType * const d_vecRow = d_inVec + (i*i64VecChannels + 0);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_imgRow, d_vecRow, d_thresholds, d_ordinals, i64TreeDepth, i64ImgChannels, i64InnerDataNum);

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

    if (i64InputIndex < i64ImgChannels)
      atomicAdd(d_inImgGradient + ((i*i64ImgChannels + i64InputIndex)*i64InnerDataNum + k), tmpSum); // Do this just once
    else
      atomicAdd(d_inVecGradient + (i*i64VecChannels + i64InputIndex - i64ImgChannels), tmpSum); // Do this just once
  }
}

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardLinearWeightsKernel(const RealType *d_inData, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights,
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

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void FusionBackwardLinearWeightsKernel(const RealType *d_inImg, const RealType *d_inVec, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights,
    const RealType *d_outDataGradient, RealType *d_inLinearWeightsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64ImgChannels, int64_t i64VecChannels, int64_t i64InnerDataNum, int64_t i64OutChannels) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (i < i64OuterNum && j < i64NumTrees && k < i64InnerDataNum) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

    const RealType * const d_imgRow = d_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k);
    const RealType * const d_vecRow = d_inVec + (i*i64VecChannels + 0);

    // leaf key, margin, ordinal index
    const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_imgRow, d_vecRow, d_thresholds, d_ordinals, i64TreeDepth, i64ImgChannels, i64InnerDataNum);

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

} // end namespace

typedef c10::IntArrayRef IntArrayRef;

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_fused_linear_gpu_forward(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

  if (inData.dim() < 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2 || inLinearWeights.dim() != 2 || inLinearBias.dim() != 1) {
    std::cerr << "Error: inData, inWeights must have at least 2 dimensions. inThresholds, inOrdinals, inLinearWeights must have 2 dimensions. inLinearBias must have 1 dimension." << std::endl;
    return torch::Tensor();
  }

  if (inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0] || inWeights.sizes()[0] != inLinearWeights.sizes()[1] || inLinearWeights.sizes()[0] != inLinearBias.sizes()[0]) {
    std::cerr << "Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0]. inLinearWeightrs.shape[0] must be the same as inLinearBias.shape[0]." << std::endl;
    return torch::Tensor();
  }
  
  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth)) {
    std::cerr << "Error: The tree depth exceeds compile-time constraints or the number of thresholds is incorrect for the tree depth (tree depth = " << i64TreeDepth << ")." << std::endl;
    return torch::Tensor();
  }

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];
  const int64_t i64OutChannels = inLinearWeights.sizes()[0];

  if (inOrdinals.min().to(torch::kCPU).item<int64_t>() < 0 || inOrdinals.max().to(torch::kCPU).item<int64_t>() >= i64NumChannels) {
    std::cerr << "Error: An ordinal value is either negative or larger than or equal to the number of channels." << std::endl;
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

  // Stripe the bias term
  {
    const dim3 threadsPerBlock(16,8,8);
    const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1)/threadsPerBlock.y, (i64InnerWeightsNum + threadsPerBlock.z-1)/threadsPerBlock.z);

    InitForwardKernel<<<numBlocks, threadsPerBlock>>>(p_inLinearBias, p_outData, i64InnerWeightsNum, i64BatchSize, i64InnerDataNum, i64OutChannels);
  }
  
  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1)/threadsPerBlock.y, (i64NumTrees + threadsPerBlock.z-1)/threadsPerBlock.z);

  ForwardKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_inLinearWeights, p_outData, 
    i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum, i64OutChannels);

  return outData;
}

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_fusion_fused_linear_gpu_forward(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

  if (inImg.dim() < 2 || inVec.dim() != 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2 || inLinearWeights.dim() != 2 || inLinearBias.dim() != 1) {
    std::cerr << "Error: inImg, inVec, inWeights must have at least 2 dimensions. inThresholds, inOrdinals, inLinearWeights must have 2 dimensions. inLinearBias must have 1 dimension." << std::endl;
    return torch::Tensor();
  }

  if (inImg.sizes()[0] != inVec.sizes()[0] || inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0] || inWeights.sizes()[0] != inLinearWeights.sizes()[1] || inLinearWeights.sizes()[0] != inLinearBias.sizes()[0]) {
    std::cerr << "Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0]. inImg.shape[0] must be the same as inVec.shape[0]. inLinearWeights.sha[e0] must be the same as inLinearBias.shape[0]." << std::endl;
    return torch::Tensor();
  }
  
  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth)) {
    std::cerr << "Error: The tree depth exceeds compile-time constraints or the number of thresholds is incorrect for the tree depth (tree depth = " << i64TreeDepth << ")." << std::endl;
    return torch::Tensor();
  }

  const int64_t i64BatchSize = inImg.sizes()[0];
  const int64_t i64ImgChannels = inImg.sizes()[1];
  const int64_t i64VecChannels = inVec.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];
  const int64_t i64OutChannels = inLinearWeights.sizes()[0];

  if (inOrdinals.min().to(torch::kCPU).item<int64_t>() < 0 || inOrdinals.max().to(torch::kCPU).item<int64_t>() >= i64ImgChannels + i64VecChannels) {
    std::cerr << "Error: An ordinal value is either negative or larger than or equal to the number of image channels + feature vector channels (" << i64ImgChannels + i64VecChannels << ")." << std::endl;
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

  // Stripe the bias term
  {
    const dim3 threadsPerBlock(16,8,8);
    const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1)/threadsPerBlock.y, (i64InnerWeightsNum + threadsPerBlock.z-1)/threadsPerBlock.z);

    InitForwardKernel<<<numBlocks, threadsPerBlock>>>(p_inLinearBias, p_outData, i64InnerWeightsNum, i64BatchSize, i64InnerDataNum, i64OutChannels);
  }
  
  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1)/threadsPerBlock.y, (i64NumTrees + threadsPerBlock.z-1)/threadsPerBlock.z);

  FusionForwardKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inImg, p_inVec, p_inThresholds, p_inOrdinals, p_inWeights, p_inLinearWeights, p_outData, 
    i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64ImgChannels, i64VecChannels, i64InnerDataNum, i64OutChannels);

  return outData;
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_fused_linear_gpu_backward(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

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
  const int64_t i64OutChannels = inLinearWeights.sizes()[0];
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth)) {
    std::cerr << "Error: The tree depth exceeds compile-time constraints or the number of thresholds is incorrect for the tree depth (tree depth = " << i64TreeDepth << ")." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];

  if (inOrdinals.min().to(torch::kCPU).item<int64_t>() < 0 || inOrdinals.max().to(torch::kCPU).item<int64_t>() >= i64NumChannels) {
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
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();

  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1)/threadsPerBlock.y, (i64NumTrees + threadsPerBlock.z-1)/threadsPerBlock.z);

  std::vector<torch::Tensor> vGradTensors(6);

  if (bInDataGrad) {
    torch::Tensor inDataGrad = torch::zeros_like(inData);
    RealType * const p_inDataGrad = inDataGrad.data_ptr<RealType>();
    
    BackwardDataKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_inLinearWeights, p_outDataGrad, p_inDataGrad, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum, i64OutChannels);

    vGradTensors[0] = inDataGrad;
  }
  
  if (bInThresholdsGrad) {
    torch::Tensor inThresholdsGrad = torch::zeros_like(inThresholds);
    RealType * const p_inThresholdsGrad = inThresholdsGrad.data_ptr<RealType>();
    
    BackwardThresholdsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_inLinearWeights, p_outDataGrad, p_inThresholdsGrad, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum, i64OutChannels);

    vGradTensors[1] = inThresholdsGrad;
  }
  
  if (bInWeightsGrad) {
    torch::Tensor inWeightsGrad = torch::zeros_like(inWeights);
    RealType * const p_inWeightsGrad = inWeightsGrad.data_ptr<RealType>();
    
    BackwardWeightsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inLinearWeights, p_outDataGrad, p_inWeightsGrad, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64NumChannels, i64InnerDataNum, i64OutChannels);

    vGradTensors[3] = inWeightsGrad;
  }

  if (bInLinearWeightsGrad) {
    torch::Tensor inLinearWeightsGrad = torch::zeros_like(inLinearWeights);
    RealType * const p_inLinearWeightsGrad = inLinearWeightsGrad.data_ptr<RealType>();

    BackwardLinearWeightsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inData, p_inThresholds, p_inOrdinals, p_inWeights, p_outDataGrad, p_inLinearWeightsGrad, 
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

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_fusion_fused_linear_gpu_backward(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

  if (bInOrdinalsGrad) { // Not differentiable, ever!
    std::cerr << "Error: Gradient on inOrdinals is requested, but inOrdinals never has a gradient." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  if (inImg.dim() < 2 || inVec.dim() != 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2 || inLinearWeights.dim() != 2 || inLinearBias.dim() != 1 || outDataGrad.dim() < 2) {
    std::cerr << "Error: inImg, inVec, inWeights must have at least 2 dimensions. inThresholds, inOrdinals, inLinearWeights must have 2 dimensions. inLinearBias must have 1 dimension." << std::endl;
    return std::vector<torch::Tensor>();
  }

  if (inImg.sizes()[0] != inVec.sizes()[0] || inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0] || inWeights.sizes()[0] != inLinearWeights.sizes()[1] || inLinearWeights.sizes()[0] != inLinearBias.sizes()[0]) {
    std::cerr << "Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0]. inImg.shape[0] must be the same as inVec.shape[0]. inLinearWeights.sha[e0] must be the same as inLinearBias.shape[0]." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  const int64_t i64OutChannels = inLinearWeights.sizes()[0];
  
  if (i64TreeDepth > TreeTraitsType::GetMaxDepth() || inThresholds.sizes()[1] != TreeTraitsType::GetThresholdCount(i64TreeDepth)) {
    std::cerr << "Error: The tree depth exceeds compile-time constraints or the number of thresholds is incorrect for the tree depth (tree depth = " << i64TreeDepth << ")." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  const int64_t i64BatchSize = inImg.sizes()[0];
  const int64_t i64ImgChannels = inImg.sizes()[1];
  const int64_t i64VecChannels = inVec.sizes()[1];
  const int64_t i64NumDecisionsPerTree = inThresholds.sizes()[1];

  if (inOrdinals.min().to(torch::kCPU).item<int64_t>() < 0 || inOrdinals.max().to(torch::kCPU).item<int64_t>() >= i64ImgChannels + i64VecChannels) {
    std::cerr << "Error: An ordinal value is either negative or larger than or equal to the number of image channels + feature vector channels (" << i64ImgChannels + i64VecChannels << ")." << std::endl;
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
    std::cerr << "Error: outDataGrad.shape does not match the expected shape (" << outDataGrad.sizes() << " != " << IntArrayRef(vSizes.data(), vSizes.size()) << ")." << std::endl;
    return std::vector<torch::Tensor>();
  }

  const RealType * const p_inImg = inImg.data_ptr<RealType>();
  const RealType * const p_inVec = inVec.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_inLinearWeights = inLinearWeights.data_ptr<RealType>();
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();

  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1)/threadsPerBlock.y, (i64NumTrees + threadsPerBlock.z-1)/threadsPerBlock.z);

  std::vector<torch::Tensor> vGradTensors(7);

  if (bInImgGrad || bInVecGrad) {
    torch::Tensor inImgGrad = torch::zeros_like(inImg);
    torch::Tensor inVecGrad = torch::zeros_like(inVec);
    RealType * const p_inImgGrad = inImgGrad.data_ptr<RealType>();
    RealType * const p_inVecGrad = inVecGrad.data_ptr<RealType>();
    
    FusionBackwardDataKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inImg, p_inVec, p_inThresholds, p_inOrdinals, p_inWeights, p_inLinearWeights, p_outDataGrad, p_inImgGrad, p_inVecGrad,
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64ImgChannels, i64VecChannels, i64InnerDataNum, i64OutChannels);

    vGradTensors[0] = inImgGrad;
    vGradTensors[1] = inVecGrad;
  }
  
  if (bInThresholdsGrad) {
    torch::Tensor inThresholdsGrad = torch::zeros_like(inThresholds);
    RealType * const p_inThresholdsGrad = inThresholdsGrad.data_ptr<RealType>();
    
    FusionBackwardThresholdsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inImg, p_inVec, p_inThresholds, p_inOrdinals, p_inWeights, p_inLinearWeights, p_outDataGrad, p_inThresholdsGrad, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64ImgChannels, i64VecChannels, i64InnerDataNum, i64OutChannels);

    vGradTensors[2] = inThresholdsGrad;
  }
  
  if (bInWeightsGrad) {
    torch::Tensor inWeightsGrad = torch::zeros_like(inWeights);
    RealType * const p_inWeightsGrad = inWeightsGrad.data_ptr<RealType>();
    
    FusionBackwardWeightsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inImg, p_inVec, p_inThresholds, p_inOrdinals, p_inLinearWeights, p_outDataGrad, p_inWeightsGrad, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64ImgChannels, i64VecChannels, i64InnerDataNum, i64OutChannels);

    vGradTensors[4] = inWeightsGrad;
  }

  if (bInLinearWeightsGrad) {
    torch::Tensor inLinearWeightsGrad = torch::zeros_like(inLinearWeights);
    RealType * const p_inLinearWeightsGrad = inLinearWeightsGrad.data_ptr<RealType>();

    FusionBackwardLinearWeightsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inImg, p_inVec, p_inThresholds, p_inOrdinals, p_inWeights, p_outDataGrad, p_inLinearWeightsGrad, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64ImgChannels, i64VecChannels, i64InnerDataNum, i64OutChannels);

    vGradTensors[5] = inLinearWeightsGrad;
  }

  if (bInLinearBiasGrad) {
    std::vector<IntArrayRef::value_type> vSumOver(vSizes.size()-1);
    vSumOver[0] = 0;
    std::iota(vSumOver.begin()+1, vSumOver.end(), 2);

    torch::Tensor inLinearBiasGrad = outDataGrad.sum(IntArrayRef(vSumOver.data(), vSumOver.size()));

    vGradTensors[6] = inLinearBiasGrad;
  }

  return vGradTensors;
}

template torch::Tensor hingetree_fused_linear_gpu_forward<float, bleak::HingeTreeCommon<float>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias);
template torch::Tensor hingetree_fused_linear_gpu_forward<double, bleak::HingeTreeCommon<double>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias);

template torch::Tensor hingetree_fused_linear_gpu_forward<float, bleak::HingeFernCommon<float>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias);
template torch::Tensor hingetree_fused_linear_gpu_forward<double, bleak::HingeFernCommon<double>>(torch::Tensor inData, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias);

template std::vector<torch::Tensor> hingetree_fused_linear_gpu_backward<float, bleak::HingeTreeCommon<float>>(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

template std::vector<torch::Tensor> hingetree_fused_linear_gpu_backward<double, bleak::HingeTreeCommon<double>>(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

template std::vector<torch::Tensor> hingetree_fused_linear_gpu_backward<float, bleak::HingeFernCommon<float>>(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

template std::vector<torch::Tensor> hingetree_fused_linear_gpu_backward<double, bleak::HingeFernCommon<double>>(torch::Tensor inData, bool bInDataGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

// Fusion functions
template torch::Tensor hingetree_fusion_fused_linear_gpu_forward<float, bleak::HingeTreeCommon<float>>(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias);
template torch::Tensor hingetree_fusion_fused_linear_gpu_forward<double, bleak::HingeTreeCommon<double>>(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias);

template torch::Tensor hingetree_fusion_fused_linear_gpu_forward<float, bleak::HingeFernCommon<float>>(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias);
template torch::Tensor hingetree_fusion_fused_linear_gpu_forward<double, bleak::HingeFernCommon<double>>(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights, torch::Tensor inLinearWeights, torch::Tensor inLinearBias);

template std::vector<torch::Tensor> hingetree_fusion_fused_linear_gpu_backward<float, bleak::HingeTreeCommon<float>>(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

template std::vector<torch::Tensor> hingetree_fusion_fused_linear_gpu_backward<double, bleak::HingeTreeCommon<double>>(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

template std::vector<torch::Tensor> hingetree_fusion_fused_linear_gpu_backward<float, bleak::HingeFernCommon<float>>(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

template std::vector<torch::Tensor> hingetree_fusion_fused_linear_gpu_backward<double, bleak::HingeFernCommon<double>>(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor inLinearWeights, bool bInLinearWeightsGrad, torch::Tensor inLinearBias, bool bInLinearBiasGrad, torch::Tensor outDataGrad);

