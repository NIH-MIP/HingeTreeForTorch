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

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void ForwardKernel(const RealType *d_inImg, const RealType *d_inVec, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, RealType *d_outData, 
    int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, int64_t i64OuterNum, int64_t i64ImgChannels, int64_t i64VecChannels, int64_t i64InnerDataNum) {

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
    RealType * const d_out = d_outData + ((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum;

    for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
      d_out[l] = d_leafWeights[l] * margin;
  }
}

// Execute each example on one tree per thread for deterministic gradients
// This is potentially *really* slow
template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void DeterministicBackwardThresholdsKernel(const RealType *d_inImg, const RealType *d_inVec, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inThresholdsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64ImgChannels, int64_t i64VecChannels, int64_t i64InnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  // Tree index
  const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (j < i64NumTrees) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;
    RealType * const d_thresholdsGradient = d_inThresholdsGradient + j*i64ThresholdStride;

    for (int64_t i = 0; i < i64OuterNum; ++i) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        const RealType * const d_imgRow = d_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k);
        const RealType * const d_vecRow = d_inVec + (i*i64VecChannels + 0);

        const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_imgRow, d_vecRow, d_thresholds, d_ordinals, i64TreeDepth, i64ImgChannels, i64InnerDataNum);

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
__global__ void BackwardThresholdsKernel(const RealType *d_inImg, const RealType *d_inVec, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inThresholdsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64ImgChannels, int64_t i64VecChannels, int64_t i64InnerDataNum) {

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
__global__ void DeterministicBackwardWeightsKernel(const RealType *d_inImg, const RealType *d_inVec, const RealType *d_inThresholds, const int64_t *d_inOrdinals, /*const RealType *d_inWeights,*/
    const RealType *d_outDataGradient, RealType *d_inWeightsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64ImgChannels, int64_t i64VecChannels, int64_t i64InnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  // Tree index
  const int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;

  if (j < i64NumTrees) {
    const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
    const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

    for (int64_t i = 0; i < i64OuterNum; ++i) {
      for (int64_t k = 0; k < i64InnerDataNum; ++k) {
        const RealType * const d_imgRow = d_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k);
        const RealType * const d_vecRow = d_inVec + (i*i64VecChannels + 0);

        // leaf key, margin, ordinal index
        const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_imgRow, d_vecRow, d_thresholds, d_ordinals, i64TreeDepth, i64ImgChannels, i64InnerDataNum);

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
__global__ void BackwardWeightsKernel(const RealType *d_inImg, const RealType *d_inVec, const RealType *d_inThresholds, const int64_t *d_inOrdinals, /*const RealType *d_inWeights,*/
    const RealType *d_outDataGradient, RealType *d_inWeightsGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64ImgChannels, int64_t i64VecChannels, int64_t i64InnerDataNum) {

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

    const RealType * const d_outGradient = d_outDataGradient + ((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum;
    RealType * const d_leafWeightsGradient = d_inWeightsGradient + (j*i64WeightsStride + key)*i64InnerWeightsNum;

    for (int64_t l = 0; l < i64InnerWeightsNum; ++l) {
      atomicAdd(d_leafWeightsGradient + l, margin * d_outGradient[l]); // Really bad!
      //d_leafWeightsGradient[l] += margin * d_outGradient[l];
    }
  }
}

#if 0
// Execute all trees on one example per thread for deterministic gradients
// This is potentially *really* slow
// XXX: This is not deterministic.
template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void DeterministicBackwardDataKernel(const RealType *d_inImg, const RealType *d_inVec, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inImgGradient, RealType *d_inVecGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64ImgChannels, int64_t i64VecChannels, int64_t i64InnerDataNum) {

  typedef typename TreeTraitsTypeGPU::KeyType KeyType;

  // Batch and inner indices
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t k = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;

  if (i < i64OuterNum && k < i64InnerDataNum) {
    const RealType * const d_imgRow = d_inImg + ((i*i64ImgChannels + 0)*i64InnerDataNum + k);
    const RealType * const d_vecRow = d_inVec + (i*i64VecChannels + 0);

    for (int64_t j = 0; j < i64NumTrees; ++j) {
      const RealType * const d_thresholds = d_inThresholds + j*i64ThresholdStride;
      const int64_t * const d_ordinals = d_inOrdinals + j*i64ThresholdStride;

      // leaf key, margin, ordinal index
      const auto keyMarginTuple = TreeTraitsTypeGPU::ComputeKeyAndSignedMargin(d_imgRow, d_vecRow, d_thresholds, d_ordinals, i64TreeDepth, i64ImgChannels, i64InnerDataNum);

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

      if (i64InputIndex < i64ImgChannels) 
        d_inImgGradient[(i*i64ImgChannels + i64InputIndex)*i64InnerDataNum + k] += tmpSum; 
      else
        d_inVecGradient[i*i64VecChannels + i64InputIndex - i64ImgChannels] += tmpSum; 
    }
  }
}
#endif

template<typename TreeTraitsTypeGPU, typename RealType>
__global__ void BackwardDataKernel(const RealType *d_inImg, const RealType *d_inVec, const RealType *d_inThresholds, const int64_t *d_inOrdinals, const RealType *d_inWeights, 
    const RealType *d_outDataGradient, RealType *d_inImgGradient, RealType *d_inVecGradient, int64_t i64TreeDepth, int64_t i64ThresholdStride, int64_t i64WeightsStride, int64_t i64InnerWeightsNum, int64_t i64NumTrees, 
    int64_t i64OuterNum, int64_t i64ImgChannels, int64_t i64VecChannels, int64_t i64InnerDataNum) {

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
    const RealType * const d_outGradient = d_outDataGradient + ((i*i64NumTrees + j)*i64InnerDataNum + k)*i64InnerWeightsNum;

    const RealType sign = RealType((RealType(0) < signedMargin) - (signedMargin < RealType(0)));
    RealType tmpSum = RealType(0);

    for (int64_t l = 0; l < i64InnerWeightsNum; ++l)
      tmpSum += d_leafWeights[l] * d_outGradient[l];

    tmpSum *= sign;

    if (i64InputIndex < i64ImgChannels)
        atomicAdd(d_inImgGradient + ((i*i64ImgChannels + i64InputIndex)*i64InnerDataNum + k), tmpSum); // Do this just once
    else
        atomicAdd(d_inVecGradient + (i*i64VecChannels + i64InputIndex - i64ImgChannels), tmpSum); // Do this just once
  }
}

} // end anonymous namespace

typedef c10::IntArrayRef IntArrayRef;

template<typename RealType, typename TreeTraitsType>
torch::Tensor hingetree_fusion_gpu_forward(torch::Tensor inImg, torch::Tensor inVec, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

  if (inImg.dim() < 2 || inVec.dim() != 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2) {
    std::cerr << "Error: inImg, inVec, inWeights must have at least 2 dimensions. inThresholds and inOrdinals must have 2 dimensions." << std::endl;
    return torch::Tensor();
  }

  if (inImg.sizes()[0] != inVec.sizes()[0] || inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0]) {
    std::cerr << "Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0]. inImg.shape[0] must be the same as inVec.shape[0]." << std::endl;
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

  if (inOrdinals.min().to(torch::kCPU).item<int64_t>() < 0 || inOrdinals.max().to(torch::kCPU).item<int64_t>() >= i64ImgChannels + i64VecChannels) {
    std::cerr << "Error: An ordinal value is either negative or larger than or equal to the number of image channels + feature vector channels (" << i64ImgChannels + i64VecChannels << ")." << std::endl;
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
  
  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1)/threadsPerBlock.y, (i64NumTrees + threadsPerBlock.z-1)/threadsPerBlock.z);

  ForwardKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inImg, p_inVec, p_inThresholds, p_inOrdinals, p_inWeights, p_outData, 
    i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64ImgChannels, i64VecChannels, i64InnerDataNum);

  return outData;
}

template<typename RealType, typename TreeTraitsType>
std::vector<torch::Tensor> hingetree_fusion_gpu_backward(torch::Tensor inImg, bool bInImgGrad, torch::Tensor inVec, bool bInVecGrad, torch::Tensor inThresholds, bool bInThresholdsGrad, torch::Tensor inOrdinals, bool bInOrdinalsGrad, torch::Tensor inWeights, bool bInWeightsGrad, torch::Tensor outDataGrad) {
  typedef bleak::HingeTreeCommonGPU<TreeTraitsType> TreeTraitsTypeGPU;

  if (bInOrdinalsGrad) { // Not differentiable, ever!
    std::cerr << "Error: Gradient on inOrdinals is requested, but inOrdinals never has a gradient." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  if (inImg.dim() < 2 || inVec.dim() != 2 || inThresholds.dim() != 2 || inOrdinals.dim() != 2 || inWeights.dim() < 2) {
    std::cerr << "Error: inImg, inVec, inThresholds, inWeights are expected to share the same torch real number data type. inOrdinals is expected to be type torch.int64." << std::endl;
    return std::vector<torch::Tensor>();
  }

  if (inImg.sizes()[0] != inVec.sizes()[0] || inThresholds.sizes() != inOrdinals.sizes() || inWeights.sizes()[0] != inThresholds.sizes()[0]) {
    std::cerr << "Error: inThresholds and inOrdinals must be the same shape. inWeights.shape[0] must be the same as inThresholds.shape[0]. inImg.shape[0] must be the same as inVec.shape[0]." << std::endl;
    return std::vector<torch::Tensor>();
  }
  
  const int64_t i64NumTrees = inWeights.sizes()[0];
  const int64_t i64NumLeavesPerTree = inWeights.sizes()[1];
  const int64_t i64TreeDepth = TreeTraitsType::ComputeDepth(i64NumLeavesPerTree);
  
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
    std::cerr << "Error: outDataGrad.shape does not match the expected shape (" << outDataGrad.sizes() << " != " << IntArrayRef(vSizes.data(), vSizes.size()) << ")." << std::endl;
    return std::vector<torch::Tensor>();
  }

  const RealType * const p_inImg = inImg.data_ptr<RealType>();
  const RealType * const p_inVec = inVec.data_ptr<RealType>();
  const RealType * const p_inThresholds = inThresholds.data_ptr<RealType>();
  const int64_t * const p_inOrdinals = inOrdinals.data_ptr<int64_t>();
  const RealType * const p_inWeights = inWeights.data_ptr<RealType>();
  const RealType * const p_outDataGrad = outDataGrad.data_ptr<RealType>();

  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64InnerDataNum + threadsPerBlock.x-1)/threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1)/threadsPerBlock.y, (i64NumTrees + threadsPerBlock.z-1)/threadsPerBlock.z);

  std::vector<torch::Tensor> vGradTensors(5);

  if (bInImgGrad || bInVecGrad) {
    torch::Tensor inImgGrad = torch::zeros_like(inImg);
    torch::Tensor inVecGrad = torch::zeros_like(inVec);
    RealType * const p_inImgGrad = inImgGrad.data_ptr<RealType>();
    RealType * const p_inVecGrad = inVecGrad.data_ptr<RealType>();
    
    BackwardDataKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inImg, p_inVec, p_inThresholds, p_inOrdinals, p_inWeights, p_outDataGrad, p_inImgGrad, p_inVecGrad,
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64ImgChannels, i64VecChannels, i64InnerDataNum);

    vGradTensors[0] = inImgGrad;
    vGradTensors[1] = inVecGrad;
  }
  
  if (bInThresholdsGrad) {
    torch::Tensor inThresholdsGrad = torch::zeros_like(inThresholds);
    RealType * const p_inThresholdsGrad = inThresholdsGrad.data_ptr<RealType>();
    
    BackwardThresholdsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inImg, p_inVec, p_inThresholds, p_inOrdinals, p_inWeights, p_outDataGrad, p_inThresholdsGrad, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64ImgChannels, i64VecChannels, i64InnerDataNum);

    vGradTensors[2] = inThresholdsGrad;
  }
  
  if (bInWeightsGrad) {
    torch::Tensor inWeightsGrad = torch::zeros_like(inWeights);
    RealType * const p_inWeightsGrad = inWeightsGrad.data_ptr<RealType>();
    
    BackwardWeightsKernel<TreeTraitsTypeGPU><<<numBlocks, threadsPerBlock>>>(p_inImg, p_inVec, p_inThresholds, p_inOrdinals, p_outDataGrad, p_inWeightsGrad, 
      i64TreeDepth, i64NumDecisionsPerTree, i64NumLeavesPerTree, i64InnerWeightsNum, i64NumTrees, i64BatchSize, i64ImgChannels, i64VecChannels, i64InnerDataNum);

    vGradTensors[4] = inWeightsGrad;
  }

  return vGradTensors;
}

template torch::Tensor hingetree_fusion_gpu_forward<float, bleak::HingeTreeCommon<float>>(torch::Tensor, torch::Tensor, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
template torch::Tensor hingetree_fusion_gpu_forward<double, bleak::HingeTreeCommon<double>>(torch::Tensor, torch::Tensor, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template torch::Tensor hingetree_fusion_gpu_forward<float, bleak::HingeFernCommon<float>>(torch::Tensor, torch::Tensor, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);
template torch::Tensor hingetree_fusion_gpu_forward<double, bleak::HingeFernCommon<double>>(torch::Tensor, torch::Tensor, torch::Tensor inThresholds, torch::Tensor inOrdinals, torch::Tensor inWeights);

template std::vector<torch::Tensor> hingetree_fusion_gpu_backward<float, bleak::HingeTreeCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);
template std::vector<torch::Tensor> hingetree_fusion_gpu_backward<double, bleak::HingeTreeCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);

template std::vector<torch::Tensor> hingetree_fusion_gpu_backward<float, bleak::HingeFernCommon<float>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);
template std::vector<torch::Tensor> hingetree_fusion_gpu_backward<double, bleak::HingeFernCommon<double>>(torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor, bool, torch::Tensor);

