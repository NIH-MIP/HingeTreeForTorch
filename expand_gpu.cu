/*-
 * Nathan Lay
 * AI Resource at National Cancer Institute
 * National Institutes of Health
 * May 2023
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
#include "torch/extension.h"

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

// Make struct that can be conveniently passed to kernels by value
template<unsigned int Dimension>
struct Size {
  constexpr static size_t size = Dimension;
  int64_t data[Dimension];
};

__device__ void contract2d_coords(int64_t &xo, int64_t &yo, int64_t &xw, int64_t &yw, int64_t k, const int64_t sizes[4]) {
  int64_t q = k / sizes[3];
  xw = k - q*sizes[3];

  k = q;
  q = k / sizes[2];

  yw = k - q*sizes[2];

  k = q;
  q = k / sizes[1];

  xo = k - q*sizes[1];

  k = q;
  q = k / sizes[0];

  yo = k - q*sizes[0];
}

__device__ void contract3d_coords(int64_t &xo, int64_t &yo, int64_t &zo, int64_t &xw, int64_t &yw, int64_t &zw, int64_t k, const int64_t sizes[6]) {
  int64_t q = k / sizes[5];
  xw = k - q*sizes[5];

  k = q;
  q = k / sizes[4];

  yw = k - q*sizes[4];

  k = q;
  q = k / sizes[3];

  zw = k - q*sizes[3];

  k = q;
  q = k / sizes[2];

  xo = k - q*sizes[2];

  k = q;
  q = k / sizes[1];

  yo = k - q*sizes[1];

  k = q;
  q = k / sizes[0];

  zo = k - q*sizes[0];
}

template<typename RealType>
__global__ void contract2d_kernel(RealType *d_outData, const RealType *d_inData, Size<6> stOutSize, Size<2> stInSize, Size<2> stPadding) {
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;

  const int64_t i64Size = stOutSize.data[2]*stOutSize.data[3]*stOutSize.data[4]*stOutSize.data[5];

  if (i < stOutSize.data[0] && j < stOutSize.data[1] && k < i64Size) {
    int64_t xo=0, yo=0, xw=0, yw=0;

    contract2d_coords(xo, yo, xw, yw, k, stOutSize.data + 2);

    const int64_t yi = yo*stOutSize.data[4] - stPadding.data[0] + yw;
    const int64_t xi = xo*stOutSize.data[5] - stPadding.data[1] + xw;

    if (yi >= 0 && xi >= 0 && yi < stInSize.data[0] && xi < stInSize.data[1])
      d_outData[(i*stOutSize.data[1] + j)*i64Size + k] = d_inData[((i*stOutSize.data[1] + j)*stInSize.data[0] + yi)*stInSize.data[1] + xi];
  }
}

template<typename RealType>
__global__ void contract3d_kernel(RealType *d_outData, const RealType *d_inData, Size<8> stOutSize, Size<3> stInSize, Size<3> stPadding) {
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;

  const int64_t i64Size = stOutSize.data[2]*stOutSize.data[3]*stOutSize.data[4]*stOutSize.data[5]*stOutSize.data[6]*stOutSize.data[7];

  if (i < stOutSize.data[0] && j < stOutSize.data[1] && k < i64Size) {
    int64_t xo=0, yo=0, zo=0, xw=0, yw=0, zw=0;

    contract3d_coords(xo, yo, zo, xw, yw, zw, k, stOutSize.data + 2);

    const int64_t zi = zo*stOutSize.data[5] - stPadding.data[0] + zw;
    const int64_t yi = yo*stOutSize.data[6] - stPadding.data[1] + yw;
    const int64_t xi = xo*stOutSize.data[7] - stPadding.data[2] + xw;

    if (zi >= 0 && yi >= 0 && xi >= 0 && zi < stInSize.data[0] && yi < stInSize.data[1] && xi < stInSize.data[2])
      d_outData[(i*stOutSize.data[1] + j)*i64Size + k] = d_inData[(((i*stOutSize.data[1] + j)*stInSize.data[0] + zi)*stInSize.data[1] + yi)*stInSize.data[2] + xi];
  }
}

template<typename RealType>
__global__ void expand2d_kernel(RealType *d_outData, const RealType *d_inData, Size<4> stOutSize, Size<4> stInSize, Size<2> stPadding) {
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;

  const int64_t i64Size = stInSize.data[0]*stInSize.data[1]*stInSize.data[2]*stInSize.data[3];

  if (i < stOutSize.data[0] && j < stOutSize.data[1] && k < i64Size) {
    int64_t xi=0, yi=0, xw=0, yw=0;
    contract2d_coords(xi, yi, xw, yw, k, stInSize.data);

    const int64_t yo = yi*stInSize.data[2] - stPadding.data[0] + yw;
    const int64_t xo = xi*stInSize.data[3] - stPadding.data[1] + xw;

    if (yo >= 0 && xo >= 0 && yo < stOutSize.data[2] && xo < stOutSize.data[3])
      d_outData[((i*stOutSize.data[1] + j)*stOutSize.data[2] + yo)*stOutSize.data[3] + xo] = d_inData[(i*stOutSize.data[1] + j)*i64Size + k];
    
  }
}

template<typename RealType>
__global__ void expand3d_kernel(RealType *d_outData, const RealType *d_inData, Size<5> stOutSize, Size<6> stInSize, Size<3> stPadding) {
  const int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
  const int64_t j = (int64_t)blockIdx.z * blockDim.z + threadIdx.z;

  const int64_t i64Size = stInSize.data[0]*stInSize.data[1]*stInSize.data[2]*stInSize.data[3]*stInSize.data[4]*stInSize.data[5];

  if (i < stOutSize.data[0] && j < stOutSize.data[1] && k < i64Size) {
    int64_t xi=0, yi=0, zi=0, xw=0, yw=0, zw=0;
    contract3d_coords(xi, yi, zi, xw, yw, zw, k, stInSize.data);

    const int64_t zo = zi*stInSize.data[3] - stPadding.data[0] + zw;
    const int64_t yo = yi*stInSize.data[4] - stPadding.data[1] + yw;
    const int64_t xo = xi*stInSize.data[5] - stPadding.data[2] + xw;

    if (zo >= 0 && yo >= 0 && xo >= 0 && zo < stOutSize.data[2] && yo < stOutSize.data[3] && xo < stOutSize.data[4])
      d_outData[(((i*stOutSize.data[1] + j)*stOutSize.data[2] + zo)*stOutSize.data[3] + yo)*stOutSize.data[4] + xo] = d_inData[(i*stOutSize.data[1] + j)*i64Size + k];
    
  }
}

} // end anonymous namespace

template<typename RealType>
torch::Tensor contract2d_gpu(torch::Tensor inData, const int64_t a_i64Window[2], const int64_t a_i64Padding[2]) {
  if (inData.dim() != 4 || a_i64Padding[0] < 0 || a_i64Padding[1] < 0) {
    InvalidArgumentStream("Error: inData must have 4 dimensions. Padding must be non-negative.").raise();
    return torch::Tensor();
  }

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64Height = inData.sizes()[2];
  const int64_t i64Width = inData.sizes()[3];

  if (a_i64Window[0] < 1 || a_i64Window[1] < 1 || a_i64Window[0] > i64Height + 2*a_i64Padding[0] || a_i64Window[1] > i64Width + 2*a_i64Padding[1]) {
    InvalidArgumentStream("Error: Window dimensions must be at least 1 but no greater than the size of the padded image.").raise();
    return torch::Tensor();
  }

  Size<6> stOutSize;
  stOutSize.data[0] = inData.sizes()[0];
  stOutSize.data[1] = inData.sizes()[1];
  stOutSize.data[2] = (i64Height + 2*a_i64Padding[0] - a_i64Window[0])/a_i64Window[0] + 1;
  stOutSize.data[3] = (i64Width + 2*a_i64Padding[1] - a_i64Window[1])/a_i64Window[1] + 1;
  stOutSize.data[4] = a_i64Window[0];
  stOutSize.data[5] = a_i64Window[1];

  const int64_t i64Size =  stOutSize.data[2]* stOutSize.data[3]* stOutSize.data[4]* stOutSize.data[5];

  Size<2> stPadding;
  stPadding.data[0] = a_i64Padding[0];
  stPadding.data[1] = a_i64Padding[1];

  Size<2> stInSize;
  stInSize.data[0] = i64Height;
  stInSize.data[1] = i64Width;

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  torch::Tensor outData = torch::zeros(IntArrayRef(stOutSize.data, stOutSize.size), clOptions);

  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64Size + threadsPerBlock.x-1) / threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1) / threadsPerBlock.y, (i64NumChannels + threadsPerBlock.z-1) / threadsPerBlock.z);

  contract2d_kernel<<<numBlocks, threadsPerBlock>>>(outData.data_ptr<RealType>(), inData.data_ptr<RealType>(), stOutSize, stInSize, stPadding);

  return outData;
}

template<typename RealType>
torch::Tensor contract3d_gpu(torch::Tensor inData, const int64_t a_i64Window[3], const int64_t a_i64Padding[3]) {
  if (inData.dim() != 5 || a_i64Padding[0] < 0 || a_i64Padding[1] < 0 || a_i64Padding[2] < 0) {
    InvalidArgumentStream("Error: inData must have 5 dimensions. Padding must be non-negative.").raise();
    return torch::Tensor();
  }

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t i64Depth = inData.sizes()[2];
  const int64_t i64Height = inData.sizes()[3];
  const int64_t i64Width = inData.sizes()[4];

  if (a_i64Window[0] < 1 || a_i64Window[1] < 1 || a_i64Window[2] < 1 || a_i64Window[0] > i64Depth + 2*a_i64Padding[0] || a_i64Window[1] > i64Height + 2*a_i64Padding[1] || a_i64Window[2] > i64Width + 2*a_i64Padding[2]) {
    InvalidArgumentStream("Error: Window dimensions must be at least 1 but no greater than the size of the padded image.").raise();
    return torch::Tensor();
  }

  Size<8> stOutSize;
  stOutSize.data[0] = inData.sizes()[0];
  stOutSize.data[1] = inData.sizes()[1];
  stOutSize.data[2] = (i64Depth + 2*a_i64Padding[0] - a_i64Window[0])/a_i64Window[0] + 1;
  stOutSize.data[3] = (i64Height + 2*a_i64Padding[1] - a_i64Window[1])/a_i64Window[1] + 1;
  stOutSize.data[4] = (i64Width + 2*a_i64Padding[2] - a_i64Window[2])/a_i64Window[2] + 1;
  stOutSize.data[5] = a_i64Window[0];
  stOutSize.data[6] = a_i64Window[1];
  stOutSize.data[7] = a_i64Window[2];

  const int64_t i64Size =  stOutSize.data[2]* stOutSize.data[3]* stOutSize.data[4]* stOutSize.data[5] * stOutSize.data[6] * stOutSize.data[7];

  Size<3> stPadding;
  stPadding.data[0] = a_i64Padding[0];
  stPadding.data[1] = a_i64Padding[1];
  stPadding.data[2] = a_i64Padding[2];

  Size<3> stInSize;
  stInSize.data[0] = i64Depth;
  stInSize.data[1] = i64Height;
  stInSize.data[2] = i64Width;

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  torch::Tensor outData = torch::zeros(IntArrayRef(stOutSize.data, stOutSize.size), clOptions);

  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64Size + threadsPerBlock.x-1) / threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1) / threadsPerBlock.y, (i64NumChannels + threadsPerBlock.z-1) / threadsPerBlock.z);

  contract3d_kernel<<<numBlocks, threadsPerBlock>>>(outData.data_ptr<RealType>(), inData.data_ptr<RealType>(), stOutSize, stInSize, stPadding);

  return outData;
}

template<typename RealType>
torch::Tensor expand2d_gpu(torch::Tensor inData, const int64_t a_i64Padding[2]) {
  if (inData.dim() != 6 || a_i64Padding[0] < 0 || a_i64Padding[1] < 0) {
    InvalidArgumentStream("Error: inData must have 6 dimensions. Padding must be non-negative.").raise();
    return torch::Tensor();
  }

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t a_i64Window[2] = { inData.sizes()[4], inData.sizes()[5] };
  const int64_t i64Height = inData.sizes()[2]*a_i64Window[0] - ((2*a_i64Padding[0])/a_i64Window[0])*a_i64Window[0];
  const int64_t i64Width = inData.sizes()[3]*a_i64Window[1] - ((2*a_i64Padding[1])/a_i64Window[1])*a_i64Window[1];

  if (i64Height < 1 || i64Width < 1 ) {
    InvalidArgumentStream("Error: The expanded size of the image must be non-empty.").raise();
    return torch::Tensor();
  }
 
  Size<4> stInSize;
  stInSize.data[0] = inData.sizes()[2];
  stInSize.data[1] = inData.sizes()[3];
  stInSize.data[2] = inData.sizes()[4];
  stInSize.data[3] = inData.sizes()[5];

  const int64_t i64Size =  stInSize.data[0]* stInSize.data[1] * stInSize.data[2]* stInSize.data[3];

  Size<2> stPadding;
  stPadding.data[0] = a_i64Padding[0];
  stPadding.data[1] = a_i64Padding[1];

  Size<4> stOutSize;
  stOutSize.data[0] = inData.sizes()[0];   
  stOutSize.data[1] = inData.sizes()[1];
  stOutSize.data[2] = i64Height;
  stOutSize.data[3] = i64Width;

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  torch::Tensor outData = torch::zeros(IntArrayRef(stOutSize.data, stOutSize.size), clOptions);

  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64Size + threadsPerBlock.x-1) / threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1) / threadsPerBlock.y, (i64NumChannels + threadsPerBlock.z-1) / threadsPerBlock.z);

  expand2d_kernel<<<numBlocks, threadsPerBlock>>>(outData.data_ptr<RealType>(), inData.data_ptr<RealType>(), stOutSize, stInSize, stPadding);

  return outData;
}

template<typename RealType>
torch::Tensor expand3d_gpu(torch::Tensor inData, const int64_t a_i64Padding[2]) {
  if (inData.dim() != 8 || a_i64Padding[0] < 0 || a_i64Padding[1] < 0 || a_i64Padding[2] < 0) {
    InvalidArgumentStream("Error: inData must have 8 dimensions. Padding must be non-negative.").raise();
    return torch::Tensor();
  }

  const int64_t i64BatchSize = inData.sizes()[0];
  const int64_t i64NumChannels = inData.sizes()[1];
  const int64_t a_i64Window[3] = { inData.sizes()[5], inData.sizes()[6], inData.sizes()[7] };
  const int64_t i64Depth = inData.sizes()[2]*a_i64Window[0] - ((2*a_i64Padding[0])/a_i64Window[0])*a_i64Window[0];
  const int64_t i64Height = inData.sizes()[3]*a_i64Window[1] - ((2*a_i64Padding[1])/a_i64Window[1])*a_i64Window[1];
  const int64_t i64Width = inData.sizes()[4]*a_i64Window[2] - ((2*a_i64Padding[2])/a_i64Window[2])*a_i64Window[2];

  if (i64Depth < 1 || i64Height < 1 || i64Width < 1) {
    InvalidArgumentStream("Error: The expanded size of the image must be non-empty.").raise();
    return torch::Tensor();
  }

  Size<6> stInSize;
  stInSize.data[0] = inData.sizes()[2];
  stInSize.data[1] = inData.sizes()[3];
  stInSize.data[2] = inData.sizes()[4];
  stInSize.data[3] = inData.sizes()[5];
  stInSize.data[4] = inData.sizes()[6];
  stInSize.data[5] = inData.sizes()[7];

  const int64_t i64Size =  stInSize.data[0]* stInSize.data[1] * stInSize.data[2]* stInSize.data[3] * stInSize.data[4] * stInSize.data[5];

  Size<3> stPadding;
  stPadding.data[0] = a_i64Padding[0];
  stPadding.data[1] = a_i64Padding[1];
  stPadding.data[2] = a_i64Padding[2];

  Size<5> stOutSize;
  stOutSize.data[0] = inData.sizes()[0];   
  stOutSize.data[1] = inData.sizes()[1];
  stOutSize.data[2] = i64Depth;
  stOutSize.data[3] = i64Height;
  stOutSize.data[4] = i64Width;

  auto clOptions = torch::TensorOptions().dtype(inData.dtype()).device(inData.device());

  torch::Tensor outData = torch::zeros(IntArrayRef(stOutSize.data, stOutSize.size), clOptions);

  const dim3 threadsPerBlock(16,8,8);
  const dim3 numBlocks((i64Size + threadsPerBlock.x-1) / threadsPerBlock.x, (i64BatchSize + threadsPerBlock.y-1) / threadsPerBlock.y, (i64NumChannels + threadsPerBlock.z-1) / threadsPerBlock.z);

  expand3d_kernel<<<numBlocks, threadsPerBlock>>>(outData.data_ptr<RealType>(), inData.data_ptr<RealType>(), stOutSize, stInSize, stPadding);

  return outData;
}

template torch::Tensor contract2d_gpu<uint8_t>(torch::Tensor, const int64_t *, const int64_t *);
template torch::Tensor expand2d_gpu<uint8_t>(torch::Tensor, const int64_t *);
template torch::Tensor contract2d_gpu<int8_t>(torch::Tensor, const int64_t *, const int64_t *);
template torch::Tensor expand2d_gpu<int8_t>(torch::Tensor, const int64_t *);
template torch::Tensor contract2d_gpu<int16_t>(torch::Tensor, const int64_t *, const int64_t *);
template torch::Tensor expand2d_gpu<int16_t>(torch::Tensor, const int64_t *);
template torch::Tensor contract2d_gpu<int32_t>(torch::Tensor, const int64_t *, const int64_t *);
template torch::Tensor expand2d_gpu<int32_t>(torch::Tensor, const int64_t *);
template torch::Tensor contract2d_gpu<int64_t>(torch::Tensor, const int64_t *, const int64_t *);
template torch::Tensor expand2d_gpu<int64_t>(torch::Tensor, const int64_t *);
template torch::Tensor contract2d_gpu<float>(torch::Tensor, const int64_t *, const int64_t *);
template torch::Tensor expand2d_gpu<float>(torch::Tensor, const int64_t *);
template torch::Tensor contract2d_gpu<double>(torch::Tensor, const int64_t *, const int64_t *);
template torch::Tensor expand2d_gpu<double>(torch::Tensor, const int64_t *);

template torch::Tensor contract3d_gpu<uint8_t>(torch::Tensor, const int64_t *, const int64_t *);
template torch::Tensor expand3d_gpu<uint8_t>(torch::Tensor, const int64_t *);
template torch::Tensor contract3d_gpu<int8_t>(torch::Tensor, const int64_t *, const int64_t *);
template torch::Tensor expand3d_gpu<int8_t>(torch::Tensor, const int64_t *);
template torch::Tensor contract3d_gpu<int16_t>(torch::Tensor, const int64_t *, const int64_t *);
template torch::Tensor expand3d_gpu<int16_t>(torch::Tensor, const int64_t *);
template torch::Tensor contract3d_gpu<int32_t>(torch::Tensor, const int64_t *, const int64_t *);
template torch::Tensor expand3d_gpu<int32_t>(torch::Tensor, const int64_t *);
template torch::Tensor contract3d_gpu<int64_t>(torch::Tensor, const int64_t *, const int64_t *);
template torch::Tensor expand3d_gpu<int64_t>(torch::Tensor, const int64_t *);
template torch::Tensor contract3d_gpu<float>(torch::Tensor, const int64_t *, const int64_t *);
template torch::Tensor expand3d_gpu<float>(torch::Tensor, const int64_t *);
template torch::Tensor contract3d_gpu<double>(torch::Tensor, const int64_t *, const int64_t *);
template torch::Tensor expand3d_gpu<double>(torch::Tensor, const int64_t *);

