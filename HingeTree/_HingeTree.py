# 
# Nathan Lay
# AI Resource at National Cancer Institute
# National Institutes of Health
# November 2020
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 

import torch
import torch.nn as nn
import torch.autograd
import hingetree_cpp

def _is_deterministic():
    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        return torch.are_deterministic_algorithms_enabled()
    elif hasattr(torch, "is_deterministic"):
        return torch.is_deterministic()

    raise RuntimeError("Unable to query torch deterministic mode.")

def contract(inData, window, padding=0):
  if isinstance(window, int):
    window = [window]*(inData.dim()-2)

  if isinstance(padding, int):
    padding = [padding]*(inData.dim()-2)

  return hingetree_cpp.contract(inData, window, padding)

def expand(inData, padding=0):
  if isinstance(padding, int):
    padding = [padding]*((inData.dim()-2)//2)

  return hingetree_cpp.expand(inData, padding)

class _Expand(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, padding):
        dim = (inData.dim()-2)//2
        ctx.window = list(inData.shape[-dim:])
        ctx.padding = padding
        return expand(inData, padding)

    @staticmethod
    def backward(ctx, outDataGrad):
        if ctx.needs_input_grad[0]:
            return contract(outDataGrad.contiguous(), ctx.window, ctx.padding), None

        return None, None

class _Contract(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, window, padding=0):
        ctx.padding = padding
        return contract(inData, window, padding)

    @staticmethod
    def backward(ctx, outDataGrad):
        if ctx.needs_input_grad[0]:
            return expand(outDataGrad.contiguous(), ctx.padding), None, None

        return None, None, None

class Expand(torch.nn.Module):
    def __init__(self, padding=0):
        super().__init__()
        self.padding = padding

    def forward(self, inData):
        return _Expand.apply(inData, self.padding)

class Contract(torch.nn.Module):
    def __init__(self, window, padding=0):
        super().__init__()
        self.window = window
        self.padding = padding

    def forward(self, inData):
        return _Contract.apply(inData, self.window, self.padding)

class HingeTree(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        return hingetree_cpp.tree_forward(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def backward(ctx, outDataGrad):
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        if _is_deterministic():
            inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.tree_backward_deterministic(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())
        else:
            inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.tree_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad

    @staticmethod
    def check_thresholds(inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_check_thresholds(inThresholds, inOrdinals, inWeights)

    @staticmethod
    def fix_thresholds(inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_fix_thresholds(inThresholds, inOrdinals, inWeights)

    @staticmethod
    def reachability(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_reachability(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def leafmap(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_leafmap(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def marginmap(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_marginmap(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def speedtest(inData):
        return hingetree_cpp.tree_speedtest(inData, _is_deterministic())

    @staticmethod
    def init_medians(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_init_medians(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def init_greedy(inData, inLabels, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.tree_init_greedy(inData, inLabels, inThresholds, inOrdinals, inWeights)

class HingeFern(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        return hingetree_cpp.fern_forward(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def backward(ctx, outDataGrad):
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        if _is_deterministic():
            inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.fern_backward_deterministic(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())
        else:
            inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.fern_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad
     
    @staticmethod
    def check_thresholds(inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.fern_check_thresholds(inThresholds, inOrdinals, inWeights)

    @staticmethod
    def fix_thresholds(inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.fern_fix_thresholds(inThresholds, inOrdinals, inWeights)

    @staticmethod
    def reachability(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.fern_reachability(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def leafmap(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.fern_leafmap(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def marginmap(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.fern_marginmap(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def speedtest(inData):
        return hingetree_cpp.fern_speedtest(inData, _is_deterministic())

    @staticmethod
    def init_medians(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.fern_init_medians(inData, inThresholds, inOrdinals, inWeights)

class SparseHingeTree(HingeTree):
    @staticmethod
    def backward(ctx, outDataGrad):
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors
        
        if _is_deterministic() and inData.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of sparse hinge tree backward on GPUs.")

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.sparse_tree_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())
        
        if inDataGrad is not None:
            inDataGrad = inDataGrad.to_dense()

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad
        
class SparseHingeFern(HingeFern):
    @staticmethod
    def backward(ctx, outDataGrad):
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors
        
        if _is_deterministic() and inData.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of sparse hinge fern backward on GPUs.")

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.sparse_fern_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())
        
        if inDataGrad is not None:
            inDataGrad = inDataGrad.to_dense()

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad

class HingeTreeFusedLinear(HingeTree):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias):
        if _is_deterministic() and inData.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of forward for hinge tree + linear on GPUs.")

        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias)

        return hingetree_cpp.tree_fused_linear_forward(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge tree + linear on GPUs.")

        inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, inLinearWeightsGrad, inLinearBiasGrad = hingetree_cpp.tree_fused_linear_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], inLinearWeights, ctx.needs_input_grad[4], inLinearBias, ctx.needs_input_grad[5], outDataGrad.contiguous())

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, inLinearWeightsGrad, inLinearBiasGrad

class HingeFernFusedLinear(HingeFern):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias):
        if _is_deterministic() and inData.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of forward for hinge fern + linear on GPUs.")

        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias)

        return hingetree_cpp.fern_fused_linear_forward(inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge fern + linear on GPUs.")

        inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, inLinearWeightsGrad, inLinearBiasGrad = hingetree_cpp.fern_fused_linear_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], inLinearWeights, ctx.needs_input_grad[4], inLinearBias, ctx.needs_input_grad[5], outDataGrad.contiguous())

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, inLinearWeightsGrad, inLinearBiasGrad

class SparseHingeTreeFusedLinear(HingeTreeFusedLinear):
    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge tree + linear on GPUs.")

        inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, inLinearWeightsGrad, inLinearBiasGrad = hingetree_cpp.sparse_tree_fused_linear_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], inLinearWeights, ctx.needs_input_grad[4], inLinearBias, ctx.needs_input_grad[5], outDataGrad.contiguous())

        if inDataGrad is not None:
            inDataGrad = inDataGrad.to_dense()

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, inLinearWeightsGrad, inLinearBiasGrad

class SparseHingeFernFusedLinear(HingeFernFusedLinear):
    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge fern + linear on GPUs.")

        inData, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, inLinearWeightsGrad, inLinearBiasGrad = hingetree_cpp.sparse_fern_fused_linear_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], inLinearWeights, ctx.needs_input_grad[4], inLinearBias, ctx.needs_input_grad[5], outDataGrad.contiguous())
        
        if inDataGrad is not None:
            inDataGrad = inDataGrad.to_dense()

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, inLinearWeightsGrad, inLinearBiasGrad

class HingeTreeFusion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inImg, inVec, inThresholds, inOrdinals, inWeights):
        ctx.save_for_backward(inImg, inVec, inThresholds, inOrdinals, inWeights)

        return hingetree_cpp.tree_fusion_forward(inImg, inVec, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge tree fusion on GPUs.")

        inImg, inVec, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inImgGrad, inVecGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.tree_fusion_backward(inImg, ctx.needs_input_grad[0], inVec, ctx.needs_input_grad[1], inThresholds, ctx.needs_input_grad[2], inOrdinals, ctx.needs_input_grad[3], inWeights, ctx.needs_input_grad[4], outDataGrad.contiguous())

        return inImgGrad, inVecGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad

class HingeFernFusion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inImg, inVec, inThresholds, inOrdinals, inWeights):
        ctx.save_for_backward(inImg, inVec, inThresholds, inOrdinals, inWeights)

        return hingetree_cpp.fern_fusion_forward(inImg, inVec, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge fern fusion on GPUs.")

        inImg, inVec, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inImgGrad, inVecGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.fern_fusion_backward(inImg, ctx.needs_input_grad[0], inVec, ctx.needs_input_grad[1], inThresholds, ctx.needs_input_grad[2], inOrdinals, ctx.needs_input_grad[3], inWeights, ctx.needs_input_grad[4], outDataGrad.contiguous())

        return inImgGrad, inVecGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad

class HingeTreeFusionFusedLinear(HingeTreeFusion):
    @staticmethod
    def forward(ctx, inImg, inVec, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias):
        if _is_deterministic() and inImg.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of forward for hinge tree fusion + linear on GPUs.")

        ctx.save_for_backward(inImg, inVec, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias)

        return hingetree_cpp.tree_fusion_fused_linear_forward(inImg, inVec, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge tree fusion + linear on GPUs.")

        inImg, inVec, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias = ctx.saved_tensors

        inImgGrad, inVecGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, inLinearWeightsGrad, inLinearBiasGrad = hingetree_cpp.tree_fusion_fused_linear_backward(inImg, ctx.needs_input_grad[0], inVec, ctx.needs_input_grad[1], inThresholds, ctx.needs_input_grad[2], inOrdinals, ctx.needs_input_grad[3], inWeights, ctx.needs_input_grad[4], inLinearWeights, ctx.needs_input_grad[5], inLinearBias, ctx.needs_input_grad[6], outDataGrad.contiguous())

        return inImgGrad, inVec, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, inLinearWeightsGrad, inLinearBiasGrad

class HingeFernFusionFusedLinear(HingeFernFusion):
    @staticmethod
    def forward(ctx, inImg, inVec, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias):
        if _is_deterministic() and inImg.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of forward for hinge fern fusion + linear on GPUs.")

        ctx.save_for_backward(inImg, inVec, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias)

        return hingetree_cpp.fern_fusion_fused_linear_forward(inImg, inVec, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge fern fusion + linear on GPUs.")

        inImgGrad, inVecGrad, inThresholds, inOrdinals, inWeights, inLinearWeights, inLinearBias = ctx.saved_tensors

        inImgGrad, inVecGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, inLinearWeightsGrad, inLinearBiasGrad = hingetree_cpp.fern_fusion_fused_linear_backward(inImg, ctx.needs_input_grad[0], inVec, ctx.needs_input_grad[1], inThresholds, ctx.needs_input_grad[2], inOrdinals, ctx.needs_input_grad[3], inWeights, ctx.needs_input_grad[4], inLinearWeights, ctx.needs_input_grad[5], inLinearBias, ctx.needs_input_grad[6], outDataGrad.contiguous())

        return inImgGrad, inVecGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, inLinearWeightsGrad, inLinearBiasGrad

# Convolution operations below

class HingeTreeConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        ctx.kernelSize = kernelSize
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return hingetree_cpp.tree_conv1d_forward(inData, inThresholds, inOrdinals, inWeights, ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge tree convolution on GPUs.")
    
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.tree_conv1d_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous(), ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, None, None, None, None

class HingeTreeConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        ctx.kernelSize = kernelSize
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return hingetree_cpp.tree_conv2d_forward(inData, inThresholds, inOrdinals, inWeights, ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge tree convolution on GPUs.")
    
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.tree_conv2d_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous(), ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, None, None, None, None

class HingeTreeConv3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        ctx.kernelSize = kernelSize
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return hingetree_cpp.tree_conv3d_forward(inData, inThresholds, inOrdinals, inWeights, ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge tree convolution on GPUs.")
    
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.tree_conv3d_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous(), ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, None, None, None, None

class HingeFernConv1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        ctx.kernelSize = kernelSize
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return hingetree_cpp.fern_conv1d_forward(inData, inThresholds, inOrdinals, inWeights, ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge fern convolution on GPUs.")
    
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.fern_conv1d_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous(), ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, None, None, None, None

class HingeFernConv2d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        ctx.kernelSize = kernelSize
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return hingetree_cpp.fern_conv2d_forward(inData, inThresholds, inOrdinals, inWeights, ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge fern convolution on GPUs.")
        
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors
        
        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.fern_conv2d_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous(), ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, None, None, None, None

class HingeFernConv3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights, kernelSize, stride, padding, dilation):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        ctx.kernelSize = kernelSize
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation

        return hingetree_cpp.fern_conv3d_forward(inData, inThresholds, inOrdinals, inWeights, ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation for hinge fern convolution on GPUs.")
        
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors
        
        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.fern_conv3d_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous(), ctx.kernelSize, ctx.stride, ctx.padding, ctx.dilation)

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad, None, None, None, None

class HingeTrie(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inData, inThresholds, inOrdinals, inWeights):
        ctx.save_for_backward(inData, inThresholds, inOrdinals, inWeights)

        return hingetree_cpp.trie_forward(inData, inThresholds, inOrdinals, inWeights)

    @staticmethod
    def backward(ctx, outDataGrad):
        if _is_deterministic() and outDataGrad.device.type != "cpu":
            raise RuntimeError("No deterministic implementation of backpropagation of hinge trie on GPUs.")
    
        inData, inThresholds, inOrdinals, inWeights = ctx.saved_tensors

        inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad = hingetree_cpp.trie_backward(inData, ctx.needs_input_grad[0], inThresholds, ctx.needs_input_grad[1], inOrdinals, ctx.needs_input_grad[2], inWeights, ctx.needs_input_grad[3], outDataGrad.contiguous())

        return inDataGrad, inThresholdsGrad, inOrdinalsGrad, inWeightsGrad

    @staticmethod
    def init_medians(inData, inThresholds, inOrdinals, inWeights):
        return hingetree_cpp.trie_init_medians(inData, inThresholds, inOrdinals, inWeights)


