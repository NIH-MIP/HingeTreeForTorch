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

import os
from setuptools import setup, Extension
import torch
from torch.utils import cpp_extension

sourceFiles = [ 'hingetree.cpp', 'hingetree_sparse.cpp', 'hingetrie.cpp', 'ImageToMatrix.cpp', 'hingetree_conv.cpp', 'hingetree_fused_linear.cpp', 'hingetree_fusion.cpp', 'expand.cpp', 'Timer.cpp' ]
extraCflags = [ '-O2' ]
extraCudaFlags = [ '-O2' ]

# Check if PyTorch is compiled against CUDA and build for all of PyTorch's supported architectures.
# Override this behavior by defining TORCH_CUDA_ARCH_LIST yourself
def should_build_cuda():
    try:
        # Is this built with CUDA?
        #arch_list = torch.cuda.get_arch_list()

        # XXX: Bypass torch.cuda.get_arch_list() since it checks torch.cuda.is_available() which requires a GPU to be present.
        # Why does a GPU need to be present to know about compile-time configurations?
        arch_flags = torch._C._cuda_getArchFlags()

        if arch_flags is None:
            return False

        arch_list = arch_flags.split()
    except AttributeError: 
        return False

    if len(arch_list) == 0:
        return False

    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        return True
    
    # compute capability version --> sm, compute
    arch_map = dict()
    
    # Translate sm_## and compute_## to TORCH_CUDA_ARCH_LIST ... so that PyTorch can translate them back!
    for arch in arch_list:
        tokens = arch.split("_")
        if len(tokens) == 2:
            compute_type, compute_cap = tokens

            # NOTE: compute_cap may end with 'a'. See _get_cuda_arch_flags in torch.utils.cpp_extension
            if compute_cap.endswith("a"):
                compute_cap = compute_cap[:-2] + "." + compute_cap[-2:]
            else:
                compute_cap = compute_cap[:-1] + "." + compute_cap[-1:]

            arch_map.setdefault(compute_cap, set()).add(compute_type)

    env_value = " ".join([ key + "+PTX" if "compute" in values else key for key, values in arch_map.items() ])
    os.environ["TORCH_CUDA_ARCH_LIST"] = env_value
    print(f"TORCH_CUDA_ARCH_LIST = {env_value}")

    return True

if should_build_cuda():
    sourceFiles.append('hingetree_gpu.cu')
    sourceFiles.append('hingetree_sparse_gpu.cu')
    sourceFiles.append('ImageToMatrix_gpu.cu')
    sourceFiles.append('hingetree_conv_gpu.cu')
    sourceFiles.append('hingetree_fused_linear_gpu.cu')
    sourceFiles.append('hingetree_fusion_gpu.cu')
    sourceFiles.append('expand_gpu.cu')
    extraCflags.append('-DWITH_CUDA=1')
    extraCudaFlags.append('-DWITH_CUDA=1')

    setup(name='hingetree_cpp', 
        version='1.1.3',
        description='Port of random hinge forest for PyTorch.',
        author='Nathan Lay',
        author_email='enslay@gmail.com',
        url='https://github.com/nslay/HingeTreeForTorch/',
        packages=["HingeTree", "RandomHingeForest"],
        ext_modules=[cpp_extension.CUDAExtension(name = 'hingetree_cpp', sources = sourceFiles, extra_compile_args = {'cxx': extraCflags, 'nvcc': extraCudaFlags})],
        cmdclass={'build_ext': cpp_extension.BuildExtension})
else:
    setup(name='hingetree_cpp', 
        version='1.1.3',
        description='Port of random hinge forest for PyTorch.',
        author='Nathan Lay',
        author_email='enslay@gmail.com',
        url='https://github.com/nslay/HingeTreeForTorch/',
        packages=["HingeTree", "RandomHingeForest"],
        ext_modules=[cpp_extension.CppExtension(name = 'hingetree_cpp', sources = sourceFiles, extra_compile_args = {'cxx': extraCflags, 'nvcc': extraCudaFlags})],
        cmdclass={'build_ext': cpp_extension.BuildExtension})

