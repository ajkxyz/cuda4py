"""
Copyright (c) 2014, Samsung Electronics Co.,Ltd.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of Samsung Electronics Co.,Ltd..
"""

"""
cuda4py - CUDA cffi bindings and helper classes.
URL: https://github.com/ajkxyz/cuda4py
Original author: Alexey Kazantsev <a.kazantsev@samsung.com>
"""

"""
Init module.
"""

from cuda4py import _cffi
from cuda4py._cffi import (initialize,

                           CU_CTX_SCHED_AUTO,
                           CU_CTX_SCHED_SPIN,
                           CU_CTX_SCHED_YIELD,
                           CU_CTX_SCHED_BLOCKING_SYNC,
                           CU_CTX_MAP_HOST,
                           CU_CTX_LMEM_RESIZE_TO_MAX,

                           CU_MEMHOSTALLOC_PORTABLE,
                           CU_MEMHOSTALLOC_DEVICEMAP,
                           CU_MEMHOSTALLOC_WRITECOMBINED,

                           CU_MEM_ATTACH_GLOBAL,
                           CU_MEM_ATTACH_HOST,
                           CU_MEM_ATTACH_SINGLE,

                           CUDA_SUCCESS,
                           CUDA_ERROR_INVALID_VALUE,
                           CUDA_ERROR_OUT_OF_MEMORY,
                           CUDA_ERROR_NOT_INITIALIZED,
                           CUDA_ERROR_DEINITIALIZED,
                           CUDA_ERROR_PROFILER_DISABLED,
                           CUDA_ERROR_PROFILER_NOT_INITIALIZED,
                           CUDA_ERROR_PROFILER_ALREADY_STARTED,
                           CUDA_ERROR_PROFILER_ALREADY_STOPPED,
                           CUDA_ERROR_NO_DEVICE,
                           CUDA_ERROR_INVALID_DEVICE,
                           CUDA_ERROR_INVALID_IMAGE,
                           CUDA_ERROR_INVALID_CONTEXT,
                           CUDA_ERROR_CONTEXT_ALREADY_CURRENT,
                           CUDA_ERROR_MAP_FAILED,
                           CUDA_ERROR_UNMAP_FAILED,
                           CUDA_ERROR_ARRAY_IS_MAPPED,
                           CUDA_ERROR_ALREADY_MAPPED,
                           CUDA_ERROR_NO_BINARY_FOR_GPU,
                           CUDA_ERROR_ALREADY_ACQUIRED,
                           CUDA_ERROR_NOT_MAPPED,
                           CUDA_ERROR_NOT_MAPPED_AS_ARRAY,
                           CUDA_ERROR_NOT_MAPPED_AS_POINTER,
                           CUDA_ERROR_ECC_UNCORRECTABLE,
                           CUDA_ERROR_UNSUPPORTED_LIMIT,
                           CUDA_ERROR_CONTEXT_ALREADY_IN_USE,
                           CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
                           CUDA_ERROR_INVALID_PTX,
                           CUDA_ERROR_INVALID_GRAPHICS_CONTEXT,
                           CUDA_ERROR_INVALID_SOURCE,
                           CUDA_ERROR_FILE_NOT_FOUND,
                           CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND,
                           CUDA_ERROR_SHARED_OBJECT_INIT_FAILED,
                           CUDA_ERROR_OPERATING_SYSTEM,
                           CUDA_ERROR_INVALID_HANDLE,
                           CUDA_ERROR_NOT_FOUND,
                           CUDA_ERROR_NOT_READY,
                           CUDA_ERROR_ILLEGAL_ADDRESS,
                           CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES,
                           CUDA_ERROR_LAUNCH_TIMEOUT,
                           CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING,
                           CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED,
                           CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
                           CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE,
                           CUDA_ERROR_CONTEXT_IS_DESTROYED,
                           CUDA_ERROR_ASSERT,
                           CUDA_ERROR_TOO_MANY_PEERS,
                           CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED,
                           CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED,
                           CUDA_ERROR_HARDWARE_STACK_ERROR,
                           CUDA_ERROR_ILLEGAL_INSTRUCTION,
                           CUDA_ERROR_MISALIGNED_ADDRESS,
                           CUDA_ERROR_INVALID_ADDRESS_SPACE,
                           CUDA_ERROR_INVALID_PC,
                           CUDA_ERROR_LAUNCH_FAILED,
                           CUDA_ERROR_NOT_PERMITTED,
                           CUDA_ERROR_NOT_SUPPORTED,
                           CUDA_ERROR_UNKNOWN)

from cuda4py._py import (CUDARuntimeError,
                         CU,
                         Memory,
                         MemAlloc,
                         MemAllocManaged,
                         MemHostAlloc,
                         skip,
                         Function,
                         Module,
                         Context,
                         Device,
                         Devices)


def get_ffi():
    """Returns CFFI() instance for the loaded shared library.
    """
    return _cffi.ffi
