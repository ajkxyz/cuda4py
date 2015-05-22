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
cffi bindings.
"""
import cffi
import threading


#: ffi parser
ffi = None


#: Loaded shared library
lib = None


#: Lock
lock = threading.Lock()


#: Error codes
CUDA_SUCCESS = 0
CUDA_ERROR_INVALID_VALUE = 1
CUDA_ERROR_OUT_OF_MEMORY = 2
CUDA_ERROR_NOT_INITIALIZED = 3
CUDA_ERROR_DEINITIALIZED = 4
CUDA_ERROR_PROFILER_DISABLED = 5
CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6
CUDA_ERROR_PROFILER_ALREADY_STARTED = 7
CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8
CUDA_ERROR_NO_DEVICE = 100
CUDA_ERROR_INVALID_DEVICE = 101
CUDA_ERROR_INVALID_IMAGE = 200
CUDA_ERROR_INVALID_CONTEXT = 201
CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202
CUDA_ERROR_MAP_FAILED = 205
CUDA_ERROR_UNMAP_FAILED = 206
CUDA_ERROR_ARRAY_IS_MAPPED = 207
CUDA_ERROR_ALREADY_MAPPED = 208
CUDA_ERROR_NO_BINARY_FOR_GPU = 209
CUDA_ERROR_ALREADY_ACQUIRED = 210
CUDA_ERROR_NOT_MAPPED = 211
CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212
CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213
CUDA_ERROR_ECC_UNCORRECTABLE = 214
CUDA_ERROR_UNSUPPORTED_LIMIT = 215
CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216
CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217
CUDA_ERROR_INVALID_PTX = 218
CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219
CUDA_ERROR_INVALID_SOURCE = 300
CUDA_ERROR_FILE_NOT_FOUND = 301
CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302
CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303
CUDA_ERROR_OPERATING_SYSTEM = 304
CUDA_ERROR_INVALID_HANDLE = 400
CUDA_ERROR_NOT_FOUND = 500
CUDA_ERROR_NOT_READY = 600
CUDA_ERROR_ILLEGAL_ADDRESS = 700
CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701
CUDA_ERROR_LAUNCH_TIMEOUT = 702
CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705
CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708
CUDA_ERROR_CONTEXT_IS_DESTROYED = 709
CUDA_ERROR_ASSERT = 710
CUDA_ERROR_TOO_MANY_PEERS = 711
CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712
CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713
CUDA_ERROR_HARDWARE_STACK_ERROR = 714
CUDA_ERROR_ILLEGAL_INSTRUCTION = 715
CUDA_ERROR_MISALIGNED_ADDRESS = 716
CUDA_ERROR_INVALID_ADDRESS_SPACE = 717
CUDA_ERROR_INVALID_PC = 718
CUDA_ERROR_LAUNCH_FAILED = 719
CUDA_ERROR_NOT_PERMITTED = 800
CUDA_ERROR_NOT_SUPPORTED = 801
CUDA_ERROR_UNKNOWN = 999


#: CUdevice_attribute
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3
CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6
CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8
CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK = 8
CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9
CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10
CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12
CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK = 12
CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13
CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14
CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15
CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17
CU_DEVICE_ATTRIBUTE_INTEGRATED = 18
CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19
CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH = 27
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT = 28
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES = 29
CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30
CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31
CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32
CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33
CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34
CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35
CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36
CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37
CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38
CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39
CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40
CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43
CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49
CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50
CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67
CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77
CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78
CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79
CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81
CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82
CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83
CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84
CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85


#: CUctx_flags
CU_CTX_SCHED_AUTO = 0x00
CU_CTX_SCHED_SPIN = 0x01
CU_CTX_SCHED_YIELD = 0x02
CU_CTX_SCHED_BLOCKING_SYNC = 0x04
CU_CTX_BLOCKING_SYNC = 0x04
CU_CTX_SCHED_MASK = 0x07
CU_CTX_MAP_HOST = 0x08
CU_CTX_LMEM_RESIZE_TO_MAX = 0x10
CU_CTX_FLAGS_MASK = 0x1f


#: Memory flags
CU_MEMHOSTALLOC_PORTABLE = 0x01
CU_MEMHOSTALLOC_DEVICEMAP = 0x02
CU_MEMHOSTALLOC_WRITECOMBINED = 0x04

CU_MEM_ATTACH_GLOBAL = 0x1
CU_MEM_ATTACH_HOST = 0x2
CU_MEM_ATTACH_SINGLE = 0x4


#: CUmemorytype
CU_MEMORYTYPE_HOST = 0x01
CU_MEMORYTYPE_DEVICE = 0x02
CU_MEMORYTYPE_ARRAY = 0x03
CU_MEMORYTYPE_UNIFIED = 0x04


def _initialize(backends):
    global lib
    if lib is not None:
        return
    # C function definitions (http://docs.nvidia.com/cuda/cuda-driver-api/)
    # size_t instead of void* is used
    # for convinience with python calls and numpy arrays.
    src = """
    typedef int CUresult;
    typedef int CUdevice;
    typedef size_t CUcontext;
    typedef size_t CUmodule;
    typedef size_t CUfunction;
    typedef size_t CUstream;
    typedef size_t CUdeviceptr;
    typedef int CUdevice_attribute;
    typedef size_t (*CUoccupancyB2DSize)(int blockSize);
    typedef int CUmemorytype;
    typedef size_t CUarray;

    typedef struct CUDA_MEMCPY3D_st {
        size_t srcXInBytes;
        size_t srcY;
        size_t srcZ;
        size_t srcLOD;
        CUmemorytype srcMemoryType;
        size_t srcHost;
        CUdeviceptr srcDevice;
        CUarray srcArray;
        void *reserved0;
        size_t srcPitch;
        size_t srcHeight;

        size_t dstXInBytes;
        size_t dstY;
        size_t dstZ;
        size_t dstLOD;
        CUmemorytype dstMemoryType;
        size_t dstHost;
        CUdeviceptr dstDevice;
        CUarray dstArray;
        void *reserved1;
        size_t dstPitch;
        size_t dstHeight;

        size_t WidthInBytes;
        size_t Height;
        size_t Depth;
    } CUDA_MEMCPY3D;

    CUresult cuInit(unsigned int Flags);

    CUresult cuDeviceGetCount(int *count);
    CUresult cuDeviceGet(CUdevice *device,
                         int ordinal);
    CUresult cuDeviceGetAttribute(int *pi,
                                  CUdevice_attribute attrib,
                                  CUdevice dev);
    CUresult cuDeviceGetName(char *name,
                             int len,
                             CUdevice dev);
    CUresult cuDeviceTotalMem_v2(size_t *bytes,
                                 CUdevice dev);
    CUresult cuDeviceGetPCIBusId(char *pciBusId,
                                 int len,
                                 CUdevice dev);
    CUresult cuDeviceGetByPCIBusId(CUdevice *dev,
                                   const char *pciBusId);

    CUresult cuCtxCreate_v2(CUcontext *pctx,
                            unsigned int flags,
                            CUdevice dev);
    CUresult cuCtxDestroy_v2(CUcontext ctx);
    CUresult cuCtxPushCurrent_v2(CUcontext ctx);
    CUresult cuCtxPopCurrent_v2(CUcontext *pctx);
    CUresult cuCtxSetCurrent(CUcontext ctx);
    CUresult cuCtxGetCurrent(CUcontext *pctx);
    CUresult cuCtxSynchronize();

    CUresult cuModuleLoadData(CUmodule *module,
                              const void *image);
    CUresult cuModuleUnload(CUmodule hmod);
    CUresult cuModuleGetFunction(CUfunction *hfunc,
                                 CUmodule hmod,
                                 const char *name);
    CUresult cuModuleGetGlobal_v2(CUdeviceptr *dptr,
                                  size_t *bytes,
                                  CUmodule hmod,
                                  const char *name);

    CUresult cuLaunchKernel(CUfunction f,
                            unsigned int gridDimX,
                            unsigned int gridDimY,
                            unsigned int gridDimZ,
                            unsigned int blockDimX,
                            unsigned int blockDimY,
                            unsigned int blockDimZ,
                            unsigned int sharedMemBytes,
                            CUstream hStream,
                            void **kernelParams,
                            void **extra);

    CUresult cuMemAlloc_v2(CUdeviceptr *dptr,
                           size_t bytesize);
    CUresult cuMemFree_v2(CUdeviceptr dptr);
    CUresult cuMemAllocManaged(CUdeviceptr* dptr,
                               size_t bytesize,
                               unsigned int flags);
    CUresult cuMemHostAlloc(size_t *pp,
                            size_t bytesize,
                            unsigned int Flags);
    CUresult cuMemFreeHost(size_t p);
    CUresult cuMemHostGetDevicePointer_v2(CUdeviceptr *pdptr,
                                          size_t p,
                                          unsigned int Flags);

    CUresult cuMemcpyDtoH_v2(size_t dstHost,
                             CUdeviceptr srcDevice,
                             size_t ByteCount);
    CUresult cuMemcpyHtoD_v2(CUdeviceptr dstDevice,
                             size_t srcHost,
                             size_t ByteCount);
    CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr dstDevice,
                                  size_t srcHost,
                                  size_t ByteCount,
                                  CUstream hStream);
    CUresult cuMemcpyDtoDAsync_v2(CUdeviceptr dstDevice,
                                  CUdeviceptr srcDevice,
                                  size_t ByteCount,
                                  CUstream hStream);
    CUresult cuMemsetD32Async(CUdeviceptr dstDevice,
                              unsigned int ui,
                              size_t N,
                              CUstream hStream);
    CUresult cuMemcpy3DAsync_v2(const CUDA_MEMCPY3D *pCopy,
                                CUstream hStream);

    CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(
                                int *numBlocks,
                                CUfunction func,
                                int blockSize,
                                size_t dynamicSMemSize);
    CUresult cuOccupancyMaxPotentialBlockSize(
                                int *minGridSize,
                                int *blockSize,
                                CUfunction func,
                                CUoccupancyB2DSize blockSizeToDynamicSMemSize,
                                size_t dynamicSMemSize,
                                int blockSizeLimit);
    """

    # Parse
    global ffi
    ffi = cffi.FFI()
    ffi.cdef(src)

    # Load library
    for libnme in backends:
        try:
            lib = ffi.dlopen(libnme)
            break
        except OSError:
            pass
    else:
        ffi = None
        raise OSError("Could not load cuda library")


def initialize(backends=("libcuda.so", "nvcuda.dll")):
    global lib
    if lib is not None:
        return
    global lock
    with lock:
        _initialize(backends)
