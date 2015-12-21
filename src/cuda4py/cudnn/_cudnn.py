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
cuDNN cffi bindings and helper classes.
"""
import cffi
import cuda4py._cffi as cuffi
from cuda4py._py import CU


#: ffi parser
ffi = None


#: Loaded shared library
lib = None


#: Error codes
CUDNN_STATUS_SUCCESS = 0
CUDNN_STATUS_NOT_INITIALIZED = 1
CUDNN_STATUS_ALLOC_FAILED = 2
CUDNN_STATUS_BAD_PARAM = 3
CUDNN_STATUS_INTERNAL_ERROR = 4
CUDNN_STATUS_INVALID_VALUE = 5
CUDNN_STATUS_ARCH_MISMATCH = 6
CUDNN_STATUS_MAPPING_ERROR = 7
CUDNN_STATUS_EXECUTION_FAILED = 8
CUDNN_STATUS_NOT_SUPPORTED = 9
CUDNN_STATUS_LICENSE_ERROR = 10


#: Error descriptions
ERRORS = {
    CUDNN_STATUS_NOT_INITIALIZED: "CUDNN_STATUS_NOT_INITIALIZED",
    CUDNN_STATUS_ALLOC_FAILED: "CUDNN_STATUS_ALLOC_FAILED",
    CUDNN_STATUS_BAD_PARAM: "CUDNN_STATUS_BAD_PARAM",
    CUDNN_STATUS_INTERNAL_ERROR: "CUDNN_STATUS_INTERNAL_ERROR",
    CUDNN_STATUS_INVALID_VALUE: "CUDNN_STATUS_INVALID_VALUE",
    CUDNN_STATUS_ARCH_MISMATCH: "CUDNN_STATUS_ARCH_MISMATCH",
    CUDNN_STATUS_MAPPING_ERROR: "CUDNN_STATUS_MAPPING_ERROR",
    CUDNN_STATUS_EXECUTION_FAILED: "CUDNN_STATUS_EXECUTION_FAILED",
    CUDNN_STATUS_NOT_SUPPORTED: "CUDNN_STATUS_NOT_SUPPORTED",
    CUDNN_STATUS_LICENSE_ERROR: "CUDNN_STATUS_LICENSE_ERROR"
}


#: cudnnDataType_t
CUDNN_DATA_FLOAT = 0
CUDNN_DATA_DOUBLE = 1
CUDNN_DATA_HALF = 2


#: cudnnTensorFormat_t
CUDNN_TENSOR_NCHW = 0
CUDNN_TENSOR_NHWC = 1


#: cudnnConvolutionMode_t
CUDNN_CONVOLUTION = 0
CUDNN_CROSS_CORRELATION = 1


#: cudnnConvolutionFwdPreference_t
CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0
CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1
CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2


#: cudnnConvolutionFwdAlgo_t
CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0
CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1
CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2
CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3
CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4
CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5


#: cudnnConvolutionBwdFilterPreference_t
CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE = 0
CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST = 1
CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT = 2


#: cudnnConvolutionBwdFilterAlgo_t
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0  # non-deterministic
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2
CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3  # non-deterministic with workspace


#: cudnnConvolutionBwdDataPreference_t
CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE = 0
CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST = 1
CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT = 2


#: cudnnConvolutionBwdDataAlgo_t
CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0  # non-deterministic
CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1
CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2
CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3


#: cudnnPoolingMode_t
CUDNN_POOLING_MAX = 0
CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1
CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2


def _initialize(backends):
    global lib
    if lib is not None:
        return
    # C function definitions
    # size_t instead of void* is used
    # for convinience with python calls and numpy arrays,
    # cffi automatically calls int() on objects also.
    src = """
    typedef int cudnnStatus_t;
    typedef size_t cudnnHandle_t;
    typedef size_t cudnnTensorDescriptor_t;
    typedef size_t cudnnConvolutionDescriptor_t;
    typedef size_t cudnnFilterDescriptor_t;
    typedef size_t cudnnPoolingDescriptor_t;
    typedef int cudnnTensorFormat_t;
    typedef int cudnnDataType_t;
    typedef int cudnnConvolutionMode_t;
    typedef int cudnnConvolutionFwdPreference_t;
    typedef int cudnnConvolutionFwdAlgo_t;
    typedef int cudnnPoolingMode_t;

    size_t cudnnGetVersion();

    cudnnStatus_t cudnnCreate(cudnnHandle_t *handle);
    cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);

    cudnnStatus_t cudnnCreateTensorDescriptor(
        cudnnTensorDescriptor_t *tensorDesc);
    cudnnStatus_t cudnnDestroyTensorDescriptor(
        cudnnTensorDescriptor_t tensorDesc);
    cudnnStatus_t cudnnSetTensor4dDescriptor(
        cudnnTensorDescriptor_t tensorDesc,
        cudnnTensorFormat_t format,
        cudnnDataType_t dataType,
        int n,
        int c,
        int h,
        int w);

    cudnnStatus_t cudnnCreateFilterDescriptor(
        cudnnFilterDescriptor_t *filterDesc);
    cudnnStatus_t cudnnDestroyFilterDescriptor(
        cudnnFilterDescriptor_t filterDesc);
    cudnnStatus_t cudnnSetFilter4dDescriptor(
        cudnnFilterDescriptor_t filterDesc,
        cudnnDataType_t dataType,
        int k,
        int c,
        int h,
        int w);

    cudnnStatus_t cudnnCreateConvolutionDescriptor(
        cudnnConvolutionDescriptor_t *convDesc);
    cudnnStatus_t cudnnDestroyConvolutionDescriptor(
        cudnnConvolutionDescriptor_t convDesc);
    cudnnStatus_t cudnnSetConvolution2dDescriptor(
        cudnnConvolutionDescriptor_t convDesc,
        int pad_h,
        int pad_w,
        int u,
        int v,
        int upscalex,
        int upscaley,
        cudnnConvolutionMode_t mode);

    cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t inputTensorDesc,
        const cudnnFilterDescriptor_t filterDesc,
        int *n,
        int *c,
        int *h,
        int *w);

    cudnnStatus_t cudnnGetConvolutionForwardAlgorithm(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t srcDesc,
        const cudnnFilterDescriptor_t filterDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t destDesc,
        cudnnConvolutionFwdPreference_t preference,
        size_t memoryLimitInbytes,
        cudnnConvolutionFwdAlgo_t *algo);

    cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t srcDesc,
        const cudnnFilterDescriptor_t filterDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t destDesc,
        cudnnConvolutionFwdAlgo_t algo,
        size_t *sizeInBytes);

    cudnnStatus_t cudnnConvolutionForward(
        cudnnHandle_t handle,
        const intptr_t alpha,
        const cudnnTensorDescriptor_t srcDesc,
        const intptr_t srcData,
        const cudnnFilterDescriptor_t filterDesc,
        const intptr_t filterData,
        const cudnnConvolutionDescriptor_t convDesc,
        cudnnConvolutionFwdAlgo_t algo,
        intptr_t workSpace,
        size_t workSpaceSizeInBytes,
        const intptr_t beta,
        const cudnnTensorDescriptor_t destDesc,
        intptr_t destData);

    cudnnStatus_t cudnnConvolutionBackwardBias(
        cudnnHandle_t handle,
        const intptr_t alpha,
        const cudnnTensorDescriptor_t srcDesc,
        const intptr_t srcData,
        const intptr_t beta,
        const cudnnTensorDescriptor_t destDesc,
        intptr_t destData);

    cudnnStatus_t cudnnTransformTensor(
        cudnnHandle_t handle,
        const intptr_t alpha,
        const cudnnTensorDescriptor_t srcDesc,
        const intptr_t srcData,
        const intptr_t beta,
        const cudnnTensorDescriptor_t destDesc,
        intptr_t destData);

    cudnnStatus_t cudnnCreatePoolingDescriptor(
        cudnnPoolingDescriptor_t *poolingDesc);
    cudnnStatus_t cudnnDestroyPoolingDescriptor(
        cudnnPoolingDescriptor_t poolingDesc);
    cudnnStatus_t cudnnSetPooling2dDescriptor(
        cudnnPoolingDescriptor_t poolingDesc,
        cudnnPoolingMode_t mode,
        int windowHeight,
        int windowWidth,
        int verticalPadding,
        int horizontalPadding,
        int verticalStride,
        int horizontalStride);
    cudnnStatus_t cudnnGetPooling2dForwardOutputDim(
        const cudnnPoolingDescriptor_t poolingDesc,
        const cudnnTensorDescriptor_t inputTensorDesc,
        int *n, int *c, int *h, int *w);
    cudnnStatus_t cudnnPoolingForward(
        cudnnHandle_t handle,
        const cudnnPoolingDescriptor_t poolingDesc,
        const intptr_t alpha,
        const cudnnTensorDescriptor_t xDesc,
        const intptr_t x,
        const intptr_t beta,
        const cudnnTensorDescriptor_t yDesc,
        intptr_t y);
    cudnnStatus_t cudnnPoolingBackward(
        cudnnHandle_t handle,
        const cudnnPoolingDescriptor_t poolingDesc,
        const intptr_t alpha,
        const cudnnTensorDescriptor_t yDesc,
        const intptr_t y,
        const cudnnTensorDescriptor_t dyDesc,
        const intptr_t dy,
        const cudnnTensorDescriptor_t xDesc,
        const intptr_t x,
        const intptr_t beta,
        const cudnnTensorDescriptor_t dxDesc,
        intptr_t dx);
    """

    src2 = """
    cudnnStatus_t cudnnConvolutionBackwardFilter(
        cudnnHandle_t handle,
        const intptr_t alpha,
        const cudnnTensorDescriptor_t srcDesc,
        const intptr_t srcData,
        const cudnnTensorDescriptor_t diffDesc,
        const intptr_t diffData,
        const cudnnConvolutionDescriptor_t convDesc,
        const intptr_t beta,
        const cudnnFilterDescriptor_t gradDesc,
        intptr_t gradData);

    cudnnStatus_t cudnnConvolutionBackwardData(
        cudnnHandle_t handle,
        const intptr_t alpha,
        const cudnnFilterDescriptor_t filterDesc,
        const intptr_t filterData,
        const cudnnTensorDescriptor_t diffDesc,
        const intptr_t diffData,
        const cudnnConvolutionDescriptor_t convDesc,
        const intptr_t beta,
        const cudnnTensorDescriptor_t gradDesc,
        intptr_t gradData);
    """

    src4 = """
    typedef int cudnnConvolutionBwdFilterAlgo_t;
    typedef int cudnnConvolutionBwdDataAlgo_t;
    typedef int cudnnConvolutionBwdFilterPreference_t;
    typedef int cudnnConvolutionBwdDataPreference_t;

    cudnnStatus_t cudnnGetConvolutionBackwardFilterAlgorithm(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t xDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnFilterDescriptor_t dwDesc,
        cudnnConvolutionBwdFilterPreference_t preference,
        size_t memoryLimitInBytes,
        cudnnConvolutionBwdFilterAlgo_t *algo);
    cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnnHandle_t handle,
        const cudnnTensorDescriptor_t xDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnFilterDescriptor_t gradDesc,
        cudnnConvolutionBwdFilterAlgo_t algo,
        size_t *sizeInBytes);
    cudnnStatus_t cudnnConvolutionBackwardFilter(
        cudnnHandle_t handle,
        const intptr_t alpha,
        const cudnnTensorDescriptor_t srcDesc,
        const intptr_t srcData,
        const cudnnTensorDescriptor_t diffDesc,
        const intptr_t diffData,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnConvolutionBwdFilterAlgo_t algo,
        intptr_t workSpace,
        size_t workSpaceSizeInBytes,
        const intptr_t beta,
        const cudnnFilterDescriptor_t gradDesc,
        intptr_t gradData);

    cudnnStatus_t cudnnGetConvolutionBackwardDataAlgorithm(
        cudnnHandle_t handle,
        const cudnnFilterDescriptor_t wDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t dxDesc,
        cudnnConvolutionBwdDataPreference_t preference,
        size_t memoryLimitInBytes,
        cudnnConvolutionBwdDataAlgo_t *algo);
    cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnnHandle_t handle,
        const cudnnFilterDescriptor_t wDesc,
        const cudnnTensorDescriptor_t dyDesc,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnTensorDescriptor_t dxDesc,
        cudnnConvolutionBwdDataAlgo_t algo,
        size_t *sizeInBytes);
    cudnnStatus_t cudnnConvolutionBackwardData(
        cudnnHandle_t handle,
        const intptr_t alpha,
        const cudnnFilterDescriptor_t filterDesc,
        const intptr_t filterData,
        const cudnnTensorDescriptor_t diffDesc,
        const intptr_t diffData,
        const cudnnConvolutionDescriptor_t convDesc,
        const cudnnConvolutionBwdDataAlgo_t algo,
        intptr_t workSpace,
        size_t workSpaceSizeInBytes,
        const intptr_t beta,
        const cudnnTensorDescriptor_t gradDesc,
        intptr_t gradData);

    cudnnStatus_t cudnnConvolutionBackwardFilter_v2(
        cudnnHandle_t handle,
        const intptr_t alpha,
        const cudnnTensorDescriptor_t srcDesc,
        const intptr_t srcData,
        const cudnnTensorDescriptor_t diffDesc,
        const intptr_t diffData,
        const cudnnConvolutionDescriptor_t convDesc,
        const intptr_t beta,
        const cudnnFilterDescriptor_t gradDesc,
        intptr_t gradData);

    cudnnStatus_t cudnnConvolutionBackwardData_v2(
        cudnnHandle_t handle,
        const intptr_t alpha,
        const cudnnFilterDescriptor_t filterDesc,
        const intptr_t filterData,
        const cudnnTensorDescriptor_t diffDesc,
        const intptr_t diffData,
        const cudnnConvolutionDescriptor_t convDesc,
        const intptr_t beta,
        const cudnnTensorDescriptor_t gradDesc,
        intptr_t gradData);
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
        raise OSError("Could not load cudnn library")

    if lib.cudnnGetVersion() < 4000:
        ffi.cdef(src2)  # specific functions for V2
    else:
        ffi.cdef(src4)  # specific functions for V4

    global ERRORS
    for code, msg in ERRORS.items():
        if code in CU.ERRORS:
            s = " | " + msg
            if s not in CU.ERRORS[code]:
                CU.ERRORS[code] += s
        else:
            CU.ERRORS[code] = msg


def initialize(backends=("libcudnn.so", "cudnn64_65.dll")):
    """Loads shared library.
    """
    cuffi.initialize()
    global lib
    if lib is not None:
        return
    with cuffi.lock:
        _initialize(backends)


class Descriptor(object):
    """CUDNN descriptor base class.
    """
    def __init__(self):
        self._lib = lib
        self._handle = None
        self._create()

    def _create(self):
        """Calls cudnnCreate*Descriptor and assigns self._handle
        """
        raise NotImplementedError()

    def _destroy(self):
        """Calls cudnnDestroy*Descriptor on self._handle
        """
        raise NotImplementedError()

    def __int__(self):
        return self.handle

    @property
    def handle(self):
        return self._handle

    def _release(self):
        if self._lib is not None and self.handle is not None:
            self._destroy()
            self._handle = None

    def __del__(self):
        self._release()


class TensorDescriptor(Descriptor):
    """CUDNN tensor descriptor.
    """
    def _create(self):
        handle = ffi.new("cudnnTensorDescriptor_t *")
        err = lib.cudnnCreateTensorDescriptor(handle)
        if err:
            raise CU.error("cudnnCreateTensorDescriptor", err)
        self._handle = int(handle[0])

    def _destroy(self):
        self._lib.cudnnDestroyTensorDescriptor(self.handle)

    def set_4d(self, fmt, data_type, n, c, h, w):
        """Initializes tensor descriptor into a 4D tensor.

        Parameters:
            fmt: CUDNN_TENSOR_NCHW or CUDNN_TENSOR_NHWC.
            data_type: CUDNN_DATA_FLOAT or CUDNN_DATA_DOUBLE.
            n: number of images.
            c: number of image channels.
            h: image height.
            w: image width.
        """
        err = self._lib.cudnnSetTensor4dDescriptor(
            self.handle, fmt, data_type, n, c, h, w)
        if err:
            raise CU.error("cudnnSetTensor4dDescriptor", err)


class FilterDescriptor(Descriptor):
    """CUDNN filter descriptor.
    """
    def _create(self):
        handle = ffi.new("cudnnFilterDescriptor_t *")
        err = lib.cudnnCreateFilterDescriptor(handle)
        if err:
            raise CU.error("cudnnCreateFilterDescriptor", err)
        self._handle = int(handle[0])

    def _destroy(self):
        self._lib.cudnnDestroyFilterDescriptor(self.handle)

    def set_4d(self, data_type, k, c, h, w):
        """Initializes tensor descriptor into a 4D tensor.

        Parameters:
            data_type: CUDNN_DATA_FLOAT or CUDNN_DATA_DOUBLE.
            k: number of kernels.
            c: number of image channels.
            h: image height.
            w: image width.
        """
        err = self._lib.cudnnSetFilter4dDescriptor(
            self.handle, data_type, k, c, h, w)
        if err:
            raise CU.error("cudnnSetFilter4dDescriptor", err)


class ConvolutionDescriptor(Descriptor):
    """CUDNN convolution descriptor.
    """
    def _create(self):
        handle = ffi.new("cudnnConvolutionDescriptor_t *")
        err = lib.cudnnCreateConvolutionDescriptor(handle)
        if err:
            raise CU.error("cudnnCreateConvolutionDescriptor", err)
        self._handle = int(handle[0])

    def _destroy(self):
        self._lib.cudnnDestroyConvolutionDescriptor(self.handle)

    def set_2d(self, pad_h, pad_w, u, v, upscalex=1, upscaley=1,
               mode=CUDNN_CROSS_CORRELATION):
        """Initializes tensor descriptor into a 4D tensor.

        Parameters:
            pad_h: zero-padding height (top & bottom).
            pad_w: zero-padding width (left & right).
            u: vertical filter stride.
            v: horizontal filter stride.
            upscalex: upscale the input in x-direction.
            upscaley: upscale the input in y-direction.
            mode: CUDNN_CROSS_CORRELATION or CUDNN_CROSS_CONVOLUTION.
        """
        err = self._lib.cudnnSetConvolution2dDescriptor(
            self.handle, pad_h, pad_w, u, v, upscalex, upscaley, mode)
        if err:
            raise CU.error("cudnnSetConvolution2dDescriptor", err)


class PoolingDescriptor(Descriptor):
    """CUDNN pooling descriptor.
    """
    def _create(self):
        handle = ffi.new("cudnnPoolingDescriptor_t *")
        err = lib.cudnnCreatePoolingDescriptor(handle)
        if err:
            raise CU.error("cudnnCreatePoolingDescriptor", err)
        self._handle = int(handle[0])

    def _destroy(self):
        self._lib.cudnnDestroyPoolingDescriptor(self.handle)

    def set_2d(self, window_hw, padding_vh, stride_vh, mode=CUDNN_POOLING_MAX):
        """Initializes tensor descriptor into a 4D tensor.

        Parameters:
            window_hw: tuple of ints for pooling window (height, width).
            padding_vh: tuple for padding (vertical, horizontal).
            stride_vh: tuple for stride (vertical, horizontal).
            mode: pooling mode.
        """
        err = self._lib.cudnnSetPooling2dDescriptor(
            self.handle, mode, window_hw[0], window_hw[1],
            padding_vh[0], padding_vh[1], stride_vh[0], stride_vh[1])
        if err:
            raise CU.error("cudnnSetPooling2dDescriptor", err)


class CUDNN(object):
    """CUDNN functions can be invoked from this class.
    """
    def __init__(self, context):
        self._context = context
        self._lib = None
        context._add_ref(self)
        initialize()
        self.version = int(lib.cudnnGetVersion())
        handle = ffi.new("cudnnHandle_t *")
        with context:
            err = lib.cudnnCreate(handle)
        if err:
            self._handle = None
            raise CU.error("cudnnCreate", err)
        self._lib = lib  # to hold the reference
        self._handle = int(handle[0])

    def __int__(self):
        return self.handle

    @property
    def handle(self):
        return self._handle

    @property
    def context(self):
        return self._context

    @staticmethod
    def get_convolution_2d_forward_output_dim(conv_desc, input_desc,
                                              filter_desc):
        """Returns tuple of n, c, h, w for an output.
        """
        n, c, h, w = (ffi.new("int *") for _ in range(4))
        err = lib.cudnnGetConvolution2dForwardOutputDim(
            conv_desc, input_desc, filter_desc, n, c, h, w)
        if err:
            raise CU.error("cudnnGetConvolution2dForwardOutputDim", err)
        return int(n[0]), int(c[0]), int(h[0]), int(w[0])

    @staticmethod
    def get_pooling_2d_forward_output_dim(pooling_desc, input_desc):
        """Returns tuple of n, c, h, w for an output.
        """
        n, c, h, w = (ffi.new("int *") for _ in range(4))
        err = lib.cudnnGetPooling2dForwardOutputDim(
            pooling_desc, input_desc, n, c, h, w)
        if err:
            raise CU.error("cudnnGetPooling2dForwardOutputDim", err)
        return int(n[0]), int(c[0]), int(h[0]), int(w[0])

    def get_convolution_forward_algorithm(
            self, src_desc, filter_desc, conv_dec, dest_desc,
            preference=CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, memory_limit=0):
        """Returns forward algorithm based on parameters.
        """
        algo = ffi.new("cudnnConvolutionFwdAlgo_t *")
        err = self._lib.cudnnGetConvolutionForwardAlgorithm(
            self.handle, src_desc, filter_desc, conv_dec, dest_desc,
            preference, memory_limit, algo)
        if err:
            raise CU.error("cudnnGetConvolutionForwardAlgorithm", err)
        return int(algo[0])

    def get_convolution_forward_workspace_size(
            self, src_desc, filter_desc, conv_dec, dest_desc, algo):
        """Returns required size of the additional temporary buffer
        for the specified forward convolution algorithm.
        """
        size = ffi.new("size_t *")
        err = self._lib.cudnnGetConvolutionForwardWorkspaceSize(
            self.handle, src_desc, filter_desc, conv_dec, dest_desc,
            algo, size)
        if err:
            raise CU.error("cudnnGetConvolutionForwardWorkspaceSize", err)
        return int(size[0])

    def convolution_forward(
            self, alpha, src_desc, src_data, filter_desc, filter_data,
            conv_desc, algo, workspace, workspace_size,
            beta, dest_desc, dest_data):
        """Does convolution forward propagation.

        Parameters:
            alpha: src_data multiplier (numpy array with one element).
            beta: dest_data multiplier (numpy array with one element).
        """
        size = ffi.new("size_t *")
        err = self._lib.cudnnConvolutionForward(
            self.handle, CU.extract_ptr(alpha), src_desc, src_data,
            filter_desc, filter_data, conv_desc,
            algo, workspace, workspace_size,
            CU.extract_ptr(beta), dest_desc, dest_data)
        if err:
            raise CU.error("cudnnConvolutionForward", err)
        return int(size[0])

    def convolution_backward_bias(self, alpha, src_desc, src_data,
                                  beta, dest_desc, dest_data):
        """Computes gradient for the bias.

        Parameters:
            alpha: src_data multiplier (numpy array with one element).
            beta: dest_data multiplier (numpy array with one element).
            src_data: error for backpropagation.
            dest_data: gradient for the bias.
        """
        err = self._lib.cudnnConvolutionBackwardBias(
            self.handle, CU.extract_ptr(alpha), src_desc, src_data,
            CU.extract_ptr(beta), dest_desc, dest_data)
        if err:
            raise CU.error("cudnnConvolutionBackwardBias", err)

    def get_convolution_backward_filter_algorithm(
            self, src_desc, diff_desc, conv_dec, grad_desc,
            preference=CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            memory_limit=0):
        """Returns backward filter algorithm based on parameters.

        Parameters:
            src_desc: descriptor of input from the forward pass.
            diff_desc: descriptor of the error for backpropagation.
            conv_desc: descriptor of the convolution (padding, stride, etc.).
            grad_desc: descriptor of the gradient for convolutional kernels.
        """
        algo = ffi.new("cudnnConvolutionBwdFilterAlgo_t *")
        err = self._lib.cudnnGetConvolutionBackwardFilterAlgorithm(
            self.handle, src_desc, diff_desc, conv_dec, grad_desc,
            preference, memory_limit, algo)
        if err:
            raise CU.error("cudnnGetConvolutionBackwardFilterAlgorithm", err)
        return int(algo[0])

    def get_convolution_backward_filter_workspace_size(
            self, src_desc, diff_desc, conv_desc, grad_desc, algo):
        """Returns required size of the additional temporary buffer
        for the specified backward filter convolution algorithm.

        Parameters:
            src_desc: descriptor of input from the forward pass.
            diff_desc: descriptor of the error for backpropagation.
            conv_desc: descriptor of the convolution (padding, stride, etc.).
            grad_desc: descriptor of the gradient for convolutional kernels.
            algo: algorithm for the computing of kernel's gradient.
        """
        size = ffi.new("size_t *")
        err = self._lib.cudnnGetConvolutionBackwardFilterWorkspaceSize(
            self.handle, src_desc, diff_desc, conv_desc, grad_desc, algo, size)
        if err:
            raise CU.error("cudnnGetConvolutionBackwardFilterWorkspaceSize",
                           err)
        return int(size[0])

    def convolution_backward_filter(
            self, alpha, src_desc, src_data, diff_desc, diff_data, conv_desc,
            beta, grad_desc, grad_data,
            algo=None, workspace=None, workspace_size=0):
        """Computes gradient for the convolutional kernels.

        Parameters:
            alpha: src_data multiplier (numpy array with one element).
            beta: grad_data multiplier (numpy array with one element).
            src_data: input from the forward pass.
            diff_data: error for backpropagation.
            grad_data: gradient for convolutional kernels.
        """
        if self.version < 4000:
            err = self._lib.cudnnConvolutionBackwardFilter(
                self.handle, CU.extract_ptr(alpha), src_desc, src_data,
                diff_desc, diff_data, conv_desc,
                CU.extract_ptr(beta), grad_desc, grad_data)
        elif algo is None:
            err = self._lib.cudnnConvolutionBackwardFilter_v2(
                self.handle, CU.extract_ptr(alpha), src_desc, src_data,
                diff_desc, diff_data, conv_desc,
                CU.extract_ptr(beta), grad_desc, grad_data)
        else:
            err = self._lib.cudnnConvolutionBackwardFilter(
                self.handle, CU.extract_ptr(alpha), src_desc, src_data,
                diff_desc, diff_data, conv_desc,
                algo, workspace, workspace_size,
                CU.extract_ptr(beta), grad_desc, grad_data)
        if err:
            raise CU.error("cudnnConvolutionBackwardFilter", err)

    def get_convolution_backward_data_algorithm(
            self, filter_desc, diff_desc, conv_desc, grad_desc,
            preference=CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            memory_limit=0):
        """Returns backward data algorithm based on parameters.

        Parameters:
            filter_desc: descriptor of the convolutional kernels.
            diff_desc: descriptor of the error for backpropagation.
            conv_desc: descriptor of the convolution (padding, stride, etc.).
            grad_desc: descriptor of the backpropagated gradient
                       (same as for input vector during forward pass).
        """
        algo = ffi.new("cudnnConvolutionBwdDataAlgo_t *")
        err = self._lib.cudnnGetConvolutionBackwardDataAlgorithm(
            self.handle, filter_desc, diff_desc, conv_desc, grad_desc,
            preference, memory_limit, algo)
        if err:
            raise CU.error("cudnnGetConvolutionBackwardDataAlgorithm", err)
        return int(algo[0])

    def get_convolution_backward_data_workspace_size(
            self, filter_desc, diff_desc, conv_desc, grad_desc, algo):
        """Returns required size of the additional temporary buffer
        for the specified backward data convolution algorithm.

        Parameters:
            filter_desc: descriptor of the convolutional kernels.
            diff_desc: descriptor of the error for backpropagation.
            conv_desc: descriptor of the convolution (padding, stride, etc.).
            grad_desc: descriptor of the backpropagated gradient
                       (same as for input vector during forward pass).
            algo: algorithm for the computing of backpropagated gradient.
        """
        size = ffi.new("size_t *")
        err = self._lib.cudnnGetConvolutionBackwardDataWorkspaceSize(
            self.handle, filter_desc, diff_desc, conv_desc, grad_desc, algo,
            size)
        if err:
            raise CU.error("cudnnGetConvolutionBackwardDataWorkspaceSize",
                           err)
        return int(size[0])

    def convolution_backward_data(
            self, alpha, filter_desc, filter_data, diff_desc, diff_data,
            conv_desc, beta, grad_desc, grad_data,
            algo=None, workspace=None, workspace_size=0):
        """Computes backpropagated error.

        Parameters:
            alpha: diff_data multiplier (numpy array with one element).
            beta: grad_data multiplier (numpy array with one element).
            filter_data: convolutional kernels.
            diff_data: error for backpropagation.
            grad_data: backpropagated error.
        """
        if self.version < 4000:
            err = self._lib.cudnnConvolutionBackwardData(
                self.handle, CU.extract_ptr(alpha), filter_desc, filter_data,
                diff_desc, diff_data, conv_desc,
                CU.extract_ptr(beta), grad_desc, grad_data)
        elif algo is None:
            err = self._lib.cudnnConvolutionBackwardData_v2(
                self.handle, CU.extract_ptr(alpha), filter_desc, filter_data,
                diff_desc, diff_data, conv_desc,
                CU.extract_ptr(beta), grad_desc, grad_data)
        else:
            err = self._lib.cudnnConvolutionBackwardData(
                self.handle, CU.extract_ptr(alpha), filter_desc, filter_data,
                diff_desc, diff_data, conv_desc,
                algo, workspace, workspace_size,
                CU.extract_ptr(beta), grad_desc, grad_data)
        if err:
            raise CU.error("cudnnConvolutionBackwardData", err)

    def pooling_forward(self, pooling_desc, alpha, src_desc, src_data,
                        beta, dest_desc, dest_data):
        """Does pooling forward propagation.

        Parameters:
            alpha: src_data multiplier (numpy array with one element).
            beta: dest_data multiplier (numpy array with one element).
        """
        err = self._lib.cudnnPoolingForward(
            self.handle, pooling_desc, CU.extract_ptr(alpha),
            src_desc, src_data,
            CU.extract_ptr(beta), dest_desc, dest_data)
        if err:
            raise CU.error("cudnnPoolingForward", err)

    def pooling_backward(self, pooling_desc, alpha, output_desc, output_data,
                         diff_desc, diff_data, input_desc, input_data,
                         beta, grad_desc, grad_data):
        """Does pooling backward propagation.

        Parameters:
            alpha: diff_data multiplier (numpy array with one element).
            beta: grad_data multiplier (numpy array with one element).
            output: output of the forward propagation.
            diff: error for backpropagation.
            input: input of the forward propagation.
            grad: backpropagated error.
        """
        err = self._lib.cudnnPoolingBackward(
            self.handle, pooling_desc, CU.extract_ptr(alpha),
            output_desc, output_data,
            diff_desc, diff_data, input_desc, input_data,
            CU.extract_ptr(beta), grad_desc, grad_data)
        if err:
            raise CU.error("cudnnPoolingBackward", err)

    def transform_tensor(self, alpha, src_desc, src_data,
                         beta, dest_desc, dest_data):
        """Transforms data from one layout to another
        (interleaved to splitted for example).

        Parameters:
            alpha: src_data multiplier (numpy array with one element).
            beta: dest_data multiplier (numpy array with one element).
        """
        err = self._lib.cudnnTransformTensor(
            self.handle, CU.extract_ptr(alpha), src_desc, src_data,
            CU.extract_ptr(beta), dest_desc, dest_data)
        if err:
            raise CU.error("cudnnTransformTensor", err)

    def _release(self):
        if self._lib is not None and self.handle is not None:
            self._lib.cudnnDestroy(self.handle)
            self._handle = None

    def __del__(self):
        if self.context.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self._release()
        self.context._del_ref(self)
