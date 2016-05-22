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
cuDNN cffi bindings.
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


#: cudnnNanPropagation_t
CUDNN_NOT_PROPAGATE_NAN = 0
CUDNN_PROPAGATE_NAN = 1


#: cudnnRNNMode_t
CUDNN_RNN_RELU = 0
CUDNN_RNN_TANH = 1
CUDNN_LSTM = 2
CUDNN_GRU = 3


#: cudnnDirectionMode_t
CUDNN_UNIDIRECTIONAL = 0
CUDNN_BIDIRECTIONAL = 1


#: cudnnRNNInputMode_t
CUDNN_LINEAR_INPUT = 0
CUDNN_SKIP_INPUT = 1


#: cudnnSoftmaxAlgorithm_t
CUDNN_SOFTMAX_FAST = 0  # does NOT max to avoid overflow
CUDNN_SOFTMAX_ACCURATE = 1  # subtracts max to avoid overflow
CUDNN_SOFTMAX_LOG = 2


#: cudnnSoftmaxMode_t
CUDNN_SOFTMAX_MODE_INSTANCE = 0  # compute over all C, H, W for each N
CUDNN_SOFTMAX_MODE_CHANNEL = 1  # compute over all C for each H, W, N


#: Cached cudnn version
cudnn_version = 0


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
    typedef int cudnnSoftmaxAlgorithm_t;
    typedef int cudnnSoftmaxMode_t;

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
        int n, int c, int h, int w);
    cudnnStatus_t cudnnGetTensor4dDescriptor(
        const cudnnTensorDescriptor_t tensorDesc,
        cudnnDataType_t *dataType,
        int *n, int *c, int *h, int *w,
        int *nStride, int *cStride, int *hStride, int *wStride);
    cudnnStatus_t cudnnSetTensorNdDescriptor(
        cudnnTensorDescriptor_t tensorDesc,
        cudnnDataType_t dataType,
        int nbDims,
        const int *dimA,
        const int *strideA);
    cudnnStatus_t cudnnGetTensorNdDescriptor(
        const cudnnTensorDescriptor_t tensorDesc,
        int nbDimsRequested,
        cudnnDataType_t *dataType,
        int *nbDims,
        int *dimA,
        int *strideA);

    cudnnStatus_t cudnnCreateFilterDescriptor(
        cudnnFilterDescriptor_t *filterDesc);
    cudnnStatus_t cudnnDestroyFilterDescriptor(
        cudnnFilterDescriptor_t filterDesc);

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
        int *n, int *c, int *h, int *w);

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

    cudnnStatus_t cudnnSoftmaxForward(
        cudnnHandle_t handle,
        cudnnSoftmaxAlgorithm_t algo,
        cudnnSoftmaxMode_t mode,
        const intptr_t alpha,
        const cudnnTensorDescriptor_t xDesc,
        const intptr_t x,
        const intptr_t beta,
        const cudnnTensorDescriptor_t yDesc,
        intptr_t y);
    cudnnStatus_t cudnnSoftmaxBackward(
        cudnnHandle_t handle,
        cudnnSoftmaxAlgorithm_t algo,
        cudnnSoftmaxMode_t mode,
        const intptr_t alpha,
        const cudnnTensorDescriptor_t yDesc,
        const intptr_t y,
        const cudnnTensorDescriptor_t dyDesc,
        const intptr_t dy,
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

    src4p = """
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
    """

    src24 = """
    cudnnStatus_t cudnnSetFilter4dDescriptor(
        cudnnFilterDescriptor_t filterDesc,
        cudnnDataType_t dataType,
        int k, int c, int h, int w);

    cudnnStatus_t cudnnSetPooling2dDescriptor(
        cudnnPoolingDescriptor_t poolingDesc,
        cudnnPoolingMode_t mode,
        int windowHeight,
        int windowWidth,
        int verticalPadding,
        int horizontalPadding,
        int verticalStride,
        int horizontalStride);
    """

    src5 = """
    typedef size_t cudnnRNNDescriptor_t;
    typedef size_t cudnnDropoutDescriptor_t;
    typedef int cudnnNanPropagation_t;
    typedef int cudnnRNNInputMode_t;
    typedef int cudnnDirectionMode_t;
    typedef int cudnnRNNMode_t;

    cudnnStatus_t cudnnSetFilter4dDescriptor(
        cudnnFilterDescriptor_t filterDesc,
        cudnnDataType_t dataType,
        cudnnTensorFormat_t format,
        int k, int c, int h, int w);
    cudnnStatus_t cudnnGetFilter4dDescriptor(
        const cudnnFilterDescriptor_t filterDesc,
        cudnnDataType_t *dataType,
        cudnnTensorFormat_t *format,
        int *k, int *c, int *h, int *w);
    cudnnStatus_t cudnnSetFilterNdDescriptor(
        cudnnFilterDescriptor_t filterDesc,
        cudnnDataType_t dataType,
        cudnnTensorFormat_t format,
        int nbDims,
        const int *filterDimA);
    cudnnStatus_t cudnnGetFilterNdDescriptor(
        const cudnnFilterDescriptor_t filterDesc,
        int nbDimsRequested,
        cudnnDataType_t *dataType,
        cudnnTensorFormat_t *format,
        int *nbDims,
        int *filterDimA);

    cudnnStatus_t cudnnSetPooling2dDescriptor(
        cudnnPoolingDescriptor_t poolingDesc,
        cudnnPoolingMode_t mode,
        cudnnNanPropagation_t maxpoolingNanOpt,
        int windowHeight,
        int windowWidth,
        int verticalPadding,
        int horizontalPadding,
        int verticalStride,
        int horizontalStride);

    cudnnStatus_t cudnnCreateDropoutDescriptor(
        cudnnDropoutDescriptor_t *dropoutDesc);
    cudnnStatus_t cudnnDestroyDropoutDescriptor(
        cudnnDropoutDescriptor_t dropoutDesc);
    cudnnStatus_t cudnnDropoutGetStatesSize(
        cudnnHandle_t handle, size_t *sizeInBytes);
    cudnnStatus_t cudnnDropoutGetReserveSpaceSize(
        cudnnTensorDescriptor_t xdesc, size_t *sizeInBytes);
    cudnnStatus_t cudnnSetDropoutDescriptor(
        cudnnDropoutDescriptor_t dropoutDesc,
        cudnnHandle_t handle,
        float dropout,
        intptr_t states,
        size_t stateSizeInBytes,
        unsigned long long seed);
    cudnnStatus_t cudnnDropoutForward(
        cudnnHandle_t handle,
        const cudnnDropoutDescriptor_t dropoutDesc,
        const cudnnTensorDescriptor_t xdesc,
        const intptr_t x,
        const cudnnTensorDescriptor_t ydesc,
        intptr_t y,
        intptr_t reserveSpace,
        size_t reserveSpaceSizeInBytes);
    cudnnStatus_t cudnnDropoutBackward(
        cudnnHandle_t handle,
        const cudnnDropoutDescriptor_t dropoutDesc,
        const cudnnTensorDescriptor_t dydesc,
        const intptr_t dy,
        const cudnnTensorDescriptor_t dxdesc,
        intptr_t dx,
        intptr_t reserveSpace,
        size_t reserveSpaceSizeInBytes);

    cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t *rnnDesc);
    cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc);
    cudnnStatus_t cudnnSetRNNDescriptor(
        cudnnRNNDescriptor_t rnnDesc,
        int hiddenSize,
        int seqLength,
        int numLayers,
        cudnnDropoutDescriptor_t dropoutDesc,
        cudnnRNNInputMode_t inputMode,
        cudnnDirectionMode_t direction,
        cudnnRNNMode_t mode,
        cudnnDataType_t dataType);
    cudnnStatus_t cudnnGetRNNWorkspaceSize(
        cudnnHandle_t handle,
        const cudnnRNNDescriptor_t rnnDesc,
        const cudnnTensorDescriptor_t *xDesc,
        size_t *sizeInBytes);
    cudnnStatus_t cudnnGetRNNTrainingReserveSize(
        cudnnHandle_t handle,
        const cudnnRNNDescriptor_t rnnDesc,
        const cudnnTensorDescriptor_t *xDesc,
        size_t *sizeInBytes);
    cudnnStatus_t cudnnGetRNNParamsSize(
        cudnnHandle_t handle,
        const cudnnRNNDescriptor_t rnnDesc,
        const cudnnTensorDescriptor_t *xDesc,
        size_t *sizeInBytes);
    cudnnStatus_t cudnnGetRNNLinLayerMatrixParams(
        cudnnHandle_t handle,
        const cudnnRNNDescriptor_t rnnDesc,
        const int layer,
        const cudnnTensorDescriptor_t *xDesc,
        const cudnnFilterDescriptor_t wDesc,
        const intptr_t w,
        const int linLayerID,
        cudnnFilterDescriptor_t linLayerMatDesc,
        intptr_t *linLayerMat);
    cudnnStatus_t cudnnGetRNNLinLayerBiasParams(
        cudnnHandle_t handle,
        const cudnnRNNDescriptor_t rnnDesc,
        const int layer,
        const cudnnTensorDescriptor_t *xDesc,
        const cudnnFilterDescriptor_t wDesc,
        const intptr_t w,
        const int linLayerID,
        cudnnFilterDescriptor_t linLayerBiasDesc,
        intptr_t *linLayerBias);
    cudnnStatus_t cudnnRNNForwardInference(
        cudnnHandle_t handle,
        const cudnnRNNDescriptor_t rnnDesc,
        const cudnnTensorDescriptor_t *xDesc,
        const intptr_t x,
        const cudnnTensorDescriptor_t hxDesc,
        const intptr_t hx,
        const cudnnTensorDescriptor_t cxDesc,
        const intptr_t cx,
        const cudnnFilterDescriptor_t wDesc,
        const intptr_t w,
        const cudnnTensorDescriptor_t *yDesc,
        intptr_t y,
        const cudnnTensorDescriptor_t hyDesc,
        intptr_t hy,
        const cudnnTensorDescriptor_t cyDesc,
        intptr_t cy,
        intptr_t workspace,
        size_t workSpaceSizeInBytes);
    cudnnStatus_t cudnnRNNForwardTraining(
        cudnnHandle_t handle,
        const cudnnRNNDescriptor_t rnnDesc,
        const cudnnTensorDescriptor_t *xDesc,
        const intptr_t x,
        const cudnnTensorDescriptor_t hxDesc,
        const intptr_t hx,
        const cudnnTensorDescriptor_t cxDesc,
        const intptr_t cx,
        const cudnnFilterDescriptor_t wDesc,
        const intptr_t w,
        const cudnnTensorDescriptor_t *yDesc,
        intptr_t y,
        const cudnnTensorDescriptor_t hyDesc,
        intptr_t hy,
        const cudnnTensorDescriptor_t cyDesc,
        intptr_t cy,
        intptr_t workspace,
        size_t workSpaceSizeInBytes,
        intptr_t reserveSpace,
        size_t reserveSpaceSizeInBytes);
    cudnnStatus_t cudnnRNNBackwardData(
        cudnnHandle_t handle,
        const cudnnRNNDescriptor_t rnnDesc,
        const cudnnTensorDescriptor_t *yDesc,
        const intptr_t y,
        const cudnnTensorDescriptor_t *dyDesc,
        const intptr_t dy,
        const cudnnTensorDescriptor_t dhyDesc,
        const intptr_t dhy,
        const cudnnTensorDescriptor_t dcyDesc,
        const intptr_t dcy,
        const cudnnFilterDescriptor_t wDesc,
        const intptr_t w,
        const cudnnTensorDescriptor_t hxDesc,
        const intptr_t hx,
        const cudnnTensorDescriptor_t cxDesc,
        const intptr_t cx,
        const cudnnTensorDescriptor_t *dxDesc,
        intptr_t dx,
        const cudnnTensorDescriptor_t dhxDesc,
        intptr_t dhx,
        const cudnnTensorDescriptor_t dcxDesc,
        intptr_t dcx,
        intptr_t workspace,
        size_t workSpaceSizeInBytes,
        const intptr_t reserveSpace,
        size_t reserveSpaceSizeInBytes);
    cudnnStatus_t cudnnRNNBackwardWeights(
        cudnnHandle_t handle,
        const cudnnRNNDescriptor_t rnnDesc,
        const cudnnTensorDescriptor_t *xDesc,
        const intptr_t x,
        const cudnnTensorDescriptor_t hxDesc,
        const intptr_t hx,
        const cudnnTensorDescriptor_t *yDesc,
        const intptr_t y,
        const intptr_t workspace,
        size_t workSpaceSizeInBytes,
        const cudnnFilterDescriptor_t dwDesc,
        intptr_t dw,
        const intptr_t reserveSpace,
        size_t reserveSpaceSizeInBytes);
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

    global cudnn_version
    cudnn_version = lib.cudnnGetVersion()
    if cudnn_version < 4000:
        ffi.cdef(src2)  # specific functions for V2
        ffi.cdef(src24)  # specific functions for V2 and V4
    else:
        ffi.cdef(src4p)  # specific functions for V4+
        if cudnn_version < 5000:
            ffi.cdef(src24)  # specific functions for V2 and V4
        else:
            ffi.cdef(src5)  # specific functions for V5

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
