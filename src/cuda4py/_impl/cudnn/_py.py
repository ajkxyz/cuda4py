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
cuDNN helper classes.
"""
import cuda4py._impl.cudnn._cffi as cudnnffi
from cuda4py._impl.cudnn._cffi import (
    CUDNN_TENSOR_NCHW, CUDNN_CROSS_CORRELATION, CUDNN_POOLING_MAX,
    CUDNN_NOT_PROPAGATE_NAN, CUDNN_LINEAR_INPUT, CUDNN_UNIDIRECTIONAL,
    CUDNN_RNN_RELU, CUDNN_RNN_TANH, CUDNN_LSTM, CUDNN_GRU,
    CUDNN_DATA_FLOAT, CUDNN_DATA_DOUBLE, CUDNN_DATA_HALF,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
    CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,
    CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
    CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
    CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE)
from cuda4py._py import CU, MemPtr


class Descriptor(object):
    """CUDNN descriptor base class.
    """
    def __init__(self):
        self._lib = cudnnffi.lib
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


class DataDescriptor(Descriptor):
    """Descriptor for data (tensor, filter).
    """
    def __init__(self):
        super(DataDescriptor, self).__init__()
        self._data_type = -1
        self._fmt = -1
        self._dims = tuple()
        self._strides = tuple()
        self._c = 0
        self._h = 0
        self._w = 0

    @property
    def data_type(self):
        """Math precision.

        CUDNN_DATA_FLOAT, CUDNN_DATA_DOUBLE or CUDNN_DATA_HALF.
        """
        return self._data_type

    @property
    def fmt(self):
        """Descriptor format if applicable.

        CUDNN_TENSOR_NCHW or CUDNN_TENSOR_NHWC.
        """
        return self._fmt

    @property
    def dims(self):
        """Dimensions of a tensor.
        """
        return self._dims

    @property
    def strides(self):
        """Strides of a tensor dimensions.
        """
        return self._strides

    @property
    def c(self):
        """Number of image channels.
        """
        return self._c

    @property
    def h(self):
        """Image height.
        """
        return self._h

    @property
    def w(self):
        """Image width.
        """
        return self._w


class TensorDescriptor(DataDescriptor):
    """CUDNN tensor descriptor.
    """
    def __init__(self):
        super(TensorDescriptor, self).__init__()
        self._n = 0

    @property
    def n(self):
        """Number of images.
        """
        return self._n

    def _create(self):
        handle = cudnnffi.ffi.new("cudnnTensorDescriptor_t *")
        err = self._lib.cudnnCreateTensorDescriptor(handle)
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
        self._fmt, self._data_type = int(fmt), int(data_type)
        self._n, self._c, self._h, self._w = int(n), int(c), int(h), int(w)

    def get_4d(self):
        """Fills data_type, n, c, h, w properties.
        """
        data_type = cudnnffi.ffi.new("cudnnDataType_t *")
        n, c, h, w = (cudnnffi.ffi.new("int *") for _i in range(4))
        ns, cs, hs, ws = (cudnnffi.ffi.new("int *") for _i in range(4))
        err = self._lib.cudnnGetTensor4dDescriptor(
            self.handle, data_type, n, c, h, w, ns, cs, hs, ws)
        if err:
            raise CU.error("cudnnGetTensor4dDescriptor", err)
        self._data_type, self._n, self._c, self._h, self._w = (
            int(x[0]) for x in (data_type, n, c, h, w))

    def set_nd(self, data_type, dims, strides=None):
        """Initializes tensor descriptor.

        Parameters:
            data_type: data type.
            dims: tuple of dimensions.
            strides: tuple of strides of None to create compact layout.
        """
        if strides is not None and len(dims) != len(strides):
            raise ValueError("len(dims) != len(strides)")
        _dims = cudnnffi.ffi.new("int[]", len(dims))
        _dims[0:len(dims)] = dims
        _strides = cudnnffi.ffi.new("int[]", len(dims))
        if strides is None:
            sz = 1
            for i, n in enumerate(dims):
                _strides[i] = sz
                sz *= n
        else:
            _strides[0:len(dims)] = strides
        err = self._lib.cudnnSetTensorNdDescriptor(
            self.handle, data_type, len(dims), _dims, _strides)
        if err:
            raise CU.error("cudnnSetTensorNdDescriptor", err)
        self._data_type = int(data_type)
        self._dims = tuple(int(x) for x in _dims)
        self._strides = tuple(int(x) for x in _strides)

    def get_nd(self, n_dims_requested):
        """Fills data_type, dims and strides properties.

        Parameters:
            n_dims_requested: number of dimensions to extract information from.
        """
        data_type = cudnnffi.ffi.new("cudnnDataType_t *")
        n_dims = cudnnffi.ffi.new("int *")
        dims = cudnnffi.ffi.new("int[]", n_dims_requested)
        strides = cudnnffi.ffi.new("int[]", n_dims_requested)
        err = self._lib.cudnnGetTensorNdDescriptor(
            self.handle, n_dims_requested, data_type, n_dims, dims, strides)
        if err:
            raise CU.error("cudnnGetTensorNdDescriptor", err)
        self._data_type = int(data_type[0])
        self._dims = tuple(int(x) for x in dims)
        self._strides = tuple(int(x) for x in strides)

    @property
    def dropout_reserve_space_size(self):
        """Returns the amount of reserve needed to run dropout with the
        current tensor.
        """
        size = cudnnffi.ffi.new("size_t *")
        err = self._lib.cudnnDropoutGetReserveSpaceSize(self.handle, size)
        if err:
            raise CU.error("cudnnDropoutGetReserveSpaceSize", err)
        return int(size[0])


class FilterDescriptor(DataDescriptor):
    """CUDNN filter descriptor.
    """
    def __init__(self):
        super(FilterDescriptor, self).__init__()
        self._k = 0

    @property
    def k(self):
        """Number of kernels.
        """
        return self._k

    def _create(self):
        handle = cudnnffi.ffi.new("cudnnFilterDescriptor_t *")
        err = self._lib.cudnnCreateFilterDescriptor(handle)
        if err:
            raise CU.error("cudnnCreateFilterDescriptor", err)
        self._handle = int(handle[0])

    def _destroy(self):
        self._lib.cudnnDestroyFilterDescriptor(self.handle)

    def set_4d(self, data_type, k, c, h, w, fmt=CUDNN_TENSOR_NCHW):
        """Initializes tensor descriptor into a 4D tensor.

        Parameters:
            data_type: CUDNN_DATA_FLOAT or CUDNN_DATA_DOUBLE.
            k: number of kernels.
            c: number of image channels.
            h: image height.
            w: image width.
            fmt: tensor format for weights.
        """
        if cudnnffi.cudnn_version < 5000:
            err = self._lib.cudnnSetFilter4dDescriptor(
                self.handle, data_type, k, c, h, w)
            self._fmt = CUDNN_TENSOR_NCHW
        else:
            err = self._lib.cudnnSetFilter4dDescriptor(
                self.handle, data_type, fmt, k, c, h, w)
            self._fmt = fmt
        self._data_type = data_type
        self._k, self._c, self._h, self._w = int(k), int(c), int(h), int(w)
        if err:
            raise CU.error("cudnnSetFilter4dDescriptor", err)

    def get_4d(self):
        """Fills data_type, fmt, k, c, h, w properties.
        """
        data_type = cudnnffi.ffi.new("cudnnDataType_t *")
        fmt = cudnnffi.ffi.new("cudnnTensorFormat_t *")
        k, c, h, w = (cudnnffi.ffi.new("int *") for _i in range(4))
        err = self._lib.cudnnGetFilter4dDescriptor(
            self.handle, data_type, fmt, k, c, h, w)
        if err:
            raise CU.error("cudnnGetFilter4dDescriptor", err)
        self._data_type, self._fmt, self._k, self._c, self._h, self._w = (
            int(x[0]) for x in (data_type, fmt, k, c, h, w))

    def set_nd(self, data_type, dims, fmt=CUDNN_TENSOR_NCHW):
        """Initializes tensor descriptor.

        Parameters:
            data_type: data type.
            dims: tuple of dimensions.
            fmt: tensor format.
        """
        _dims = cudnnffi.ffi.new("int[]", len(dims))
        _dims[0:len(dims)] = dims
        err = self._lib.cudnnSetFilterNdDescriptor(
            self.handle, data_type, fmt, len(dims), _dims)
        if err:
            raise CU.error("cudnnSetFilterNdDescriptor", err)
        self._data_type, self._fmt = int(data_type), int(fmt)
        self._dims = tuple(int(x) for x in _dims)

    def get_nd(self, n_dims_requested):
        """Fills data_type, fmt and dims properties.

        Parameters:
            n_dims_requested: number of dimensions to extract information from.
        """
        data_type = cudnnffi.ffi.new("cudnnDataType_t *")
        fmt = cudnnffi.ffi.new("cudnnTensorFormat_t *")
        n_dims = cudnnffi.ffi.new("int *")
        dims = cudnnffi.ffi.new("int[]", n_dims_requested)
        err = self._lib.cudnnGetFilterNdDescriptor(
            self.handle, n_dims_requested, data_type, fmt, n_dims, dims)
        if err:
            raise CU.error("cudnnGetFilterNdDescriptor", err)
        self._data_type, self._fmt = int(data_type[0]), int(fmt[0])
        self._dims = tuple(int(x) for x in dims)


class ConvolutionDescriptor(Descriptor):
    """CUDNN convolution descriptor.
    """
    def _create(self):
        handle = cudnnffi.ffi.new("cudnnConvolutionDescriptor_t *")
        err = self._lib.cudnnCreateConvolutionDescriptor(handle)
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
        handle = cudnnffi.ffi.new("cudnnPoolingDescriptor_t *")
        err = self._lib.cudnnCreatePoolingDescriptor(handle)
        if err:
            raise CU.error("cudnnCreatePoolingDescriptor", err)
        self._handle = int(handle[0])

    def _destroy(self):
        self._lib.cudnnDestroyPoolingDescriptor(self.handle)

    def set_2d(self, window_hw, padding_vh, stride_vh, mode=CUDNN_POOLING_MAX,
               maxpooling_nan_opt=CUDNN_NOT_PROPAGATE_NAN):
        """Initializes tensor descriptor into a 4D tensor.

        Parameters:
            window_hw: tuple of ints for pooling window (height, width).
            padding_vh: tuple for padding (vertical, horizontal).
            stride_vh: tuple for stride (vertical, horizontal).
            mode: pooling mode.
        """
        if cudnnffi.cudnn_version < 5000:
            err = self._lib.cudnnSetPooling2dDescriptor(
                self.handle, mode, window_hw[0], window_hw[1],
                padding_vh[0], padding_vh[1], stride_vh[0], stride_vh[1])
        else:
            err = self._lib.cudnnSetPooling2dDescriptor(
                self.handle, mode, maxpooling_nan_opt,
                window_hw[0], window_hw[1],
                padding_vh[0], padding_vh[1], stride_vh[0], stride_vh[1])
        if err:
            raise CU.error("cudnnSetPooling2dDescriptor", err)


class DropoutDescriptor(Descriptor):
    """CUDNN dropout descriptor.
    """
    def _create(self):
        handle = cudnnffi.ffi.new("cudnnDropoutDescriptor_t *")
        err = self._lib.cudnnCreateDropoutDescriptor(handle)
        if err:
            raise CU.error("cudnnCreateDropoutDescriptor", err)
        self._handle = int(handle[0])

    def _destroy(self):
        self._lib.cudnnDestroyDropoutDescriptor(self.handle)


class RNNDescriptor(Descriptor):
    """CUDNN RNN descriptor.
    """
    def __init__(self):
        super(RNNDescriptor, self).__init__()
        self._hidden_size = 0
        self._seq_length = 0
        self._num_layers = 0
        self._dropout_desc = None
        self._input_mode = -1
        self._direction = -1
        self._mode = -1
        self._data_type = -1

    @property
    def hidden_size(self):
        """Size of the internal hidden state for each layer.
        """
        return self._hidden_size

    @property
    def seq_length(self):
        """Number of iterations to unroll over.
        """
        return self._seq_length

    @property
    def num_layers(self):
        """Number of layers.
        """
        return self._num_layers

    @property
    def dropout_desc(self):
        """Dropout descriptor.
        """
        return self._dropout_desc

    @property
    def input_mode(self):
        """Behavior at the input to the first layer.
        """
        return self._input_mode

    @property
    def direction(self):
        """Recurrence pattern, e.g. bidirectional.
        """
        return self._direction

    @property
    def mode(self):
        """The type of RNN.
        """
        return self._mode

    @property
    def data_type(self):
        """Math precision.
        """
        return self._data_type

    @property
    def num_linear_layers(self):
        """Number of linear layers in RNN cell.
        """
        return {
            CUDNN_RNN_RELU: 2,
            CUDNN_RNN_TANH: 2,
            CUDNN_LSTM: 8,
            CUDNN_GRU: 6}.get(self.mode, 0)

    def _create(self):
        handle = cudnnffi.ffi.new("cudnnRNNDescriptor_t *")
        err = self._lib.cudnnCreateRNNDescriptor(handle)
        if err:
            raise CU.error("cudnnCreateRNNDescriptor", err)
        self._handle = int(handle[0])

    def _destroy(self):
        self._lib.cudnnDestroyRNNDescriptor(self.handle)

    def set(self, hidden_size, seq_length, num_layers, dropout_desc,
            input_mode=CUDNN_LINEAR_INPUT, direction=CUDNN_UNIDIRECTIONAL,
            mode=CUDNN_LSTM, data_type=CUDNN_DATA_FLOAT):
        """Initializes RNN descriptor.

        Parameters:
            hidden_size: size of the internal hidden state for each layer.
            seq_length: number of iterations to unroll over.
            num_layers: number of layers.
            dropout_desc: DropoutDescriptor instance.
            input_mode: specifies the behavior at the input to the first layer.
            direction: specifies the recurrence pattern, e.g. bidirectional.
            mode: the type of RNN to compute.
            data_type: math precision.
        """
        err = self._lib.cudnnSetRNNDescriptor(
            self.handle, hidden_size, seq_length, num_layers, dropout_desc,
            input_mode, direction, mode, data_type)
        if err:
            raise CU.error("cudnnSetRNNDescriptor", err)
        self._hidden_size = hidden_size
        self._seq_length = seq_length
        self._num_layers = num_layers
        self._dropout_desc = dropout_desc
        self._input_mode = input_mode
        self._direction = direction
        self._mode = mode
        self._data_type = data_type

    def _descs_to_cffi(self, descs):
        """Converts iterable of the descriptors to cffi array.
        """
        if self.seq_length <= 0:
            raise ValueError("rnn_desc.set() should be called beforehand")
        descs = tuple(descs)
        if len(descs) != self.seq_length:
            raise ValueError(
                "Length of xdescs should be equal to the rnn_desc.seq_length")
        _descs = cudnnffi.ffi.new(
            "cudnnTensorDescriptor_t[]", self.seq_length)
        _descs[0:self.seq_length] = descs
        return _descs


class CUDNN(object):
    """CUDNN functions can be invoked from this class.
    """
    def __init__(self, context):
        self._context = context
        self._lib = None
        context._add_ref(self)
        cudnnffi.initialize()
        handle = cudnnffi.ffi.new("cudnnHandle_t *")
        with context:
            err = cudnnffi.lib.cudnnCreate(handle)
        if err:
            self._handle = None
            raise CU.error("cudnnCreate", err)
        self._lib = cudnnffi.lib  # to hold the reference
        self._handle = int(handle[0])
        self._dropout_desc = None
        self._dropout_states = None

    @property
    def version(self):
        return cudnnffi.cudnn_version

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
        n, c, h, w = (cudnnffi.ffi.new("int *") for _ in range(4))
        err = cudnnffi.lib.cudnnGetConvolution2dForwardOutputDim(
            conv_desc, input_desc, filter_desc, n, c, h, w)
        if err:
            raise CU.error("cudnnGetConvolution2dForwardOutputDim", err)
        return int(n[0]), int(c[0]), int(h[0]), int(w[0])

    @staticmethod
    def get_pooling_2d_forward_output_dim(pooling_desc, input_desc):
        """Returns tuple of n, c, h, w for an output.
        """
        n, c, h, w = (cudnnffi.ffi.new("int *") for _ in range(4))
        err = cudnnffi.lib.cudnnGetPooling2dForwardOutputDim(
            pooling_desc, input_desc, n, c, h, w)
        if err:
            raise CU.error("cudnnGetPooling2dForwardOutputDim", err)
        return int(n[0]), int(c[0]), int(h[0]), int(w[0])

    def get_convolution_forward_algorithm(
            self, src_desc, filter_desc, conv_dec, dest_desc,
            preference=CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, memory_limit=0):
        """Returns forward algorithm based on parameters.
        """
        algo = cudnnffi.ffi.new("cudnnConvolutionFwdAlgo_t *")
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
        size = cudnnffi.ffi.new("size_t *")
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
        size = cudnnffi.ffi.new("size_t *")
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
        algo = cudnnffi.ffi.new("cudnnConvolutionBwdFilterAlgo_t *")
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
        size = cudnnffi.ffi.new("size_t *")
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
            if err:
                raise CU.error("cudnnConvolutionBackwardFilter", err)
            return
        if algo is None:
            # Get the algorithm
            algo = self.get_convolution_backward_filter_algorithm(
                src_desc, diff_desc, conv_desc, grad_desc,
                CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
                if workspace_size else
                CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, workspace_size)
        # Compute weights gradient with the selected algorithm
        err = self._lib.cudnnConvolutionBackwardFilter(
            self.handle, CU.extract_ptr(alpha), src_desc, src_data,
            diff_desc, diff_data, conv_desc,
            algo, 0 if workspace is None else workspace, workspace_size,
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
        algo = cudnnffi.ffi.new("cudnnConvolutionBwdDataAlgo_t *")
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
        size = cudnnffi.ffi.new("size_t *")
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
            if err:
                raise CU.error("cudnnConvolutionBackwardData", err)
            return
        if algo is None:
            # Get the algorithm
            algo = self.get_convolution_backward_data_algorithm(
                filter_desc, diff_desc, conv_desc, grad_desc,
                CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT
                if workspace_size else
                CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, workspace_size)
        # Backpropagate the error with the selected algorithm
        err = self._lib.cudnnConvolutionBackwardData(
            self.handle, CU.extract_ptr(alpha), filter_desc, filter_data,
            diff_desc, diff_data, conv_desc,
            algo, 0 if workspace is None else workspace, workspace_size,
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

    @property
    def dropout_states_size(self):
        """Returns the amount of space required to store the states of the
        random number generators for dropout operation.
        """
        size = cudnnffi.ffi.new("size_t *")
        err = self._lib.cudnnDropoutGetStatesSize(self.handle, size)
        if err:
            raise CU.error("cudnnDropoutGetStatesSize", err)
        return int(size[0])

    def set_dropout_descriptor(self, dropout_desc, dropout=0.5,
                               states=None, states_size=0, seed=0):
        """Sets dropout value, optionally initializing the states.

        Parameters:
            dropout_desc: DropoutDescriptor instance.
            dropout: dropout value.
            states: random states to initialize or None.
            states_size: size of random states in bytes.
            seed: seed to initialize random states.
        """
        err = self._lib.cudnnSetDropoutDescriptor(
            dropout_desc, self.handle, dropout,
            0 if states is None else states, states_size, seed)
        if err:
            raise CU.error("cudnnSetDropoutDescriptor", err)
        # Save the references
        self._dropout_desc = dropout_desc
        self._dropout_states = states

    @property
    def dropout_desc(self):
        """Returns previously set dropout descriptor or None.
        """
        return self._dropout_desc

    @property
    def dropout_states(self):
        """Returns previously set dropout states or None.
        """
        return self._dropout_states

    def dropout_forward(self, dropout_desc, xdesc, x, ydesc, y,
                        reserve_space, reserve_space_size):
        """Does dropout forward propagation.

        Parameters:
            droput_desc: DropoutDescriptor instance.
            xdesc: TensorDescriptor for the input.
            x: input.
            ydesc: TensorDescriptor for the output.
            y: output.
            reserve_space: result state of the dropout operation,
                           should be passed to dropout_backward.
            reserve_space_size: size of the reserve_space.
        """
        err = self._lib.cudnnDropoutForward(
            self.handle, dropout_desc, xdesc, x, ydesc, y,
            reserve_space, reserve_space_size)
        if err:
            raise CU.error("cudnnDropoutForward", err)

    def dropout_backward(self, dropout_desc, dydesc, dy, dxdesc, dx,
                         reserve_space, reserve_space_size):
        """Does dropout backward propagation.

        Parameters:
            droput_desc: DropoutDescriptor instance.
            dydesc: TensorDescriptor for the error to backpropagate.
            dy: error to backpropagate.
            dxdesc: TensorDescriptor for the backpropagated error.
            dx: backpropagated error.
            reserve_space: result state of the dropout operation.
            reserve_space_size: size of the reserve_space.
        """
        err = self._lib.cudnnDropoutBackward(
            self.handle, dropout_desc, dydesc, dy, dxdesc, dx,
            reserve_space, reserve_space_size)
        if err:
            raise CU.error("cudnnDropoutBackward", err)

    def get_rnn_workspace_size(self, rnn_desc, xdescs):
        """Gets the amount of work space required to execute the RNN.

        Parameters:
            rnn_desc: RNNDescriptor instance.
            xdescs: iterable of the descriptors of the input
                    for each unroll step.
        """
        size = cudnnffi.ffi.new("size_t *")
        err = self._lib.cudnnGetRNNWorkspaceSize(
            self.handle, rnn_desc, rnn_desc._descs_to_cffi(xdescs), size)
        if err:
            raise CU.error("cudnnGetRNNWorkspaceSize", err)
        return int(size[0])

    def get_rnn_training_reserve_size(self, rnn_desc, xdescs):
        """Gets the amount of work space required to train the RNN.

        Parameters:
            rnn_desc: RNNDescriptor instance.
            xdescs: iterable of the descriptors of the input
                    for each unroll step.
        """
        size = cudnnffi.ffi.new("size_t *")
        err = self._lib.cudnnGetRNNTrainingReserveSize(
            self.handle, rnn_desc, rnn_desc._descs_to_cffi(xdescs), size)
        if err:
            raise CU.error("cudnnGetRNNTrainingReserveSize", err)
        return int(size[0])

    def get_rnn_params_size(self, rnn_desc, xdescs):
        """Gets the amount of parameter space required to execute the RNN.

        Weights and biases will be stored here.

        Parameters:
            rnn_desc: RNNDescriptor instance.
            xdescs: iterable of the descriptors of the input
                    for each unroll step.
        """
        size = cudnnffi.ffi.new("size_t *")
        err = self._lib.cudnnGetRNNParamsSize(
            self.handle, rnn_desc, rnn_desc._descs_to_cffi(xdescs), size)
        if err:
            raise CU.error("cudnnGetRNNParamsSize", err)
        return int(size[0])

    def get_rnn_lin_layer_matrix_params(self, rnn_desc, layer, xdescs,
                                        wdesc, w, lin_layer_id,
                                        lin_layer_mat_desc):
        """Get a pointer and descriptor for the specified matrix parameter.

        Parameters:
            rnn_desc: RNNDescriptor instance.
            layer: layer number to retrieve info from.
            xdescs: iterable of the descriptors of the input
                    for each unroll step.
            wdesc: descriptor for weights/bias storage space.
            w: weights/bias storage space.
            lin_layer_id: id of the weights matrix.
            lin_layer_mat_desc: FilterDescriptor to be filled.

        Returns:
            MemPtr instance pointed to the requested weights matrix.
        """
        lin_layer_mat = cudnnffi.ffi.new("intptr_t *")
        err = self._lib.cudnnGetRNNLinLayerMatrixParams(
            self.handle, rnn_desc, layer, rnn_desc._descs_to_cffi(xdescs),
            wdesc, w, lin_layer_id, lin_layer_mat_desc, lin_layer_mat)
        if err:
            raise CU.error("cudnnGetRNNLinLayerMatrixParams", err)
        sz = 0
        if isinstance(lin_layer_mat_desc, FilterDescriptor):
            lin_layer_mat_desc.get_nd(3)
            item_size = {CUDNN_DATA_FLOAT: 4, CUDNN_DATA_DOUBLE: 8,
                         CUDNN_DATA_HALF: 2}.get(
                lin_layer_mat_desc.data_type, 0)
            sz = (lin_layer_mat_desc.dims[0] * lin_layer_mat_desc.dims[1] *
                  lin_layer_mat_desc.dims[2] * item_size)
        return MemPtr(self.context, lin_layer_mat[0], w, sz)

    def get_rnn_lin_layer_bias_params(self, rnn_desc, layer, xdescs,
                                      wdesc, w, lin_layer_id,
                                      lin_layer_bias_desc):
        """Get a pointer and descriptor for the specified bias parameter.

        Parameters:
            rnn_desc: RNNDescriptor instance.
            layer: layer number to retrieve info from.
            xdescs: iterable of the descriptors of the input
                    for each unroll step.
            wdesc: descriptor for weights/bias storage space.
            w: weights/bias storage space.
            lin_layer_id: id of the bias vector.
            lin_layer_bias_desc: FilterDescriptor to be filled.

        Returns:
            MemPtr instance pointed to the requested bias vector.
        """
        lin_layer_bias = cudnnffi.ffi.new("intptr_t *")
        err = self._lib.cudnnGetRNNLinLayerBiasParams(
            self.handle, rnn_desc, layer, rnn_desc._descs_to_cffi(xdescs),
            wdesc, w, lin_layer_id, lin_layer_bias_desc, lin_layer_bias)
        if err:
            raise CU.error("cudnnGetRNNLinLayerBiasParams", err)
        sz = 0
        if isinstance(lin_layer_bias_desc, FilterDescriptor):
            lin_layer_bias_desc.get_nd(3)
            item_size = {CUDNN_DATA_FLOAT: 4, CUDNN_DATA_DOUBLE: 8,
                         CUDNN_DATA_HALF: 2}.get(
                lin_layer_bias_desc.data_type, 0)
            sz = (lin_layer_bias_desc.dims[0] * lin_layer_bias_desc.dims[1] *
                  lin_layer_bias_desc.dims[2] * item_size)
        return MemPtr(self.context, lin_layer_bias[0], w, sz)

    def rnn_forward_inference(self, rnn_desc, xdescs, x, hx_desc, hx,
                              cx_desc, cx, wdesc, w, ydescs, y, hy_desc, hy,
                              cy_desc, cy, workspace, workspace_size):
        """Does forward inference of RNN up to it's unroll size.

        Parameters:
            rnn_desc: RNNDescriptor instance.
            xdescs: iterable of the descriptors of the input
                    for each unroll step.
            x: single array with inputs for all unrolls.
            hx_desc: descriptor for initial hidden states.
            hx: initial hidden states (can be None).
            cx_desc: descriptor for initial memory cells.
            cx: initial memory cells (can be None).
            wdesc: descriptor for weights & bias storage space.
            w: weights & bias storage space.
            ydescs: iterable of the descriptors of the output
                    for each unroll step.
            y: single array with outputs for all unrolls.
            hy_desc: descriptor for the final hidden states.
            hy: final hidden states (can be None).
            cy_desc: descriptor for the final memory cells.
            cy: final memory cells (can be None).
            workspace: workspace with size >= get_rnn_workspace_size().
            workspace_size: workspace size in bytes.
        """
        err = self._lib.cudnnRNNForwardInference(
            self.handle, rnn_desc, rnn_desc._descs_to_cffi(xdescs), x,
            hx_desc, 0 if hx is None else hx, cx_desc, 0 if cx is None else cx,
            wdesc, w, rnn_desc._descs_to_cffi(ydescs), y,
            hy_desc, 0 if hy is None else hy, cy_desc, 0 if cy is None else cy,
            workspace, workspace_size)
        if err:
            raise CU.error("cudnnRNNForwardInference", err)

    def rnn_forward_training(self, rnn_desc, xdescs, x, hx_desc, hx,
                             cx_desc, cx, wdesc, w, ydescs, y, hy_desc, hy,
                             cy_desc, cy, workspace, workspace_size,
                             reserve_space, reserve_space_size):
        """Does forward inference for RNN training.

        Parameters:
            rnn_desc: RNNDescriptor instance.
            xdescs: iterable of the descriptors of the input
                    for each unroll step.
            x: single array with inputs for all unrolls.
            hx_desc: descriptor for initial hidden states.
            hx: initial hidden states (can be None).
            cx_desc: descriptor for initial memory cells.
            cx: initial memory cells (can be None).
            wdesc: descriptor for weights & bias storage space.
            w: weights & bias storage space.
            ydescs: iterable of the descriptors of the output
                    for each unroll step.
            y: single array with outputs for all unrolls.
            hy_desc: descriptor for the final hidden states.
            hy: final hidden states (can be None).
            cy_desc: descriptor for the final memory cells.
            cy: final memory cells (can be None).
            workspace: workspace with size >= get_rnn_workspace_size().
            workspace_size: workspace size in bytes.
            reserve_space: additional space for RNN training
                           with size >= get_rnn_training_reserve_size().
            reserve_space_size: size in bytes of reserve_space.
        """
        err = self._lib.cudnnRNNForwardTraining(
            self.handle, rnn_desc, rnn_desc._descs_to_cffi(xdescs), x,
            hx_desc, 0 if hx is None else hx, cx_desc, 0 if cx is None else cx,
            wdesc, w, rnn_desc._descs_to_cffi(ydescs), y,
            hy_desc, 0 if hy is None else hy, cy_desc, 0 if cy is None else cy,
            workspace, workspace_size, reserve_space, reserve_space_size)
        if err:
            raise CU.error("cudnnRNNForwardTraining", err)

    def rnn_backward_data(self, rnn_desc, y_descs, y, dy_descs, dy,
                          dhy_desc, dhy, dcy_desc, dcy, wdesc, w,
                          hx_desc, hx, cx_desc, cx, dx_descs, dx,
                          dhx_desc, dhx, dcx_desc, dcx,
                          workspace, workspace_size,
                          reserve_space, reserve_space_size):
        """Backpropagates the error through RNN.

        Parameters:
            rnn_desc: RNNDescriptor instance.
            y_descs: descriptors of outputs for all unroll steps.
            y: single array with outputs for all unroll steps.
            dy_descs: descriptors of output gradients for all unroll steps.
            dy: single array with output gradients for all unroll steps.
            dhy_desc: descriptor for gradients at the final hidden state.
            dhy: gradients at the final hidden state (can be None).
            dcy_desc: descriptor for gradients at the final memory cell.
            dcy: gradients at the final memory cell (can be None).
            wdesc: descriptor for weights & bias storage space.
            w: weights & bias storage space.
            hx_desc: descriptor for initial hidden states.
            hx: initial hidden states (can be None).
            cx_desc: descriptor for initial memory cells.
            cx: initial memory cells (can be None).
            dx_descs: descriptors for input gradients at each unroll step.
            dx: single array for input gradients at each unroll step.
            dhx_desc: descriptor for gradient at the initial hidden states.
            dhx: gradient at the initial hidden states (can be None).
            dcx_desc: descriptor for gradient at the initial memory cells.
            dcx: gradient at the initial memory cells (can be None).
            workspace: workspace with size >= get_rnn_workspace_size().
            workspace_size: workspace size in bytes.
            reserve_space: additional space for RNN training
                           with size >= get_rnn_training_reserve_size().
            reserve_space_size: size in bytes of reserve_space.
        """
        err = self._lib.cudnnRNNBackwardData(
            self.handle, rnn_desc, rnn_desc._descs_to_cffi(y_descs), y,
            rnn_desc._descs_to_cffi(dy_descs), dy,
            dhy_desc, 0 if dhy is None else dhy,
            dcy_desc, 0 if dcy is None else dcy, wdesc, w,
            hx_desc, 0 if hx is None else hx,
            cx_desc, 0 if cx is None else cx,
            rnn_desc._descs_to_cffi(dx_descs), dx,
            dhx_desc, 0 if dhx is None else dhx,
            dcx_desc, 0 if dcx is None else dcx,
            workspace, workspace_size, reserve_space, reserve_space_size)
        if err:
            raise CU.error("cudnnRNNBackwardData", err)

    def rnn_backward_weights(self, rnn_desc, xdescs, x, hx_desc, hx,
                             y_descs, y, workspace, workspace_size,
                             dw_desc, dw, reserve_space, reserve_space_size):
        """Backpropagates the error through RNN.

        Parameters:
            rnn_desc: RNNDescriptor instance.
            xdescs: iterable of the descriptors of the input
                    for each unroll step.
            x: single array with inputs for all unrolls.
            hx_desc: descriptor for initial hidden states.
            hx: initial hidden states (can be None).
            y_descs: descriptors of outputs for all unroll steps.
            y: single array with outputs for all unroll steps.
            workspace: workspace with size >= get_rnn_workspace_size().
            workspace_size: workspace size in bytes.
            dw_desc: descriptor of storage space for weights/biases gradients.
            dw: storage space for weights/biases gradients.
            reserve_space: additional space for RNN training
                           with size >= get_rnn_training_reserve_size().
            reserve_space_size: size in bytes of reserve_space.
        """
        err = self._lib.cudnnRNNBackwardWeights(
            self.handle, rnn_desc, rnn_desc._descs_to_cffi(xdescs), x,
            hx_desc, 0 if hx is None else hx,
            rnn_desc._descs_to_cffi(y_descs), y,
            workspace, workspace_size, dw_desc, dw,
            reserve_space, reserve_space_size)
        if err:
            raise CU.error("cudnnRNNBackwardWeights", err)

    def softmax_forward(self, alpha, x_desc, x, beta, y_desc, y,
                        algo=CUDNN_SOFTMAX_ACCURATE,
                        mode=CUDNN_SOFTMAX_MODE_INSTANCE):
        """Computes Softmax.

        y = beta * y + alpha * softmax(x).

        Parameters:
            alpha: numpy array with single value.
            x_desc: descriptor for an input.
            x: input.
            beta: numpy array with single value.
            y_desc: descriptor for an output.
            y: output.
            algo: algorithm for the Softmax computation.
            mode: on what input channels to apply the Softmax.
        """
        err = self._lib.cudnnSoftmaxForward(
            self.handle, algo, mode, CU.extract_ptr(alpha),
            x_desc, x, CU.extract_ptr(beta), y_desc, y)
        if err:
            raise CU.error("cudnnSoftmaxForward", err)

    def softmax_backward(self, alpha, y_desc, y, dy_desc, dy,
                         beta, dx_desc, dx,
                         algo=CUDNN_SOFTMAX_ACCURATE,
                         mode=CUDNN_SOFTMAX_MODE_INSTANCE):
        """Computes gradient for the Softmax.

        dx = beta * dx + alpha * GradSoftmax(dy).

        Parameters:
            alpha: numpy array with single value.
            y_desc: descriptor for an output.
            y: output with the computed Softmax.
            dy_desc: descriptor for an error to backpropagate.
            dy: error to backpropagate.
            beta: numpy array with single value.
            dx_desc: descriptor for a backpropagated error.
            dx: backpropagated error.
            algo: algorithm for the Softmax computation.
            mode: on what input channels to apply the Softmax.
        """
        err = self._lib.cudnnSoftmaxBackward(
            self.handle, algo, mode, CU.extract_ptr(alpha),
            y_desc, y, dy_desc, dy, CU.extract_ptr(beta), dx_desc, dx)
        if err:
            raise CU.error("cudnnSoftmaxBackward", err)

    def _release(self):
        if self._lib is not None and self.handle is not None:
            self._lib.cudnnDestroy(self.handle)
            self._handle = None

    def __del__(self):
        if self.context.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self._release()
        self.context._del_ref(self)
