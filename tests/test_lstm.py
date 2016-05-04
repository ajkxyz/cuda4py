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
Tests cuDNN's LSTM with numeric differentiation.
"""
import cuda4py as cu
import cuda4py.cudnn as cudnn
import gc
import logging
import math
import numpy
import os
import unittest


ITEMSIZE = 8
DTYPE = numpy.float64
CUTYPE = cudnn.CUDNN_DATA_DOUBLE


class NumDiff(object):
    """Numeric differentiation helper.

    WARNING: it is invalid for single precision float data type.
    """
    def __init__(self):
        self.h = 1.0e-6
        self.points = (2.0 * self.h, self.h, -self.h, -2.0 * self.h)
        self.coeffs = numpy.array((-1.0, 8.0, -8.0, 1.0),
                                  dtype=DTYPE)
        self.divizor = 12.0 * self.h
        self.errs = numpy.zeros_like(self.points, dtype=DTYPE)

    @property
    def derivative(self):
        return (self.errs * self.coeffs).sum() / self.divizor

    @staticmethod
    def sse(y, t):
        return numpy.square(y - t).sum() * 0.5

    def check_diff(self, x, y, target, dx, forward, f_err=None):
        if f_err is None:
            f_err = NumDiff.sse
        assert len(x.shape) == len(dx.shape)
        assert y.shape == target.shape
        max_diff = 0.0
        for i, vle in numpy.ndenumerate(x):
            for j, d in enumerate(self.points):
                x[i] = vle + d
                forward()
                self.errs[j] = f_err(y, target)
            x[i] = vle
            if abs(dx[i]) > 1.0e-10:
                max_diff = max(max_diff, abs(1.0 - self.derivative / dx[i]))
            else:
                logging.debug("err_x close to zero for index %d", i)
        logging.debug(
            "max_diff is %.6f %s",
            max_diff, "FAIL, should be less than 0.5" if max_diff >= 0.5
            else "OK")


class Test(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.old_env = os.environ.get("CUDA_DEVICE")
        if self.old_env is None:
            os.environ["CUDA_DEVICE"] = "0"
        self.ctx = cu.Devices().create_some_context()
        self.cudnn = cudnn.CUDNN(self.ctx)
        self.path = os.path.dirname(__file__)
        if not len(self.path):
            self.path = "."

    def tearDown(self):
        if self.old_env is None:
            del os.environ["CUDA_DEVICE"]
        else:
            os.environ["CUDA_DEVICE"] = self.old_env
        del self.old_env
        del self.cudnn
        del self.ctx
        gc.collect()

    def test_numdiff(self):
        logging.debug("ENTER: test_numdiff")
        numdiff = NumDiff()
        for i in range(10):
            x = 0.1 * i
            dx = math.cos(x)
            for j, p in enumerate(numdiff.points):
                numdiff.errs[j] = math.sin(x + p)
            delta = abs(dx - numdiff.derivative)
            logging.debug("%.3e %.3e", delta,
                          abs(1.0 - numdiff.derivative / dx)
                          if numdiff.derivative else 0)
            self.assertLess(delta, 1.0e-6)
        logging.debug("EXIT: test_numdiff")

    def test_lstm(self):
        if self.cudnn.version < 5000:
            return
        logging.debug("ENTER: test_lstm")

        drop = cudnn.DropoutDescriptor()
        drop_states = cu.MemAlloc(self.ctx, self.cudnn.dropout_states_size)
        self.cudnn.set_dropout_descriptor(
            drop, 0.0, drop_states, drop_states.size, 1234)

        rnn = cudnn.RNNDescriptor()
        self.assertEqual(rnn.hidden_size, 0)
        self.assertEqual(rnn.seq_length, 0)
        self.assertEqual(rnn.num_layers, 0)
        self.assertIsNone(rnn.dropout_desc)
        self.assertEqual(rnn.input_mode, -1)
        self.assertEqual(rnn.direction, -1)
        self.assertEqual(rnn.mode, -1)
        self.assertEqual(rnn.data_type, -1)
        self.assertEqual(rnn.num_linear_layers, 0)

        batch_size = 8
        x_arr = numpy.zeros((batch_size, 16),  # minibatch, input size
                            dtype=DTYPE)
        numpy.random.seed(1234)
        x_arr[:] = numpy.random.rand(x_arr.size).reshape(x_arr.shape) - 0.5
        x_desc = cudnn.TensorDescriptor()
        # Set input as 3-dimensional like in cudnn example.
        x_desc.set_nd(CUTYPE,
                      (x_arr.shape[1], x_arr.shape[0], 1))
        n_unroll = 5
        hidden_size = 16
        n_layers = 3

        def assert_values():
            self.assertEqual(rnn.hidden_size, hidden_size)
            self.assertEqual(rnn.seq_length, n_unroll)
            self.assertEqual(rnn.num_layers, n_layers)
            self.assertIs(rnn.dropout_desc, drop)
            self.assertEqual(rnn.input_mode, cudnn.CUDNN_LINEAR_INPUT)
            self.assertEqual(rnn.direction, cudnn.CUDNN_UNIDIRECTIONAL)
            self.assertEqual(rnn.mode, cudnn.CUDNN_LSTM)
            self.assertEqual(rnn.data_type, CUTYPE)
            self.assertEqual(rnn.num_linear_layers, 8)

        # Full syntax
        rnn = cudnn.RNNDescriptor()
        rnn.set(hidden_size, n_unroll, n_layers, drop,
                input_mode=cudnn.CUDNN_LINEAR_INPUT,
                direction=cudnn.CUDNN_UNIDIRECTIONAL,
                mode=cudnn.CUDNN_LSTM, data_type=CUTYPE)
        assert_values()

        x_descs = tuple(x_desc for _i in range(n_unroll))

        def get_sz(func):
            sz = func(rnn, x_descs)
            self.assertIsInstance(sz, int)
            return sz

        sz_work = get_sz(self.cudnn.get_rnn_workspace_size)
        logging.debug("RNN workspace size for %s with %d unrolls is %d",
                      x_arr.shape, n_unroll, sz_work)

        sz_train = get_sz(self.cudnn.get_rnn_training_reserve_size)
        logging.debug("RNN train size for %s with %d unrolls is %d",
                      x_arr.shape, n_unroll, sz_train)

        sz_weights = get_sz(self.cudnn.get_rnn_params_size)
        logging.debug("RNN weights size for %s with %d unrolls is %d",
                      x_arr.shape, n_unroll, sz_weights)
        sz_expected = ITEMSIZE * (
            4 * (x_arr.shape[1] + hidden_size + 2) * hidden_size +
            4 * (hidden_size + hidden_size + 2) * hidden_size * (n_layers - 1))
        self.assertEqual(sz_weights, sz_expected)

        weights_desc = cudnn.FilterDescriptor()
        weights_desc.set_nd(CUTYPE, (sz_weights // ITEMSIZE, 1, 1))
        weights = cu.MemAlloc(self.ctx, sz_weights)
        weights_arr = numpy.random.rand(sz_weights // ITEMSIZE).astype(DTYPE)
        weights_arr -= 0.5
        weights_arr *= 0.1
        weights.to_device(weights_arr)
        w_desc = cudnn.FilterDescriptor()
        w = self.cudnn.get_rnn_lin_layer_matrix_params(
            rnn, 0, x_descs, weights_desc, weights, 0, w_desc)
        logging.debug("Got matrix 0 of dimensions: %s, fmt=%d, sz=%d",
                      w_desc.dims, w_desc.fmt, w.size)
        self.assertEqual(w.size, hidden_size * x_arr.shape[1] * ITEMSIZE)

        b_desc = cudnn.FilterDescriptor()
        b = self.cudnn.get_rnn_lin_layer_bias_params(
            rnn, 0, x_descs, weights_desc, weights, 0, b_desc)
        logging.debug("Got bias 0 of dimensions: %s, fmt=%d, sz=%d",
                      b_desc.dims, b_desc.fmt, b.size)
        self.assertEqual(b.size, hidden_size * ITEMSIZE)

        work_buf = cu.MemAlloc(self.ctx, sz_work)
        work_buf.memset32_async()
        x = cu.MemAlloc(self.ctx, x_arr.nbytes * n_unroll)
        for i in range(n_unroll):  # will feed the same input
            x.to_device(x_arr, x_arr.nbytes * i, x_arr.nbytes)
        y_arr = numpy.zeros((n_unroll, batch_size, hidden_size),
                            dtype=DTYPE)
        y = cu.MemAlloc(self.ctx, y_arr)
        hx_arr = numpy.zeros((n_layers, batch_size, hidden_size),
                             dtype=DTYPE)
        hx_arr[:] = numpy.random.rand(hx_arr.size).reshape(hx_arr.shape)
        hx_arr -= 0.5
        hx = cu.MemAlloc(self.ctx, hx_arr)
        hy = cu.MemAlloc(self.ctx, hx.size)
        hy.memset32_async()
        cx_arr = numpy.zeros((n_layers, batch_size, hidden_size), dtype=DTYPE)
        cx_arr[:] = numpy.random.rand(cx_arr.size).reshape(cx_arr.shape)
        cx_arr -= 0.5
        cx = cu.MemAlloc(self.ctx, cx_arr)
        cy = cu.MemAlloc(self.ctx, cx.size)
        cy.memset32_async()

        y_desc = cudnn.TensorDescriptor()
        y_desc.set_nd(CUTYPE, (hidden_size, batch_size, 1))
        y_descs = tuple(y_desc for _i in range(n_unroll))

        h_desc = cudnn.TensorDescriptor()
        h_desc.set_nd(CUTYPE, (hidden_size, batch_size, n_layers))

        train_buf = cu.MemAlloc(self.ctx, sz_train)
        train_buf.memset32_async()
        self.cudnn.rnn_forward_training(
            rnn, x_descs, x, h_desc, hx, h_desc, cx, weights_desc, weights,
            y_descs, y, h_desc, hy, h_desc, cy, work_buf, sz_work,
            train_buf, sz_train)
        self.ctx.synchronize()
        logging.debug("Forward training done")

        y.to_host(y_arr)
        target = numpy.random.rand(
            y_arr.size).reshape(y_arr.shape).astype(y_arr.dtype) - 0.5
        dy_arr = y_arr - target

        dy = cu.MemAlloc(self.ctx, dy_arr)
        dhy = cu.MemAlloc(self.ctx, hx.size)
        dhy.memset32_async()
        dcy = cu.MemAlloc(self.ctx, cx.size)
        dcy.memset32_async()
        dx_arr = numpy.zeros_like(x_arr)
        dx = cu.MemAlloc(self.ctx, dx_arr)
        dhx_arr = numpy.zeros_like(hx_arr)
        dhx = cu.MemAlloc(self.ctx, dhx_arr)
        dcx_arr = numpy.zeros_like(cx_arr)
        dcx = cu.MemAlloc(self.ctx, dcx_arr)

        self.cudnn.rnn_backward_data(
            rnn, y_descs, y, y_descs, dy,
            h_desc, dhy, h_desc, dcy, weights_desc, weights,
            h_desc, hx, h_desc, cx,
            x_descs, dx, h_desc, dhx, h_desc, dcx,
            work_buf, sz_work, train_buf, sz_train)
        logging.debug("Backpropagation done")

        dx.to_host(dx_arr)
        dhx.to_host(dhx_arr)
        dcx.to_host(dcx_arr)

        def forward():
            x.to_device_async(x_arr)
            hx.to_device_async(hx_arr)
            cx.to_device_async(cx_arr)
            self.cudnn.rnn_forward_inference(
                rnn, x_descs, x,
                h_desc, hx, h_desc, cx, weights_desc, weights,
                y_descs, y, h_desc, hy, h_desc, cy, work_buf, sz_work)
            y.to_host(y_arr)

        numdiff = NumDiff()

        logging.debug("Checking dx...")
        numdiff.check_diff(x_arr, y_arr, target, dx_arr, forward)

        logging.debug("Checking dhx...")
        numdiff.check_diff(hx_arr, y_arr, target, dhx_arr, forward)

        logging.debug("Checking dcx...")
        numdiff.check_diff(cx_arr, y_arr, target, dcx_arr, forward)

        logging.debug("EXIT: test_lstm")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
