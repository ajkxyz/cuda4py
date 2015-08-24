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
Tests some of the api in cuda4py.blas._cublas module.
"""
import cuda4py as cu
import cuda4py.blas as blas
import gc
import logging
import numpy
import os
import unittest


class Test(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.old_env = os.environ.get("CUDA_DEVICE")
        if self.old_env is None:
            os.environ["CUDA_DEVICE"] = "0"
        self.ctx = cu.Devices().create_some_context()
        self.blas = blas.CUBLAS(self.ctx)
        self.path = os.path.dirname(__file__)
        if not len(self.path):
            self.path = "."

    def tearDown(self):
        if self.old_env is None:
            del os.environ["CUDA_DEVICE"]
        else:
            os.environ["CUDA_DEVICE"] = self.old_env
        del self.old_env
        del self.blas
        del self.ctx
        gc.collect()

    def test_constants(self):
        self.assertEqual(blas.CUBLAS_OP_N, 0)
        self.assertEqual(blas.CUBLAS_OP_T, 1)
        self.assertEqual(blas.CUBLAS_OP_C, 2)

        self.assertEqual(blas.CUBLAS_DATA_FLOAT, 0)
        self.assertEqual(blas.CUBLAS_DATA_DOUBLE, 1)
        self.assertEqual(blas.CUBLAS_DATA_HALF, 2)
        self.assertEqual(blas.CUBLAS_DATA_INT8, 3)

        self.assertEqual(blas.CUBLAS_POINTER_MODE_HOST, 0)
        self.assertEqual(blas.CUBLAS_POINTER_MODE_DEVICE, 1)

        self.assertEqual(blas.CUBLAS_STATUS_SUCCESS, 0)
        self.assertEqual(blas.CUBLAS_STATUS_NOT_INITIALIZED, 1)
        self.assertEqual(blas.CUBLAS_STATUS_ALLOC_FAILED, 3)
        self.assertEqual(blas.CUBLAS_STATUS_INVALID_VALUE, 7)
        self.assertEqual(blas.CUBLAS_STATUS_ARCH_MISMATCH, 8)
        self.assertEqual(blas.CUBLAS_STATUS_MAPPING_ERROR, 11)
        self.assertEqual(blas.CUBLAS_STATUS_EXECUTION_FAILED, 13)
        self.assertEqual(blas.CUBLAS_STATUS_INTERNAL_ERROR, 14)
        self.assertEqual(blas.CUBLAS_STATUS_NOT_SUPPORTED, 15)
        self.assertEqual(blas.CUBLAS_STATUS_LICENSE_ERROR, 16)

    def test_errors(self):
        idx = cu.CU.ERRORS[blas.CUBLAS_STATUS_NOT_INITIALIZED].find(" | ")
        self.assertGreater(idx, 0)

    def _test_gemm(self, gemm, dtype):
        for mode in (blas.CUBLAS_POINTER_MODE_HOST,
                     blas.CUBLAS_POINTER_MODE_DEVICE):
            self._test_gemm_with_mode(gemm, dtype, mode)

    def _test_gemm_with_mode(self, gemm, dtype, mode):
        self.blas.set_pointer_mode(mode)
        a = numpy.zeros([127, 353], dtype=dtype)
        b = numpy.zeros([135, a.shape[1]], dtype=dtype)
        c = numpy.zeros([a.shape[0], b.shape[0]], dtype=dtype)
        try:
            numpy.random.seed(123)
        except AttributeError:  # PyPy workaround
            pass
        a[:] = numpy.random.rand(a.size).astype(dtype).reshape(a.shape) - 0.5
        b[:] = numpy.random.rand(b.size).astype(dtype).reshape(b.shape) - 0.5
        gold_c = numpy.dot(a.astype(numpy.float64),
                           b.transpose().astype(numpy.float64))
        a_buf = cu.MemAlloc(self.ctx, a.nbytes)
        b_buf = cu.MemAlloc(self.ctx, b.nbytes)
        c_buf = cu.MemAlloc(self.ctx, c.nbytes * 2)

        alpha = numpy.ones(
            1, dtype={numpy.float16: numpy.float32}.get(dtype, dtype))
        beta = numpy.zeros(
            1, dtype={numpy.float16: numpy.float32}.get(dtype, dtype))
        if mode == blas.CUBLAS_POINTER_MODE_DEVICE:
            alpha = cu.MemAlloc(self.ctx, alpha)
            beta = cu.MemAlloc(self.ctx, beta)

        a_buf.to_device_async(a)
        b_buf.to_device_async(b)
        c_buf.to_device_async(c)
        c_buf.to_device_async(c, c.nbytes)

        gemm(blas.CUBLAS_OP_T, blas.CUBLAS_OP_N,
             b.shape[0], a.shape[0], a.shape[1],
             alpha, b_buf, a_buf, beta, c_buf)

        c_buf.to_host(c)
        max_diff = numpy.fabs(gold_c - c.astype(numpy.float64)).max()
        logging.debug("Maximum difference is %.6f", max_diff)
        self.assertLess(
            max_diff, {numpy.float32: 1.0e-5, numpy.float64: 1.0e-13,
                       numpy.float16: 3.0e-3}[dtype])
        c_buf.to_host(c, c.nbytes)
        max_diff = numpy.fabs(c).max()

        # To avoid destructor call before gemm completion
        del beta
        del alpha

        self.assertEqual(max_diff, 0,
                         "Written some values outside of the target array")

    def test_sgemm(self):
        logging.debug("ENTER: test_sgemm")
        with self.ctx:
            self._test_gemm(self.blas.sgemm, numpy.float32)
        logging.debug("EXIT: test_sgemm")

    def test_dgemm(self):
        logging.debug("ENTER: test_dgemm")
        with self.ctx:
            self._test_gemm(self.blas.dgemm, numpy.float64)
        logging.debug("EXIT: test_dgemm")

    def test_sgemm_ex(self):
        logging.debug("ENTER: test_sgemm_ex")
        with self.ctx:
            self._test_gemm(self.blas.sgemm_ex, numpy.float16)
        logging.debug("EXIT: test_sgemm_ex")

    def test_kernel(self):
        logging.debug("ENTER: test_kernel")
        if self.ctx.device.compute_capability < (3, 5):
            logging.debug("Requires compute capability >= (3, 5)")
            logging.debug("EXIT: test_kernel")
            return
        with self.ctx:
            module = cu.Module(
                self.ctx, source_file=("%s/cublas.cu" % self.path),
                nvcc_options2=cu.Module.OPTIONS_CUBLAS)
            logging.debug("Compiled")
            f = module.create_function("test")
            logging.debug("Got function")

            n = 256
            a = numpy.random.rand(n, n).astype(numpy.float32)
            b = numpy.random.rand(n, n).astype(numpy.float32)
            c = numpy.zeros_like(a)
            c_gold = numpy.dot(a.transpose(), b.transpose()).transpose()
            a_ = cu.MemAlloc(self.ctx, a)
            b_ = cu.MemAlloc(self.ctx, b)
            c_ = cu.MemAlloc(self.ctx, c)
            zero_ = cu.MemAlloc(self.ctx, numpy.zeros(1, dtype=numpy.float32))
            one_ = cu.MemAlloc(self.ctx, numpy.ones(1, dtype=numpy.float32))
            logging.debug("Allocated arrays")

            f.set_args(numpy.array([n], dtype=numpy.int64), one_, a_, b_,
                       zero_, c_)
            logging.debug("Set args")

            f((1, 1, 1), (1, 1, 1))
            logging.debug("Executed")

            c_.to_host(c)
            max_diff = numpy.fabs(c - c_gold).max()
            logging.debug("Maximum difference is %.6f", max_diff)
            self.assertLess(max_diff, 1.0e-3)
        logging.debug("EXIT: test_kernel")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
