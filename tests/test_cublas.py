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
import unittest
import logging
import numpy
import cuda4py as cu
import cuda4py.blas as blas
import os


class Test(unittest.TestCase):
    def setUp(self):
        logging.basicConfig(level=logging.DEBUG)
        self.old_env = os.environ.get("CUDA_DEVICE")
        if self.old_env is None:
            os.environ["CUDA_DEVICE"] = "0"
        self.ctx = cu.Devices().create_some_context()
        self.blas = blas.CUBLAS(self.ctx)

    def tearDown(self):
        if self.old_env is None:
            del os.environ["CUDA_DEVICE"]
        else:
            os.environ["CUDA_DEVICE"] = self.old_env
        del self.old_env

    def _test_gemm(self, gemm, dtype):
        a = numpy.zeros([127, 353], dtype=dtype)
        b = numpy.zeros([135, a.shape[1]], dtype=dtype)
        c = numpy.zeros([a.shape[0], b.shape[0]], dtype=dtype)
        numpy.random.seed(123)
        a[:] = numpy.random.rand(a.size).astype(dtype).reshape(a.shape) - 0.5
        b[:] = numpy.random.rand(b.size).astype(dtype).reshape(b.shape) - 0.5
        gold_c = numpy.dot(a, b.transpose())
        a_buf = cu.MemAlloc(self.ctx, a.nbytes)
        b_buf = cu.MemAlloc(self.ctx, b.nbytes)
        c_buf = cu.MemAlloc(self.ctx, c.nbytes * 2)
        alpha = numpy.ones(1, dtype=dtype)
        beta = numpy.zeros(1, dtype=dtype)
        a_buf.to_device_async(a)
        b_buf.to_device_async(b)
        c_buf.to_device_async(c)
        c_buf.to_device_async(c, c.nbytes)
        gemm(blas.CUBLAS_OP_T, blas.CUBLAS_OP_N,
             b.shape[0], a.shape[0], a.shape[1],
             alpha, b_buf, a_buf, beta, c_buf)
        c_buf.to_host(c)
        max_diff = numpy.fabs(c - gold_c).max()
        self.assertLess(max_diff, 0.0001)
        c_buf.to_host(c, c.nbytes)
        max_diff = numpy.fabs(c).max()
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
