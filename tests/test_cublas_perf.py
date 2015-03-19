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
Tests CUBLAS matrix multiplication and it's performance,
comparing time required for:
    1. Call Sgemm from CUDA kernel, receiving blas handle as parameter;
    2. Call Sgemm from CUDA kernel, creating blas handle every time;
    3. Call Sgemm from host (CUBLAS shared library).

2. executes in about 2 times longer, so cublasCreate/cublasDestroy
is a costly operation.

In the simple case 3. can be the winner (depending on the system),
so calling CUBLAS from CUDA kernel may be not neccessary when calling pattern
does not depend on the previous CUDA kernel launches.
"""
import cuda4py as cu
import cuda4py.blas as cublas
import gc
import logging
import numpy
import os
import time
import unittest


class Test(unittest.TestCase):
    def setUp(self):
        self.old_env = os.environ.get("CUDA_DEVICE")
        if self.old_env is None:
            os.environ["CUDA_DEVICE"] = "0"
        self.path = os.path.dirname(__file__)
        if not len(self.path):
            self.path = "."

    def tearDown(self):
        if self.old_env is None:
            del os.environ["CUDA_DEVICE"]
        else:
            os.environ["CUDA_DEVICE"] = self.old_env
        del self.old_env
        gc.collect()

    def test_kernel(self):
        path = os.path.dirname(__file__)
        if not len(path):
            path = "."
        with cu.Devices().create_some_context() as ctx:
            if ctx.device.compute_capability < (3, 5):
                return

            logging.info("Compiling...")
            m = ctx.create_module(
                source_file=(path + "/cublas_perf.cu"),
                nvcc_options2=cu.Module.OPTIONS_CUBLAS)
            create_cublas = m.create_function("create_cublas")
            destroy_cublas = m.create_function("destroy_cublas")
            test = m.create_function("test")
            test_full = m.create_function("test_full")
            dummy = m.create_function("dummy")
            logging.info("Done")

            blas = numpy.zeros(1, dtype=numpy.uint64)
            blas_ = ctx.mem_alloc(blas)
            create_cublas.set_args(blas_)
            create_cublas((1, 1, 1))
            blas_.to_host(blas)

            n = 256
            one_ = ctx.mem_alloc(numpy.ones(1, dtype=numpy.float32))
            zero_ = ctx.mem_alloc(numpy.zeros(1, dtype=numpy.float32))

            a = numpy.random.rand(n, n).astype(numpy.float32)
            b = numpy.random.rand(n, n).astype(numpy.float32)
            c = numpy.zeros_like(a)

            c_gold = numpy.dot(a.transpose(), b.transpose()).transpose()

            a_ = ctx.mem_alloc(a)
            b_ = ctx.mem_alloc(b)
            c_ = ctx.mem_alloc(c)

            N = 10000

            test.set_args(numpy.array([N], dtype=numpy.int64),
                          blas,
                          numpy.array([n], dtype=numpy.int64),
                          one_, a_, b_, zero_, c_)
            ctx.synchronize()
            t0 = time.time()
            test((1, 1, 1))
            ctx.synchronize()
            dt = time.time() - t0
            logging.info("With external blas handle completed in %.6f sec", dt)

            destroy_cublas.set_args(blas)
            destroy_cublas((1, 1, 1))

            c_.to_host(c)
            max_diff = numpy.fabs(c - c_gold).max()
            logging.info("max_diff = %.6f", max_diff)
            self.assertLess(max_diff, 1.0e-3)

            test_full.set_args(numpy.array([N], dtype=numpy.int64),
                               numpy.array([n], dtype=numpy.int64),
                               one_, a_, b_, zero_, c_)
            ctx.synchronize()
            t0 = time.time()
            test_full((1, 1, 1))
            ctx.synchronize()
            dt = time.time() - t0
            logging.info("With local blas handle completed in %.6f sec", dt)

            c_.to_host(c)
            max_diff = numpy.fabs(c - c_gold).max()
            logging.info("max_diff = %.6f", max_diff)
            self.assertLess(max_diff, 1.0e-3)

            blas = cublas.CUBLAS(ctx)
            ctx.synchronize()
            one = numpy.ones(1, dtype=numpy.float32)
            zero = numpy.zeros(1, dtype=numpy.float32)
            dummy.set_args(c_, c_)
            t0 = time.time()
            for i in range(N):
                blas.sgemm(cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_N, n, n, n,
                           one, a_ if i & 1 else b_, b_ if i & 1 else a_, zero,
                           c_, n, n, n)
                dummy((1, 1, 1))  # interleave with some dummy kernel
            ctx.synchronize()
            dt = time.time() - t0
            logging.info("With shared library blas completed in %.6f sec", dt)

            c_.to_host(c)
            max_diff = numpy.fabs(c - c_gold).max()
            logging.info("max_diff = %.6f", max_diff)
            self.assertLess(max_diff, 1.0e-3)

        logging.info("Succeeded")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
