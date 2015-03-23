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
Tests correctness of destructors.
"""
import gc
import logging
import cuda4py as cu
import cuda4py.blas as blas
from io import StringIO
import os
import sys
import unittest


class Container(object):
    pass


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

    def test_del(self):
        logging.debug("ENTER: test_del")
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            self._test_del()
        finally:
            sys.stderr = old_stderr
        logging.debug("EXIT: test_del")

    def _test_del(self):
        self._test_bad()
        gc.collect()
        logging.debug("Remaining context count: %d", cu.Context.context_count)
        gc.collect()  # this one is for PyPy
        logging.debug("Remaining context count: %d", cu.Context.context_count)
        s = sys.stderr.getvalue()
        self.assertEqual(len(s), 0, s)
        # __del__ may not be called at all, so the following line is commented
        # self.assertEqual(cu.Context.context_count, 0)
        # see previous debug output to ensure that destructor was called

        self._test_good()
        gc.collect()
        logging.debug("Remaining context count: %d", cu.Context.context_count)
        gc.collect()  # this one is for PyPy
        logging.debug("Remaining context count: %d", cu.Context.context_count)
        s = sys.stderr.getvalue()
        self.assertEqual(len(s), 0, s)
        # __del__ may not be called at all, so the following line is commented
        # self.assertEqual(cu.Context.context_count, 0)
        # see previous debug output to ensure that destructor was called

    def _test_bad(self):
        ctx = cu.Devices().create_some_context()
        a = Container()
        a.ctx = ctx
        b = Container()
        b.mem = cu.MemAlloc(ctx, 4096)
        b.module = cu.Module(ctx, source="""
            __global__ void test(float *a) {
                a[blockIdx.x * blockDim.x + threadIdx.x] *= 1.1f;
            }""")
        b.blas = blas.CUBLAS(ctx)

        # Create external circular reference
        a.b = b
        b.a = a

        logging.debug("Remaining context count: %d", cu.Context.context_count)
        # self.assertEqual(cu.Context.context_count, 1)
        self.assertIsNotNone(ctx)  # to hold ctx up to this point

    def _test_good(self):
        ctx = cu.Devices().create_some_context()
        a = Container()
        a.ctx = ctx
        b = Container()
        b.mem = cu.MemAlloc(ctx, 4096)
        b.module = cu.Module(ctx, source="""
            __global__ void test(float *a) {
                a[blockIdx.x * blockDim.x + threadIdx.x] *= 1.1f;
            }""")
        b.blas = blas.CUBLAS(ctx)

        logging.debug("Remaining context count: %d", cu.Context.context_count)
        # self.assertEqual(cu.Context.context_count, 1)
        self.assertIsNotNone(ctx)  # to hold ctx up to this point


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
