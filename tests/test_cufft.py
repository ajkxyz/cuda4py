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
Tests some of the api in cuda4py.cufft package.
"""
import cuda4py as cu
import cuda4py.cufft as cufft
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
        self.path = os.path.dirname(__file__)
        if not len(self.path):
            self.path = "."

    def tearDown(self):
        if self.old_env is None:
            del os.environ["CUDA_DEVICE"]
        else:
            os.environ["CUDA_DEVICE"] = self.old_env
        del self.old_env
        del self.ctx
        gc.collect()

    def test_constants(self):
        self.assertEqual(cufft.CUFFT_SUCCESS, 0)
        self.assertEqual(cufft.CUFFT_INVALID_PLAN, 1)
        self.assertEqual(cufft.CUFFT_ALLOC_FAILED, 2)
        self.assertEqual(cufft.CUFFT_INVALID_TYPE, 3)
        self.assertEqual(cufft.CUFFT_INVALID_VALUE, 4)
        self.assertEqual(cufft.CUFFT_INTERNAL_ERROR, 5)
        self.assertEqual(cufft.CUFFT_EXEC_FAILED, 6)
        self.assertEqual(cufft.CUFFT_SETUP_FAILED, 7)
        self.assertEqual(cufft.CUFFT_INVALID_SIZE, 8)
        self.assertEqual(cufft.CUFFT_UNALIGNED_DATA, 9)
        self.assertEqual(cufft.CUFFT_INCOMPLETE_PARAMETER_LIST, 10)
        self.assertEqual(cufft.CUFFT_INVALID_DEVICE, 11)
        self.assertEqual(cufft.CUFFT_PARSE_ERROR, 12)
        self.assertEqual(cufft.CUFFT_NO_WORKSPACE, 13)

        self.assertEqual(cufft.CUFFT_R2C, 0x2a)
        self.assertEqual(cufft.CUFFT_C2R, 0x2c)
        self.assertEqual(cufft.CUFFT_C2C, 0x29)
        self.assertEqual(cufft.CUFFT_D2Z, 0x6a)
        self.assertEqual(cufft.CUFFT_Z2D, 0x6c)
        self.assertEqual(cufft.CUFFT_Z2Z, 0x69)

        self.assertEqual(cufft.CUFFT_FORWARD, -1)
        self.assertEqual(cufft.CUFFT_INVERSE, 1)

    def test_errors(self):
        idx = cu.CU.ERRORS[cufft.CUFFT_INVALID_PLAN].find(" | ")
        self.assertGreater(idx, 0)

    def test_version(self):
        fft = cufft.CUFFT(self.ctx)
        ver = fft.version
        logging.debug("cuFFT version is %d", ver)
        self.assertTrue(ver == int(ver))

    def test_auto_allocation(self):
        fft = cufft.CUFFT(self.ctx)
        self.assertTrue(fft.auto_allocation)
        fft.auto_allocation = False
        self.assertFalse(fft.auto_allocation)
        fft.auto_allocation = True
        self.assertTrue(fft.auto_allocation)

    def test_make_plan_many(self):
        fft = cufft.CUFFT(self.ctx)
        fft.auto_allocation = False
        sz = fft.make_plan_many((256, 128), 8, cufft.CUFFT_C2C)
        logging.debug(
            "make_plan_many (default layout) for 256x128 x8 returned %d", sz)
        logging.debug("size is %d", fft.size)
        self.assertEqual(fft.execute, fft.exec_c2c)

        fft = cufft.CUFFT(self.ctx)
        fft.auto_allocation = False
        sz = fft.make_plan_many((256, 128), 8, cufft.CUFFT_C2C,
                                (256, 128), 1, 256 * 128,
                                (256, 128), 1, 256 * 128)
        logging.debug(
            "make_plan_many (tight layout) for 256x128 x8 returned is %d", sz)
        logging.debug("size is %d", fft.size)

    def _test_exec(self, dtype):
        x = numpy.zeros([32, 64], dtype=dtype)
        x[:] = numpy.random.rand(x.size).reshape(x.shape) - 0.5
        y = numpy.ones((x.shape[0], x.shape[1] // 2 + 1),
                       dtype={numpy.float32: numpy.complex64,
                              numpy.float64: numpy.complex128}[dtype])
        x_gold = x.copy()
        try:
            y_gold = numpy.fft.rfft2(x)
        except TypeError:
            y_gold = None  # for pypy
        xbuf = cu.MemAlloc(self.ctx, x)
        ybuf = cu.MemAlloc(self.ctx, y)

        # Forward transform
        fft = cufft.CUFFT(self.ctx)
        fft.auto_allocation = False
        sz = fft.make_plan_many(x.shape, 1,
                                {numpy.float32: cufft.CUFFT_R2C,
                                 numpy.float64: cufft.CUFFT_D2Z}[dtype])
        tmp = cu.MemAlloc(self.ctx, sz)
        fft.workarea = tmp
        self.assertEqual(fft.workarea, tmp)

        self.assertEqual(fft.execute,
                         {numpy.float32: fft.exec_r2c,
                          numpy.float64: fft.exec_d2z}[dtype])
        fft.execute(xbuf, ybuf)
        ybuf.to_host(y)

        if y_gold is not None:
            delta = y - y_gold
            max_diff = numpy.fabs(numpy.sqrt(delta.real * delta.real +
                                             delta.imag * delta.imag)).max()
            logging.debug("Forward max_diff is %.6e", max_diff)
            self.assertLess(max_diff, {numpy.float32: 1.0e-3,
                                       numpy.float64: 1.0e-6}[dtype])

        # Inverse transform
        fft = cufft.CUFFT(self.ctx)
        fft.auto_allocation = False
        sz = fft.make_plan_many(x.shape, 1,
                                {numpy.float32: cufft.CUFFT_C2R,
                                 numpy.float64: cufft.CUFFT_Z2D}[dtype])
        fft.workarea = cu.MemAlloc(self.ctx, sz)

        y /= x.size  # correct scale before inverting
        ybuf.to_device_async(y)
        xbuf.memset32_async(0)  # reset the resulting vector
        self.assertEqual(fft.execute,
                         {numpy.float32: fft.exec_c2r,
                          numpy.float64: fft.exec_z2d}[dtype])
        fft.execute(ybuf, xbuf)
        xbuf.to_host(x)

        max_diff = numpy.fabs(x - x_gold).max()
        logging.debug("Inverse max_diff is %.6e", max_diff)
        self.assertLess(max_diff, {numpy.float32: 1.0e-3,
                                   numpy.float64: 1.0e-6}[dtype])

    def test_exec_float(self):
        logging.debug("ENTER: test_exec_float")
        self._test_exec(numpy.float32)
        logging.debug("EXIT: test_exec_float")

    def test_exec_double(self):
        logging.debug("ENTER: test_exec_double")
        self._test_exec(numpy.float64)
        logging.debug("EXIT: test_exec_double")

    def _test_exec_complex(self, dtype):
        x = numpy.zeros([32, 64], dtype=dtype)
        x.real = numpy.random.rand(x.size).reshape(x.shape) - 0.5
        x.imag = numpy.random.rand(x.size).reshape(x.shape) - 0.5
        y = numpy.ones_like(x)
        x_gold = x.copy()
        try:
            y_gold = numpy.fft.fft2(x)
        except TypeError:
            y_gold = None  # for pypy
        xbuf = cu.MemAlloc(self.ctx, x)
        ybuf = cu.MemAlloc(self.ctx, y)

        # Forward transform
        fft = cufft.CUFFT(self.ctx)
        fft.auto_allocation = False
        sz = fft.make_plan_many(x.shape, 1,
                                {numpy.complex64: cufft.CUFFT_C2C,
                                 numpy.complex128: cufft.CUFFT_Z2Z}[dtype])
        tmp = cu.MemAlloc(self.ctx, sz)
        fft.workarea = tmp
        self.assertEqual(fft.workarea, tmp)

        self.assertEqual(fft.execute, {numpy.complex64: fft.exec_c2c,
                                       numpy.complex128: fft.exec_z2z}[dtype])
        fft.execute(xbuf, ybuf, cufft.CUFFT_FORWARD)
        ybuf.to_host(y)

        if y_gold is not None:
            delta = y - y_gold
            max_diff = numpy.fabs(numpy.sqrt(delta.real * delta.real +
                                             delta.imag * delta.imag)).max()
            logging.debug("Forward max_diff is %.6e", max_diff)
            self.assertLess(max_diff, {numpy.complex64: 1.0e-3,
                                       numpy.complex128: 1.0e-6}[dtype])

        # Inverse transform
        y /= x.size  # correct scale before inverting
        ybuf.to_device_async(y)
        xbuf.memset32_async(0)  # reset the resulting vector
        fft.execute(ybuf, xbuf, cufft.CUFFT_INVERSE)
        xbuf.to_host(x)

        delta = x - x_gold
        max_diff = numpy.fabs(numpy.sqrt(delta.real * delta.real +
                                         delta.imag * delta.imag)).max()
        logging.debug("Inverse max_diff is %.6e", max_diff)
        self.assertLess(max_diff, {numpy.complex64: 1.0e-3,
                                   numpy.complex128: 1.0e-6}[dtype])

    def test_exec_complex_float(self):
        logging.debug("ENTER: test_exec_complex_float")
        self._test_exec_complex(numpy.complex64)
        logging.debug("EXIT: test_exec_complex_float")

    def test_exec_complex_double(self):
        logging.debug("ENTER: test_exec_complex_double")
        self._test_exec_complex(numpy.complex128)
        logging.debug("EXIT: test_exec_complex_double")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
