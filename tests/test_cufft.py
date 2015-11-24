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
        self.fft = cufft.CUFFT(self.ctx)
        self.path = os.path.dirname(__file__)
        if not len(self.path):
            self.path = "."

    def tearDown(self):
        if self.old_env is None:
            del os.environ["CUDA_DEVICE"]
        else:
            os.environ["CUDA_DEVICE"] = self.old_env
        del self.old_env
        del self.fft
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

    def test_errors(self):
        idx = cu.CU.ERRORS[cufft.CUFFT_INVALID_PLAN].find(" | ")
        self.assertGreater(idx, 0)

    def test_version(self):
        ver = self.fft.version
        logging.debug("cuFFT version is %d", ver)
        self.assertTrue(ver == int(ver))

    def test_auto_allocation(self):
        self.assertTrue(self.fft.auto_allocation)
        self.fft.auto_allocation = False
        self.assertFalse(self.fft.auto_allocation)
        self.fft.auto_allocation = True
        self.assertTrue(self.fft.auto_allocation)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
