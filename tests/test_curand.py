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
Tests some of the api in cuda4py.curand package.
"""
import cuda4py as cu
import cuda4py.curand as curand
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
        self.assertEqual(curand.CURAND_STATUS_SUCCESS, 0)
        self.assertEqual(curand.CURAND_STATUS_VERSION_MISMATCH, 100)
        self.assertEqual(curand.CURAND_STATUS_NOT_INITIALIZED, 101)
        self.assertEqual(curand.CURAND_STATUS_ALLOCATION_FAILED, 102)
        self.assertEqual(curand.CURAND_STATUS_TYPE_ERROR, 103)
        self.assertEqual(curand.CURAND_STATUS_OUT_OF_RANGE, 104)
        self.assertEqual(curand.CURAND_STATUS_LENGTH_NOT_MULTIPLE, 105)
        self.assertEqual(curand.CURAND_STATUS_DOUBLE_PRECISION_REQUIRED, 106)
        self.assertEqual(curand.CURAND_STATUS_LAUNCH_FAILURE, 201)
        self.assertEqual(curand.CURAND_STATUS_PREEXISTING_FAILURE, 202)
        self.assertEqual(curand.CURAND_STATUS_INITIALIZATION_FAILED, 203)
        self.assertEqual(curand.CURAND_STATUS_ARCH_MISMATCH, 204)
        self.assertEqual(curand.CURAND_STATUS_INTERNAL_ERROR, 999)

        self.assertEqual(curand.CURAND_RNG_TEST, 0)
        self.assertEqual(curand.CURAND_RNG_PSEUDO_DEFAULT, 100)
        self.assertEqual(curand.CURAND_RNG_PSEUDO_XORWOW, 101)
        self.assertEqual(curand.CURAND_RNG_PSEUDO_MRG32K3A, 121)
        self.assertEqual(curand.CURAND_RNG_PSEUDO_MTGP32, 141)
        self.assertEqual(curand.CURAND_RNG_PSEUDO_MT19937, 142)
        self.assertEqual(curand.CURAND_RNG_PSEUDO_PHILOX4_32_10, 161)
        self.assertEqual(curand.CURAND_RNG_QUASI_DEFAULT, 200)
        self.assertEqual(curand.CURAND_RNG_QUASI_SOBOL32, 201)
        self.assertEqual(curand.CURAND_RNG_QUASI_SCRAMBLED_SOBOL32, 202)
        self.assertEqual(curand.CURAND_RNG_QUASI_SOBOL64, 203)
        self.assertEqual(curand.CURAND_RNG_QUASI_SCRAMBLED_SOBOL64, 204)

    def test_errors(self):
        idx = cu.CU.ERRORS[curand.CURAND_STATUS_NOT_INITIALIZED].find(" | ")
        self.assertGreater(idx, 0)

    def test_create(self):
        rng = curand.CURAND(self.ctx)
        del rng

    def test_version(self):
        rng = curand.CURAND(self.ctx)
        ver = rng.version
        logging.debug("cuRAND version is %d", ver)
        self.assertTrue(ver == int(ver))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
