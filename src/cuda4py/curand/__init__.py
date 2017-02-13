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
Init module for cuRAND cffi bindings and helper classes.
"""

from cuda4py.curand._curand import (CURAND,
                                    initialize,

                                    CURAND_RNG_TEST,
                                    CURAND_RNG_PSEUDO_DEFAULT,
                                    CURAND_RNG_PSEUDO_XORWOW,
                                    CURAND_RNG_PSEUDO_MRG32K3A,
                                    CURAND_RNG_PSEUDO_MTGP32,
                                    CURAND_RNG_PSEUDO_MT19937,
                                    CURAND_RNG_PSEUDO_PHILOX4_32_10,
                                    CURAND_RNG_QUASI_DEFAULT,
                                    CURAND_RNG_QUASI_SOBOL32,
                                    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32,
                                    CURAND_RNG_QUASI_SOBOL64,
                                    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64,

                                    CURAND_ORDERING_PSEUDO_BEST,
                                    CURAND_ORDERING_PSEUDO_DEFAULT,
                                    CURAND_ORDERING_PSEUDO_SEEDED,
                                    CURAND_ORDERING_QUASI_DEFAULT,

                                    CURAND_STATUS_SUCCESS,
                                    CURAND_STATUS_VERSION_MISMATCH,
                                    CURAND_STATUS_NOT_INITIALIZED,
                                    CURAND_STATUS_ALLOCATION_FAILED,
                                    CURAND_STATUS_TYPE_ERROR,
                                    CURAND_STATUS_OUT_OF_RANGE,
                                    CURAND_STATUS_LENGTH_NOT_MULTIPLE,
                                    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED,
                                    CURAND_STATUS_LAUNCH_FAILURE,
                                    CURAND_STATUS_PREEXISTING_FAILURE,
                                    CURAND_STATUS_INITIALIZATION_FAILED,
                                    CURAND_STATUS_ARCH_MISMATCH,
                                    CURAND_STATUS_INTERNAL_ERROR)
