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
Init module for BLAS cffi bindings and helper classes.
"""

from cuda4py.blas._cublas import (CUBLAS,
                                  initialize,

                                  CUBLAS_OP_N,
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_C,

                                  CUBLAS_DATA_FLOAT,
                                  CUBLAS_DATA_DOUBLE,
                                  CUBLAS_DATA_HALF,
                                  CUBLAS_DATA_INT8,

                                  CUBLAS_POINTER_MODE_HOST,
                                  CUBLAS_POINTER_MODE_DEVICE,

                                  CUBLAS_STATUS_SUCCESS,
                                  CUBLAS_STATUS_NOT_INITIALIZED,
                                  CUBLAS_STATUS_ALLOC_FAILED,
                                  CUBLAS_STATUS_INVALID_VALUE,
                                  CUBLAS_STATUS_ARCH_MISMATCH,
                                  CUBLAS_STATUS_MAPPING_ERROR,
                                  CUBLAS_STATUS_EXECUTION_FAILED,
                                  CUBLAS_STATUS_INTERNAL_ERROR,
                                  CUBLAS_STATUS_NOT_SUPPORTED,
                                  CUBLAS_STATUS_LICENSE_ERROR)
