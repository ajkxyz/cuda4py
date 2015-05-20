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
CUBLAS cffi bindings and helper classes.
"""
import cffi
import cuda4py._cffi as cuffi
from cuda4py._py import CU


#: ffi parser
ffi = None


#: Loaded shared library
lib = None


#: Error codes
CUDNN_STATUS_SUCCESS = 0
CUDNN_STATUS_NOT_INITIALIZED = 1
CUDNN_STATUS_ALLOC_FAILED = 2
CUDNN_STATUS_BAD_PARAM = 3
CUDNN_STATUS_INTERNAL_ERROR = 4
CUDNN_STATUS_INVALID_VALUE = 5
CUDNN_STATUS_ARCH_MISMATCH = 6
CUDNN_STATUS_MAPPING_ERROR = 7
CUDNN_STATUS_EXECUTION_FAILED = 8
CUDNN_STATUS_NOT_SUPPORTED = 9
CUDNN_STATUS_LICENSE_ERROR = 10


#: Error descriptions
ERRORS = {
    CUDNN_STATUS_NOT_INITIALIZED: "CUDNN_STATUS_NOT_INITIALIZED",
    CUDNN_STATUS_ALLOC_FAILED: "CUDNN_STATUS_ALLOC_FAILED",
    CUDNN_STATUS_BAD_PARAM: "CUDNN_STATUS_BAD_PARAM",
    CUDNN_STATUS_INTERNAL_ERROR: "CUDNN_STATUS_INTERNAL_ERROR",
    CUDNN_STATUS_INVALID_VALUE: "CUDNN_STATUS_INVALID_VALUE",
    CUDNN_STATUS_ARCH_MISMATCH: "CUDNN_STATUS_ARCH_MISMATCH",
    CUDNN_STATUS_MAPPING_ERROR: "CUDNN_STATUS_MAPPING_ERROR",
    CUDNN_STATUS_EXECUTION_FAILED: "CUDNN_STATUS_EXECUTION_FAILED",
    CUDNN_STATUS_NOT_SUPPORTED: "CUDNN_STATUS_NOT_SUPPORTED",
    CUDNN_STATUS_LICENSE_ERROR: "CUDNN_STATUS_LICENSE_ERROR"
}


#: cublasOperation_t


def _initialize(backends):
    global lib
    if lib is not None:
        return
    # C function definitions
    # size_t instead of void* is used
    # for convinience with python calls and numpy arrays.
    src = """
    typedef int cudnnStatus_t;
    typedef void *cudnnHandle_t;

    cudnnStatus_t cudnnCreate(cudnnHandle_t *handle);
    cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);
    """

    # Parse
    global ffi
    ffi = cffi.FFI()
    ffi.cdef(src)

    # Load library
    for libnme in backends:
        try:
            lib = ffi.dlopen(libnme)
            break
        except OSError:
            pass
    else:
        ffi = None
        raise OSError("Could not load cudnn library")

    global ERRORS
    for code, msg in ERRORS.items():
        if code in CU.ERRORS:
            s = " | " + msg
            if s not in CU.ERRORS[code]:
                CU.ERRORS[code] += s
        else:
            CU.ERRORS[code] = msg


def initialize(backends=("libcudnn.so", "cudnn64_65.dll")):
    """Loads shared library.
    """
    cuffi.initialize()
    global lib
    if lib is not None:
        return
    with cuffi.lock:
        _initialize(backends)


class CUDNN(object):
    """CUDNN functions can be invoked from this class.
    """
    def __init__(self, context):
        self._context = context
        self._lib = None
        context._add_ref(self)
        initialize()
        handle = ffi.new("cudnnHandle_t *")
        with context:
            err = lib.cudnnCreate(handle)
        if err:
            self._handle = None
            raise CU.error("cudnnCreate", err)
        self._lib = lib  # to hold the reference
        self._handle = handle[0]

    @property
    def context(self):
        return self._context

    def _release(self):
        if self._lib is not None and self._handle is not None:
            self._lib.cudnnDestroy(self._handle)
            self._handle = None

    def __del__(self):
        if self.context.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self._release()
        self.context._del_ref(self)
