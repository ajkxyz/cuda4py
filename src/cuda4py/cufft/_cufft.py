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
CUFFT_SUCCESS = 0
CUFFT_INVALID_PLAN = 1
CUFFT_ALLOC_FAILED = 2
CUFFT_INVALID_TYPE = 3
CUFFT_INVALID_VALUE = 4
CUFFT_INTERNAL_ERROR = 5
CUFFT_EXEC_FAILED = 6
CUFFT_SETUP_FAILED = 7
CUFFT_INVALID_SIZE = 8
CUFFT_UNALIGNED_DATA = 9
CUFFT_INCOMPLETE_PARAMETER_LIST = 10
CUFFT_INVALID_DEVICE = 11
CUFFT_PARSE_ERROR = 12
CUFFT_NO_WORKSPACE = 13


#: Error descriptions
ERRORS = {
    CUFFT_INVALID_PLAN: "CUFFT_INVALID_PLAN",
    CUFFT_ALLOC_FAILED: "CUFFT_ALLOC_FAILED",
    CUFFT_INVALID_TYPE: "CUFFT_INVALID_TYPE",
    CUFFT_INVALID_VALUE: "CUFFT_INVALID_VALUE",
    CUFFT_INTERNAL_ERROR: "CUFFT_INTERNAL_ERROR",
    CUFFT_EXEC_FAILED: "CUFFT_EXEC_FAILED",
    CUFFT_SETUP_FAILED: "CUFFT_SETUP_FAILED",
    CUFFT_INVALID_SIZE: "CUFFT_INVALID_SIZE",
    CUFFT_UNALIGNED_DATA: "CUFFT_UNALIGNED_DATA",
    CUFFT_INCOMPLETE_PARAMETER_LIST: "CUFFT_INCOMPLETE_PARAMETER_LIST",
    CUFFT_INVALID_DEVICE: "CUFFT_INVALID_DEVICE",
    CUFFT_PARSE_ERROR: "CUFFT_PARSE_ERROR",
    CUFFT_NO_WORKSPACE: "CUFFT_NO_WORKSPACE"
}


#: cufftType
CUFFT_R2C = 0x2a  # Real to Complex (interleaved)
CUFFT_C2R = 0x2c  # Complex (interleaved) to Real
CUFFT_C2C = 0x29  # Complex to Complex, interleaved
CUFFT_D2Z = 0x6a  # Double to Double-Complex
CUFFT_Z2D = 0x6c  # Double-Complex to Double
CUFFT_Z2Z = 0x69  # Double-Complex to Double-Complex


def _initialize(backends):
    global lib
    if lib is not None:
        return
    # C function definitions
    # size_t instead of void* is used
    # for convinience with python calls and numpy arrays,
    # cffi automatically calls int() on objects also.
    src = """
    typedef int cufftResult;
    typedef int cufftHandle;
    typedef int cufftType;

    cufftResult cufftCreate(cufftHandle *plan);
    cufftResult cufftDestroy(cufftHandle plan);

    cufftResult cufftSetAutoAllocation(cufftHandle plan, int autoAllocate);
    cufftResult cufftMakePlanMany(cufftHandle plan,
                                  int rank, int *n,
                                  int *inembed, int istride, int idist,
                                  int *onembed, int ostride, int odist,
                                  cufftType type, int batch, size_t *workSize);
    cufftResult cufftGetSize(cufftHandle handle, size_t *workSize);
    cufftResult cufftSetWorkArea(cufftHandle plan, size_t workArea);

    cufftResult cufftExecR2C(cufftHandle plan,
                             size_t idata,
                             size_t odata);
    cufftResult cufftExecD2Z(cufftHandle plan,
                             size_t idata,
                             size_t odata);

    cufftResult cufftExecC2R(cufftHandle plan,
                             size_t idata,
                             size_t odata);
    cufftResult cufftExecZ2D(cufftHandle plan,
                             size_t idata,
                             size_t odata);

    cufftResult cufftSetStream(cufftHandle plan, size_t stream);
    cufftResult cufftGetVersion(int *version);
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
        raise OSError("Could not load cufft library")

    global ERRORS
    for code, msg in ERRORS.items():
        if code in CU.ERRORS:
            s = " | " + msg
            if s not in CU.ERRORS[code]:
                CU.ERRORS[code] += s
        else:
            CU.ERRORS[code] = msg


def initialize(backends=("libcufft.so", "cufft64_65.dll")):
    """Loads shared library.
    """
    cuffi.initialize()
    global lib
    if lib is not None:
        return
    with cuffi.lock:
        _initialize(backends)


class CUFFT(object):
    """cuFFT plan.
    """
    def __init__(self, context):
        self._context = context
        self._lib = None
        context._add_ref(self)
        initialize()
        handle = ffi.new("cufftHandle *")
        with context:
            err = lib.cufftCreate(handle)
        if err:
            self._handle = None
            raise CU.error("cufftCreate", err)
        self._lib = lib  # to hold the reference
        self._handle = int(handle[0])
        self._auto_allocation = True

    def __int__(self):
        return self.handle

    @property
    def handle(self):
        return self._handle

    @property
    def context(self):
        return self._context

    @property
    def version(self):
        """Returns cuFFT version.
        """
        version = ffi.new("int *")
        err = lib.cufftGetVersion(version)
        if err:
            raise CU.error("cufftGetVersion", err)
        return int(version[0])

    @property
    def auto_allocation(self):
        return self._auto_allocation

    @auto_allocation.setter
    def auto_allocation(self, value):
        alloc = bool(value)
        err = lib.cufftSetAutoAllocation(self.handle, alloc)
        if err:
            raise CU.error("cufftSetAutoAllocation", err)
        self._auto_allocation = alloc

    def make_plan_many(self, xyz, batch, fft_type,
                       inembed=None, istride=0, idist=0,
                       onembed=None, ostride=0, odist=0):
        """Makes 1, 2 or 3 dimensional FFT plan.

        Parameters:
            xyz: tuple of dimensions.
            batch: number of FFTs to make.
            fft_type: type of FFT (CUFFT_R2C, CUFFT_C2R etc.).
            inembed: tuple with storage dimensions of the input data in memory
                     (can be None).
            istride: distance between two successive input elements
                     in the least significant (i.e., innermost) dimension.
            idist: distance between the first element of two consecutive
                   signals in a batch of the input data.
            onembed: tuple with storage dimensions of the output data in memory
                     (can be None).
            ostride: distance between two successive output elements
                     in the least significant (i.e., innermost) dimension.
            odist: distance between the first element of two consecutive
                   signals in a batch of the output data.

        Returns:
            Required work size.
        """
        rank = len(xyz)
        n = ffi.new("int[]", rank)
        n[0:rank] = xyz
        if inembed is None:
            _inembed = ffi.NULL
        else:
            _inembed = ffi.new("int[]", rank)
            _inembed[0:rank] = inembed
        if onembed is None:
            _onembed = ffi.NULL
        else:
            _onembed = ffi.new("int[]", rank)
            _onembed[0:rank] = onembed
        sz = ffi.new("size_t *")
        err = lib.cufftMakePlanMany(self.handle, rank, n,
                                    _inembed, istride, idist,
                                    _onembed, ostride, odist,
                                    fft_type, batch, sz)
        if err:
            raise CU.error("cufftMakePlanMany", err)
        return int(sz[0])

    def _release(self):
        if self._lib is not None and self.handle is not None:
            self._lib.cufftDestroy(self.handle)
            self._handle = None

    def __del__(self):
        if self.context.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self._release()
        self.context._del_ref(self)
