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


#: FFT direction
CUFFT_FORWARD = -1
CUFFT_INVERSE = 1


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

    cufftResult cufftExecC2C(cufftHandle plan, size_t idata,
                             size_t odata, int direction);
    cufftResult cufftExecZ2Z(cufftHandle plan, size_t idata,
                             size_t odata, int direction);

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

    Attributes:
        execute: handle to one of exec_r2c, etc. depending on fft_type
                 parameter of make_plan_many().
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
        self._workarea = None
        self.execute = self._exec_unknown

    def _exec_unknown(self, idata, odata):
        raise ValueError(
            "make_plan_many() has not yet been called with known plan")

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
        err = self._lib.cufftGetVersion(version)
        if err:
            raise CU.error("cufftGetVersion", err)
        return int(version[0])

    @property
    def auto_allocation(self):
        return self._auto_allocation

    @auto_allocation.setter
    def auto_allocation(self, value):
        alloc = bool(value)
        err = self._lib.cufftSetAutoAllocation(self.handle, alloc)
        if err:
            raise CU.error("cufftSetAutoAllocation", err)
        self._auto_allocation = alloc

    def make_plan_many(self, xyz, batch, fft_type,
                       inembed=None, istride=1, idist=0,
                       onembed=None, ostride=1, odist=0):
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

        Will assign self.execute based on fft_type.

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
        sz = ffi.new("size_t[]", 4)
        err = self._lib.cufftMakePlanMany(self.handle, rank, n,
                                          _inembed, istride, idist,
                                          _onembed, ostride, odist,
                                          fft_type, batch, sz)
        if err:
            raise CU.error("cufftMakePlanMany", err)
        self.execute = {
            CUFFT_R2C: self.exec_r2c,
            CUFFT_C2R: self.exec_c2r,
            CUFFT_C2C: self.exec_c2c,
            CUFFT_D2Z: self.exec_d2z,
            CUFFT_Z2D: self.exec_z2d,
            CUFFT_Z2Z: self.exec_z2z
        }.get(fft_type, self._exec_unknown)
        return int(sz[0])

    @property
    def size(self):
        """Returns actual size of the work area required to support the plan.
        """
        sz = ffi.new("size_t[]", 4)
        err = self._lib.cufftGetSize(self.handle, sz)
        if err:
            raise CU.error("cufftGetSize", err)
        return int(sz[0])

    @property
    def workarea(self):
        """Returns previously set workarea.
        """
        return self._workarea

    @workarea.setter
    def workarea(self, value):
        """Sets workarea for plan execution.
        """
        err = self._lib.cufftSetWorkArea(self.handle, value)
        if err:
            raise CU.error("cufftSetWorkArea", err)
        self._workarea = value

    def exec_r2c(self, idata, odata):
        """Executes a single-precision real-to-complex,
        implicitly forward, cuFFT transform plan.
        """
        err = self._lib.cufftExecR2C(self.handle, idata, odata)
        if err:
            raise CU.error("cufftExecR2C", err)

    def exec_d2z(self, idata, odata):
        """Executes a double-precision real-to-complex,
        implicitly forward, cuFFT transform plan.
        """
        err = self._lib.cufftExecD2Z(self.handle, idata, odata)
        if err:
            raise CU.error("cufftExecD2Z", err)

    def exec_c2r(self, idata, odata):
        """Executes a single-precision complex-to-real,
        implicitly inverse, cuFFT transform plan.
        """
        err = self._lib.cufftExecC2R(self.handle, idata, odata)
        if err:
            raise CU.error("cufftExecC2R", err)

    def exec_z2d(self, idata, odata):
        """Executes a double-precision complex-to-real,
        implicitly inverse, cuFFT transform plan.
        """
        err = self._lib.cufftExecZ2D(self.handle, idata, odata)
        if err:
            raise CU.error("cufftExecZ2D", err)

    def exec_c2c(self, idata, odata, direction):
        """Executes a single-precision complex-to-complex
        cuFFT transform plan.
        """
        err = self._lib.cufftExecC2C(self.handle, idata, odata, direction)
        if err:
            raise CU.error("cufftExecC2C", err)

    def exec_z2z(self, idata, odata, direction):
        """Executes a double-precision complex-to-complex
        cuFFT transform plan.
        """
        err = self._lib.cufftExecZ2Z(self.handle, idata, odata, direction)
        if err:
            raise CU.error("cufftExecZ2Z", err)

    def _release(self):
        if self._lib is not None and self.handle is not None:
            self._lib.cufftDestroy(self.handle)
            self._handle = None

    def __del__(self):
        if self.context.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self._release()
        self.context._del_ref(self)
