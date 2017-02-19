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
cuRAND cffi bindings and helper classes.
"""
import cffi
import cuda4py._cffi as cuffi
from cuda4py._py import CU


#: ffi parser
ffi = None


#: Loaded shared library
lib = None


#: Error codes
CURAND_STATUS_SUCCESS = 0
CURAND_STATUS_VERSION_MISMATCH = 100
CURAND_STATUS_NOT_INITIALIZED = 101
CURAND_STATUS_ALLOCATION_FAILED = 102
CURAND_STATUS_TYPE_ERROR = 103
CURAND_STATUS_OUT_OF_RANGE = 104
CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105
CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106
CURAND_STATUS_LAUNCH_FAILURE = 201
CURAND_STATUS_PREEXISTING_FAILURE = 202
CURAND_STATUS_INITIALIZATION_FAILED = 203
CURAND_STATUS_ARCH_MISMATCH = 204
CURAND_STATUS_INTERNAL_ERROR = 999


#: Error descriptions
ERRORS = {
    CURAND_STATUS_VERSION_MISMATCH: "CURAND_STATUS_VERSION_MISMATCH",
    CURAND_STATUS_NOT_INITIALIZED: "CURAND_STATUS_NOT_INITIALIZED",
    CURAND_STATUS_ALLOCATION_FAILED: "CURAND_STATUS_ALLOCATION_FAILED",
    CURAND_STATUS_TYPE_ERROR: "CURAND_STATUS_TYPE_ERROR",
    CURAND_STATUS_OUT_OF_RANGE: "CURAND_STATUS_OUT_OF_RANGE",
    CURAND_STATUS_LENGTH_NOT_MULTIPLE: "CURAND_STATUS_LENGTH_NOT_MULTIPLE",
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED",
    CURAND_STATUS_LAUNCH_FAILURE: "CURAND_STATUS_LAUNCH_FAILURE",
    CURAND_STATUS_PREEXISTING_FAILURE: "CURAND_STATUS_PREEXISTING_FAILURE",
    CURAND_STATUS_INITIALIZATION_FAILED: "CURAND_STATUS_INITIALIZATION_FAILED",
    CURAND_STATUS_ARCH_MISMATCH: "CURAND_STATUS_ARCH_MISMATCH",
    CURAND_STATUS_INTERNAL_ERROR: "CURAND_STATUS_INTERNAL_ERROR"
}


#: curandRngType
CURAND_RNG_TEST = 0
CURAND_RNG_PSEUDO_DEFAULT = 100  # Default pseudorandom
CURAND_RNG_PSEUDO_XORWOW = 101  # XORWOW pseudorandom
CURAND_RNG_PSEUDO_MRG32K3A = 121  # MRG32k3a pseudorandom
CURAND_RNG_PSEUDO_MTGP32 = 141  # Mersenne Twister MTGP32 pseudorandom
CURAND_RNG_PSEUDO_MT19937 = 142  # Mersenne Twister MT19937 pseudorandom
CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161  # PHILOX-4x32-10 pseudorandom
CURAND_RNG_QUASI_DEFAULT = 200  # Default quasirandom
CURAND_RNG_QUASI_SOBOL32 = 201  # Sobol32 quasirandom
CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202  # Scrambled Sobol32 quasirandom
CURAND_RNG_QUASI_SOBOL64 = 203  # Sobol64 quasirandom
CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204  # Scrambled Sobol64 quasirandom


#: curandOrdering
CURAND_ORDERING_PSEUDO_BEST = 100  # Best ordering for pseudorandom results
CURAND_ORDERING_PSEUDO_DEFAULT = 101  # Specific default 4096 thread sequence
CURAND_ORDERING_PSEUDO_SEEDED = 102  # Fast lower quality pseudorandom results
CURAND_ORDERING_QUASI_DEFAULT = 201  # N-dimensional ordering for quasirandom


def _initialize(backends):
    global lib
    if lib is not None:
        return
    # C function definitions
    # intptr_t instead of void* is used
    # for convinience with python calls and numpy arrays,
    # cffi automatically calls int() on objects also.
    src = """
    typedef int curandStatus_t;
    typedef intptr_t curandGenerator_t;
    typedef int curandRngType_t;
    typedef int curandOrdering_t;

    curandStatus_t curandCreateGenerator(
        curandGenerator_t *generator, curandRngType_t rng_type);
    curandStatus_t curandDestroyGenerator(curandGenerator_t generator);

    curandStatus_t curandGetVersion(int *version);

    curandStatus_t curandSetPseudoRandomGeneratorSeed(
        curandGenerator_t generator, unsigned long long seed);
    curandStatus_t curandSetGeneratorOffset(
        curandGenerator_t generator, unsigned long long offset);
    curandStatus_t curandSetGeneratorOrdering(
        curandGenerator_t generator, curandOrdering_t order);
    curandStatus_t curandSetQuasiRandomGeneratorDimensions(
        curandGenerator_t generator, unsigned int num_dimensions);

    curandStatus_t curandGenerate(
        curandGenerator_t generator, intptr_t outputPtr, size_t num);
    curandStatus_t curandGenerateLongLong(
        curandGenerator_t generator, intptr_t outputPtr, size_t num);

    curandStatus_t curandGenerateUniform(
        curandGenerator_t generator, intptr_t outputPtr, size_t num);
    curandStatus_t curandGenerateUniformDouble(
        curandGenerator_t generator, intptr_t outputPtr, size_t num);

    curandStatus_t curandGenerateNormal(
        curandGenerator_t generator, intptr_t outputPtr, size_t n,
        float mean, float stddev);
    curandStatus_t curandGenerateNormalDouble(
        curandGenerator_t generator, intptr_t outputPtr, size_t n,
        double mean, double stddev);

    curandStatus_t curandGenerateLogNormal(
        curandGenerator_t generator, intptr_t outputPtr, size_t n,
        float mean, float stddev);
    curandStatus_t curandGenerateLogNormalDouble(
        curandGenerator_t generator, intptr_t outputPtr, size_t n,
        double mean, double stddev);

    curandStatus_t curandGeneratePoisson(
        curandGenerator_t generator, intptr_t outputPtr,
        size_t n, double lambda);

    curandStatus_t curandCreateGeneratorHost(
        curandGenerator_t *generator, curandRngType_t rng_type);
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
        raise OSError("Could not load curand library")

    global ERRORS
    for code, msg in ERRORS.items():
        if code in CU.ERRORS:
            s = " | " + msg
            if s not in CU.ERRORS[code]:
                CU.ERRORS[code] += s
        else:
            CU.ERRORS[code] = msg


def initialize(backends=("libcurand.so", "curand64_65.dll")):
    """Loads shared library.
    """
    cuffi.initialize()
    global lib
    if lib is not None:
        return
    with cuffi.lock:
        _initialize(backends)


class CURAND(object):
    """cuRAND base object.

    Attributes:
        _context: CUDA context or None in case of host generator.
        _lib: handle to shared library.
        _handle: cuRAND Generator's instance.
        _rng_type: type of generator passed to the constructor.
        _seed: last successfully set seed (default 0).
        _offset: last successfully set offset (default 0).
        _ordering: last successfully set ordering (default 0).
        _dimensions: last seccessfully set dimensions (default 0),
                     meaningfull only for quasirandom generators.
    """
    def __init__(self, context, rng_type=CURAND_RNG_PSEUDO_DEFAULT):
        """Constructor.

        Parameters:
            context: CUDA context handle or None to use the host generator.
            rng_type: type of the random generator.
        """
        self._context = context
        self._lib = None
        if context is not None:
            context._add_ref(self)
        initialize()
        handle = ffi.new("curandGenerator_t *")
        if context is not None:
            with context:
                err = lib.curandCreateGenerator(handle, int(rng_type))
        else:
            err = lib.curandCreateGeneratorHost(handle, int(rng_type))
        if err:
            self._handle = None
            raise CU.error("curandCreateGenerator" if context is not None
                           else "curandCreateGeneratorHost", err)
        self._lib = lib  # to hold the reference
        self._handle = int(handle[0])
        self._rng_type = int(rng_type)
        self._seed = 0
        self._offset = 0
        self._ordering = 0
        self._dimensions = 0

    def __int__(self):
        return self.handle

    @property
    def handle(self):
        return self._handle

    @property
    def context(self):
        return self._context

    @property
    def rng_type(self):
        """Returns type of generator passed to the constructor.
        """
        return self._rng_type

    @property
    def version(self):
        """Returns cuRAND version.
        """
        version = ffi.new("int *")
        err = self._lib.curandGetVersion(version)
        if err:
            raise CU.error("curandGetVersion", err)
        return int(version[0])

    @property
    def seed(self):
        """Returns last successfully set seed.
        """
        return self._seed

    @seed.setter
    def seed(self, value):
        """Sets generator seed as an 64-bit integer.
        """
        err = self._lib.curandSetPseudoRandomGeneratorSeed(
            self.handle, int(value))
        if err:
            raise CU.error("curandSetPseudoRandomGeneratorSeed", err)
        self._seed = int(value)

    @property
    def offset(self):
        """Returns last successfully set offset.
        """
        return self._offset

    @offset.setter
    def offset(self, value):
        """Sets generator offset as an 64-bit integer.
        """
        err = self._lib.curandSetGeneratorOffset(
            self.handle, int(value))
        if err:
            raise CU.error("curandSetGeneratorOffset", err)
        self._offset = int(value)

    @property
    def ordering(self):
        """Returns last successfully set ordering.
        """
        return self._ordering

    @ordering.setter
    def ordering(self, value):
        """Sets generator ordering.
        """
        err = self._lib.curandSetGeneratorOrdering(
            self.handle, int(value))
        if err:
            raise CU.error("curandSetGeneratorOrdering", err)
        self._ordering = int(value)

    @property
    def dimensions(self):
        """Returns last successfully set dimensions.
        """
        return self._dimensions

    @dimensions.setter
    def dimensions(self, value):
        """Sets quasirandom generator dimensions.
        """
        err = self._lib.curandSetQuasiRandomGeneratorDimensions(
            self.handle, int(value))
        if err:
            raise CU.error("curandSetQuasiRandomGeneratorDimensions", err)
        self._dimensions = int(value)

    def _extract_ptr_and_count(self, arr, count, itemsize):
        """Returns tuple of address of an arr and extracted item count
        casted to int.

        It will clamp requested count to an array size if possible.
        """
        if self.context is None:
            arr, size = CU.extract_ptr_and_size(arr, None)
        elif count is None:
            size = arr.size
        else:
            size = getattr(arr, "size", count * itemsize)
        size = size if count is None else min(count * itemsize, size)
        return int(arr), int(size) // itemsize

    def generate32(self, dst, count=None):
        """Generates specified number of 32-bit random values.

        Not valid for 64-bit generators.

        Parameters:
            dst: buffer to store the results or
                 numpy array in case of host generator.
            count: number of 32-bit values to put to dst or
                   None to fill full dst when the it's size is available.
        """
        dst, count = self._extract_ptr_and_count(dst, count, 4)
        err = self._lib.curandGenerate(self.handle, dst, count)
        if err:
            raise CU.error("curandGenerate", err)

    def generate64(self, dst, count=None):
        """Generates specified number of 64-bit random values.

        Valid only for 64-bit generators.

        Parameters:
            dst: buffer to store the results or
                 numpy array in case of host generator.
            count: number of 64-bit values to put to dst or
                   None to fill full dst when the it's size is available.
        """
        dst, count = self._extract_ptr_and_count(dst, count, 8)
        err = self._lib.curandGenerateLongLong(self.handle, dst, count)
        if err:
            raise CU.error("curandGenerateLongLong", err)

    def generate_uniform(self, dst, count=None):
        """Generates specified number of 32-bit uniformly distributed floats.

        Will generate values in range (0, 1].

        Parameters:
            dst: buffer to store the results or
                 numpy array in case of host generator.
            count: number of 32-bit floats to put to dst or
                   None to fill full dst when the it's size is available.
        """
        dst, count = self._extract_ptr_and_count(dst, count, 4)
        err = self._lib.curandGenerateUniform(self.handle, dst, count)
        if err:
            raise CU.error("curandGenerateUniform", err)

    def generate_uniform_double(self, dst, count=None):
        """Generates specified number of 64-bit uniformly distributed floats.

        Will generate values in range (0, 1].

        Parameters:
            dst: buffer to store the results or
                 numpy array in case of host generator.
            count: number of 64-bit floats to put to dst or
                   None to fill full dst when the it's size is available.
        """
        dst, count = self._extract_ptr_and_count(dst, count, 8)
        err = self._lib.curandGenerateUniformDouble(
            self.handle, dst, count)
        if err:
            raise CU.error("curandGenerateUniformDouble", err)

    def generate_normal(self, dst, count=None, mean=0.0, stddev=1.0):
        """Generates specified number of 32-bit normally distributed floats.

        Parameters:
            dst: buffer to store the results or
                 numpy array in case of host generator.
            count: number of 32-bit floats to put to dst or
                   None to fill full dst when the it's size is available.
            mean: mean of normal distribution to generate.
            stddev: stddev of normal distribution to generate.
        """
        dst, count = self._extract_ptr_and_count(dst, count, 4)
        err = self._lib.curandGenerateNormal(
            self.handle, dst, count, float(mean), float(stddev))
        if err:
            raise CU.error("curandGenerateNormal", err)

    def generate_normal_double(self, dst, count=None, mean=0.0, stddev=1.0):
        """Generates specified number of 64-bit normally distributed floats.

        Parameters:
            dst: buffer to store the results or
                 numpy array in case of host generator.
            count: number of 64-bit floats to put to dst or
                   None to fill full dst when the it's size is available.
            mean: mean of normal distribution to generate.
            stddev: stddev of normal distribution to generate.
        """
        dst, count = self._extract_ptr_and_count(dst, count, 8)
        err = self._lib.curandGenerateNormalDouble(
            self.handle, dst, count, float(mean), float(stddev))
        if err:
            raise CU.error("curandGenerateNormalDouble", err)

    def generate_log_normal(self, dst, count=None, mean=0.0, stddev=1.0):
        """Generates specified number of 32-bit log-normally distributed
        floats.

        Parameters:
            dst: buffer to store the results or
                 numpy array in case of host generator.
            count: number of 32-bit floats to put to dst or
                   None to fill full dst when the it's size is available.
            mean: mean of associated normal distribution.
            stddev: stddev of associated normal distribution.
        """
        dst, count = self._extract_ptr_and_count(dst, count, 4)
        err = self._lib.curandGenerateLogNormal(
            self.handle, dst, count, float(mean), float(stddev))
        if err:
            raise CU.error("curandGenerateLogNormal", err)

    def generate_log_normal_double(self, dst, count=None,
                                   mean=0.0, stddev=1.0):
        """Generates specified number of 64-bit log-normally distributed
        floats.

        Parameters:
            dst: buffer to store the results or
                 numpy array in case of host generator.
            count: number of 64-bit floats to put to dst or
                   None to fill full dst when the it's size is available.
            mean: mean of associated normal distribution.
            stddev: stddev of associated normal distribution.
        """
        dst, count = self._extract_ptr_and_count(dst, count, 8)
        err = self._lib.curandGenerateLogNormalDouble(
            self.handle, dst, count, float(mean), float(stddev))
        if err:
            raise CU.error("curandGenerateLogNormalDouble", err)

    def generate_poisson(self, dst, count=None, lam=1.0):
        """Generates specified number of 32-bit unsigned int point values
        with Poisson distribution.

        Parameters:
            dst: buffer to store the results or
                 numpy array in case of host generator.
            count: number of 32-bit unsigned ints to put to dst or
                   None to fill full dst when the it's size is available.
            lam: lambda value of Poisson distribution.
        """
        dst, count = self._extract_ptr_and_count(dst, count, 4)
        err = self._lib.curandGeneratePoisson(
            self.handle, dst, count, float(lam))
        if err:
            raise CU.error("curandGeneratePoisson", err)

    def _release(self):
        if self._lib is not None and self.handle is not None:
            self._lib.curandDestroyGenerator(self.handle)
            self._handle = None

    def __del__(self):
        if self.context is not None and self.context.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self._release()
        if self.context is not None:
            self.context._del_ref(self)
