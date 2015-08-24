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
CUBLAS_STATUS_SUCCESS = 0
CUBLAS_STATUS_NOT_INITIALIZED = 1
CUBLAS_STATUS_ALLOC_FAILED = 3
CUBLAS_STATUS_INVALID_VALUE = 7
CUBLAS_STATUS_ARCH_MISMATCH = 8
CUBLAS_STATUS_MAPPING_ERROR = 11
CUBLAS_STATUS_EXECUTION_FAILED = 13
CUBLAS_STATUS_INTERNAL_ERROR = 14
CUBLAS_STATUS_NOT_SUPPORTED = 15
CUBLAS_STATUS_LICENSE_ERROR = 16


#: Error descriptions
ERRORS = {
    CUBLAS_STATUS_NOT_INITIALIZED: "CUBLAS_STATUS_NOT_INITIALIZED",
    CUBLAS_STATUS_ALLOC_FAILED: "CUBLAS_STATUS_ALLOC_FAILED",
    CUBLAS_STATUS_INVALID_VALUE: "CUBLAS_STATUS_INVALID_VALUE",
    CUBLAS_STATUS_ARCH_MISMATCH: "CUBLAS_STATUS_ARCH_MISMATCH",
    CUBLAS_STATUS_MAPPING_ERROR: "CUBLAS_STATUS_MAPPING_ERROR",
    CUBLAS_STATUS_EXECUTION_FAILED: "CUBLAS_STATUS_EXECUTION_FAILED",
    CUBLAS_STATUS_INTERNAL_ERROR: "CUBLAS_STATUS_INTERNAL_ERROR",
    CUBLAS_STATUS_NOT_SUPPORTED: "CUBLAS_STATUS_NOT_SUPPORTED",
    CUBLAS_STATUS_LICENSE_ERROR: "CUBLAS_STATUS_LICENSE_ERROR"
}


#: cublasOperation_t
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2


#: cublasDataType_t
CUBLAS_DATA_FLOAT = 0
CUBLAS_DATA_DOUBLE = 1
CUBLAS_DATA_HALF = 2
CUBLAS_DATA_INT8 = 3


#: cublasPointerMode_t
CUBLAS_POINTER_MODE_HOST = 0
CUBLAS_POINTER_MODE_DEVICE = 1


def _initialize(backends):
    global lib
    if lib is not None:
        return
    # C function definitions
    # size_t instead of void* is used
    # for convinience with python calls and numpy arrays.
    src = """
    typedef int cublasStatus_t;
    typedef void *cublasHandle_t;
    typedef int cublasOperation_t;
    typedef int cublasPointerMode_t;
    typedef int cublasDataType_t;

    cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
    cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);

    cublasStatus_t cublasSgemm_v2(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        size_t alpha,
        size_t A,
        int lda,
        size_t B,
        int ldb,
        size_t beta,
        size_t C,
        int ldc);
    cublasStatus_t cublasDgemm_v2(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        size_t alpha,
        size_t A,
        int lda,
        size_t B,
        int ldb,
        size_t beta,
        size_t C,
        int ldc);
    cublasStatus_t cublasSgemmEx(
        cublasHandle_t handle,
        cublasOperation_t transa,
        cublasOperation_t transb,
        int m,
        int n,
        int k,
        size_t alpha,
        size_t A,
        cublasDataType_t Atype,
        int lda,
        size_t B,
        cublasDataType_t Btype,
        int ldb,
        size_t beta,
        size_t C,
        cublasDataType_t Ctype,
        int ldc);

    cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle,
                                           cublasPointerMode_t mode);
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
        raise OSError("Could not load cublas library")

    global ERRORS
    for code, msg in ERRORS.items():
        if code in CU.ERRORS:
            s = " | " + msg
            if s not in CU.ERRORS[code]:
                CU.ERRORS[code] += s
        else:
            CU.ERRORS[code] = msg


def initialize(backends=("libcublas.so", "cublas64_65.dll")):
    """Loads shared library.
    """
    cuffi.initialize()
    global lib
    if lib is not None:
        return
    with cuffi.lock:
        _initialize(backends)


class CUBLAS(object):
    """CUBLAS functions can be invoked from this class.
    """
    def __init__(self, context):
        self._context = context
        self._lib = None
        context._add_ref(self)
        initialize()
        handle = ffi.new("cublasHandle_t *")
        with context:
            err = lib.cublasCreate_v2(handle)
        if err:
            self._handle = None
            raise CU.error("cublasCreate_v2", err)
        self._lib = lib  # to hold the reference
        self._handle = handle[0]

    def __int__(self):
        return self.handle

    @property
    def handle(self):
        return self._handle

    @property
    def context(self):
        return self._context

    def set_pointer_mode(self, mode=CUBLAS_POINTER_MODE_DEVICE):
        """Sets the pointer mode used by the cuBLAS library.

        Parameters:
            mode: CUBLAS_POINTER_MODE_HOST or CUBLAS_POINTER_MODE_DEVICE
                  (the default cuBLAS mode is CUBLAS_POINTER_MODE_HOST).
        """
        err = self._lib.cublasSetPointerMode_v2(self.handle, mode)
        if err:
            raise CU.error("cublasSetPointerMode_v2", err)

    def sgemm(self, transA, transB,
              rowsCountA, columnCountB, commonSideLength,
              alpha, A, B, beta, C,
              strideA=0, strideB=0, strideC=0):
        """Single precision (float) GEneral Matrix Multiplication.

        Matrices are always in column order.

        C = alpha * dot(A, B) + beta * C
        C = alpha * dot(A^T, B) + beta * C
        C = alpha * dot(A, B^T) + beta * C
        C = alpha * dot(A^T, B^T) + beta * C

        alpha, A, B, beta, C can be numpy array, Memory object,
        cffi pointer or int.

        Parameters:
            transA: how matrix A is to be transposed
                    (CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C).
            transB: how matrix B is to be transposed
                    (CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C).
            rowsCountA: number of rows in matrix A.
            columnCountB: number of columns in matrix B.
            commonSideLength: length of the common side of the matrices.
            alpha: the factor of matrix A.
            A: matrix A.
            B: matrix B.
            beta: the factor of matrix C.
            C: Buffer object storing matrix C.
            strideA: leading dimension of matrix A:
                     clblasTrans: >= commonSideLength,
                     else: >= rowsCountA.
            strideB: leading dimension of matrix B:
                     clblasTrans: >= columnCountB,
                     else: >= commonSideLength.
            strideC: leading dimension of matrix C: >= rowsCountA.

        Returns:
            None.
        """
        if not strideA:
            strideA = commonSideLength if transA != CUBLAS_OP_N else rowsCountA
        if not strideB:
            strideB = (columnCountB if transB != CUBLAS_OP_N
                       else commonSideLength)
        if not strideC:
            strideC = rowsCountA
        err = self._lib.cublasSgemm_v2(
            self.handle, transA, transB, rowsCountA, columnCountB,
            commonSideLength, CU.extract_ptr(alpha), A, strideA,
            B, strideB, CU.extract_ptr(beta), C, strideC)
        if err:
            raise CU.error("cublasSgemm_v2", err)

    def dgemm(self, transA, transB,
              rowsCountA, columnCountB, commonSideLength,
              alpha, A, B, beta, C,
              strideA=0, strideB=0, strideC=0):
        """Double precision (double) GEneral Matrix Multiplication.

        Matrices are always in column order.

        C = alpha * dot(A, B) + beta * C
        C = alpha * dot(A^T, B) + beta * C
        C = alpha * dot(A, B^T) + beta * C
        C = alpha * dot(A^T, B^T) + beta * C

        alpha, A, B, beta, C can be numpy array, Memory object,
        cffi pointer or int.

        Parameters:
            transA: how matrix A is to be transposed
                    (CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C).
            transB: how matrix B is to be transposed
                    (CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C).
            rowsCountA: number of rows in matrix A.
            columnCountB: number of columns in matrix B.
            commonSideLength: length of the common side of the matrices.
            alpha: the factor of matrix A.
            A: matrix A.
            B: matrix B.
            beta: the factor of matrix C.
            C: Buffer object storing matrix C.
            strideA: leading dimension of matrix A:
                     clblasTrans: >= commonSideLength,
                     else: >= rowsCountA.
            strideB: leading dimension of matrix B:
                     clblasTrans: >= columnCountB,
                     else: >= commonSideLength.
            strideC: leading dimension of matrix C: >= rowsCountA.

        Returns:
            None.
        """
        if not strideA:
            strideA = commonSideLength if transA != CUBLAS_OP_N else rowsCountA
        if not strideB:
            strideB = (columnCountB if transB != CUBLAS_OP_N
                       else commonSideLength)
        if not strideC:
            strideC = rowsCountA
        err = self._lib.cublasDgemm_v2(
            self.handle, transA, transB, rowsCountA, columnCountB,
            commonSideLength, CU.extract_ptr(alpha), A, strideA,
            B, strideB, CU.extract_ptr(beta), C, strideC)
        if err:
            raise CU.error("cublasDgemm_v2", err)

    def sgemm_ex(self, transA, transB,
                 rowsCountA, columnCountB, commonSideLength,
                 alpha, A, B, beta, C,
                 strideA=0, strideB=0, strideC=0,
                 dtypeA=CUBLAS_DATA_HALF, dtypeB=CUBLAS_DATA_HALF,
                 dtypeC=CUBLAS_DATA_HALF):
        """Single precision (float) GEneral Matrix Multiplication
        with support of different data types for each matrix.

        Matrices are always in column order.

        C = alpha * dot(A, B) + beta * C
        C = alpha * dot(A^T, B) + beta * C
        C = alpha * dot(A, B^T) + beta * C
        C = alpha * dot(A^T, B^T) + beta * C

        alpha, A, B, beta, C can be numpy array, Memory object,
        cffi pointer or int.

        Parameters:
            transA: how matrix A is to be transposed
                    (CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C).
            transB: how matrix B is to be transposed
                    (CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C).
            rowsCountA: number of rows in matrix A.
            columnCountB: number of columns in matrix B.
            commonSideLength: length of the common side of the matrices.
            alpha: the factor of matrix A.
            A: matrix A.
            B: matrix B.
            beta: the factor of matrix C.
            C: Buffer object storing matrix C.
            strideA: leading dimension of matrix A:
                     clblasTrans: >= commonSideLength,
                     else: >= rowsCountA.
            strideB: leading dimension of matrix B:
                     clblasTrans: >= columnCountB,
                     else: >= commonSideLength.
            strideC: leading dimension of matrix C: >= rowsCountA.
            dtypeA: data type of matrix A
                    (CUBLAS_DATA_FLOAT, CUBLAS_DATA_DOUBLE,
                     CUBLAS_DATA_HALF, CUBLAS_DATA_INT8).
            dtypeB: data type of matrix B
                    (CUBLAS_DATA_FLOAT, CUBLAS_DATA_DOUBLE,
                     CUBLAS_DATA_HALF, CUBLAS_DATA_INT8).
            dtypeC: data type of matrix C
                    (CUBLAS_DATA_FLOAT, CUBLAS_DATA_DOUBLE,
                     CUBLAS_DATA_HALF, CUBLAS_DATA_INT8).

        Returns:
            None.
        """
        if not strideA:
            strideA = commonSideLength if transA != CUBLAS_OP_N else rowsCountA
        if not strideB:
            strideB = (columnCountB if transB != CUBLAS_OP_N
                       else commonSideLength)
        if not strideC:
            strideC = rowsCountA
        err = self._lib.cublasSgemmEx(
            self.handle, transA, transB, rowsCountA, columnCountB,
            commonSideLength, CU.extract_ptr(alpha), A, dtypeA, strideA,
            B, dtypeB, strideB, CU.extract_ptr(beta), C, dtypeC, strideC)
        if err:
            raise CU.error("cublasSgemmEx", err)

    @staticmethod
    def gemm(dtype):
        import numpy
        if dtype == numpy.float32:
            return CUBLAS.sgemm
        if dtype == numpy.float64:
            return CUBLAS.dgemm
        if dtype == numpy.float16:
            return CUBLAS.sgemm_ex
        raise ValueError("Invalid dtype %s" % dtype)

    def _release(self):
        if self._lib is not None and self.handle is not None:
            self._lib.cublasDestroy_v2(self.handle)
            self._handle = None

    def __del__(self):
        if self.context.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self._release()
        self.context._del_ref(self)
