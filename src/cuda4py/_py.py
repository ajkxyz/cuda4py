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
Helper classes.
"""
import cuda4py._cffi as cu
import gc
import os
import subprocess
import sys
import tempfile


class CUDARuntimeError(RuntimeError):
    def __init__(self, msg, code):
        super(CUDARuntimeError, self).__init__(msg)
        self.code = code


class CU(object):
    """Base CUDA class.

    Attributes:
        _lib: handle to cffi.FFI object.
        _handle: cffi handle to CUDA object.
    """
    ERRORS = {
        cu.CUDA_SUCCESS: "CUDA_SUCCESS",
        cu.CUDA_ERROR_INVALID_VALUE: "CUDA_ERROR_INVALID_VALUE",
        cu.CUDA_ERROR_OUT_OF_MEMORY: "CUDA_ERROR_OUT_OF_MEMORY",
        cu.CUDA_ERROR_NOT_INITIALIZED: "CUDA_ERROR_NOT_INITIALIZED",
        cu.CUDA_ERROR_DEINITIALIZED: "CUDA_ERROR_DEINITIALIZED",
        cu.CUDA_ERROR_PROFILER_DISABLED: "CUDA_ERROR_PROFILER_DISABLED",
        cu.CUDA_ERROR_PROFILER_NOT_INITIALIZED:
        "CUDA_ERROR_PROFILER_NOT_INITIALIZED",
        cu.CUDA_ERROR_PROFILER_ALREADY_STARTED:
        "CUDA_ERROR_PROFILER_ALREADY_STARTED",
        cu.CUDA_ERROR_PROFILER_ALREADY_STOPPED:
        "CUDA_ERROR_PROFILER_ALREADY_STOPPED",
        cu.CUDA_ERROR_NO_DEVICE: "CUDA_ERROR_NO_DEVICE",
        cu.CUDA_ERROR_INVALID_DEVICE: "CUDA_ERROR_INVALID_DEVICE",
        cu.CUDA_ERROR_INVALID_IMAGE: "CUDA_ERROR_INVALID_IMAGE",
        cu.CUDA_ERROR_INVALID_CONTEXT: "CUDA_ERROR_INVALID_CONTEXT",
        cu.CUDA_ERROR_CONTEXT_ALREADY_CURRENT:
        "CUDA_ERROR_CONTEXT_ALREADY_CURRENT",
        cu.CUDA_ERROR_MAP_FAILED: "CUDA_ERROR_MAP_FAILED",
        cu.CUDA_ERROR_UNMAP_FAILED: "CUDA_ERROR_UNMAP_FAILED",
        cu.CUDA_ERROR_ARRAY_IS_MAPPED: "CUDA_ERROR_ARRAY_IS_MAPPED",
        cu.CUDA_ERROR_ALREADY_MAPPED: "CUDA_ERROR_ALREADY_MAPPED",
        cu.CUDA_ERROR_NO_BINARY_FOR_GPU: "CUDA_ERROR_NO_BINARY_FOR_GPU",
        cu.CUDA_ERROR_ALREADY_ACQUIRED: "CUDA_ERROR_ALREADY_ACQUIRED",
        cu.CUDA_ERROR_NOT_MAPPED: "CUDA_ERROR_NOT_MAPPED",
        cu.CUDA_ERROR_NOT_MAPPED_AS_ARRAY: "CUDA_ERROR_NOT_MAPPED_AS_ARRAY",
        cu.CUDA_ERROR_NOT_MAPPED_AS_POINTER:
        "CUDA_ERROR_NOT_MAPPED_AS_POINTER",
        cu.CUDA_ERROR_ECC_UNCORRECTABLE: "CUDA_ERROR_ECC_UNCORRECTABLE",
        cu.CUDA_ERROR_UNSUPPORTED_LIMIT: "CUDA_ERROR_UNSUPPORTED_LIMIT",
        cu.CUDA_ERROR_CONTEXT_ALREADY_IN_USE:
        "CUDA_ERROR_CONTEXT_ALREADY_IN_USE",
        cu.CUDA_ERROR_PEER_ACCESS_UNSUPPORTED:
        "CUDA_ERROR_PEER_ACCESS_UNSUPPORTED",
        cu.CUDA_ERROR_INVALID_PTX: "CUDA_ERROR_INVALID_PTX",
        cu.CUDA_ERROR_INVALID_GRAPHICS_CONTEXT:
        "CUDA_ERROR_INVALID_GRAPHICS_CONTEXT",
        cu.CUDA_ERROR_INVALID_SOURCE: "CUDA_ERROR_INVALID_SOURCE",
        cu.CUDA_ERROR_FILE_NOT_FOUND: "CUDA_ERROR_FILE_NOT_FOUND",
        cu.CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND:
        "CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND",
        cu.CUDA_ERROR_SHARED_OBJECT_INIT_FAILED:
        "CUDA_ERROR_SHARED_OBJECT_INIT_FAILED",
        cu.CUDA_ERROR_OPERATING_SYSTEM: "CUDA_ERROR_OPERATING_SYSTEM",
        cu.CUDA_ERROR_INVALID_HANDLE: "CUDA_ERROR_INVALID_HANDLE",
        cu.CUDA_ERROR_NOT_FOUND: "CUDA_ERROR_NOT_FOUND",
        cu.CUDA_ERROR_NOT_READY: "CUDA_ERROR_NOT_READY",
        cu.CUDA_ERROR_ILLEGAL_ADDRESS: "CUDA_ERROR_ILLEGAL_ADDRESS",
        cu.CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES:
        "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES",
        cu.CUDA_ERROR_LAUNCH_TIMEOUT: "CUDA_ERROR_LAUNCH_TIMEOUT",
        cu.CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING:
        "CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING",
        cu.CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
        "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED",
        cu.CUDA_ERROR_PEER_ACCESS_NOT_ENABLED:
        "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED",
        cu.CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE:
        "CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE",
        cu.CUDA_ERROR_CONTEXT_IS_DESTROYED: "CUDA_ERROR_CONTEXT_IS_DESTROYED",
        cu.CUDA_ERROR_ASSERT: "CUDA_ERROR_ASSERT",
        cu.CUDA_ERROR_TOO_MANY_PEERS: "CUDA_ERROR_TOO_MANY_PEERS",
        cu.CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED:
        "CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED",
        cu.CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED:
        "CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED",
        cu.CUDA_ERROR_HARDWARE_STACK_ERROR: "CUDA_ERROR_HARDWARE_STACK_ERROR",
        cu.CUDA_ERROR_ILLEGAL_INSTRUCTION: "CUDA_ERROR_ILLEGAL_INSTRUCTION",
        cu.CUDA_ERROR_MISALIGNED_ADDRESS: "CUDA_ERROR_MISALIGNED_ADDRESS",
        cu.CUDA_ERROR_INVALID_ADDRESS_SPACE:
        "CUDA_ERROR_INVALID_ADDRESS_SPACE",
        cu.CUDA_ERROR_INVALID_PC: "CUDA_ERROR_INVALID_PC",
        cu.CUDA_ERROR_LAUNCH_FAILED: "CUDA_ERROR_LAUNCH_FAILED",
        cu.CUDA_ERROR_NOT_PERMITTED: "CUDA_ERROR_NOT_PERMITTED",
        cu.CUDA_ERROR_NOT_SUPPORTED: "CUDA_ERROR_NOT_SUPPORTED",
        cu.CUDA_ERROR_UNKNOWN: "CUDA_ERROR_UNKNOWN"
    }

    def __init__(self):
        self._lib = cu.lib  # to hold the reference
        self._handle = None

    def __int__(self):
        return self.handle

    @property
    def handle(self):
        """Returns cffi handle to CUDA object.
        """
        return self._handle

    @staticmethod
    def get_error_name_from_code(code):
        return CU.ERRORS.get(code, "UNKNOWN")

    @staticmethod
    def get_error_description(code):
        return "%s (%d)" % (CU.get_error_name_from_code(code), code)

    @staticmethod
    def error(method_name, code):
        return CUDARuntimeError(
            "%s() failed with error %s" %
            (method_name, CU.get_error_description(code)), code)

    @staticmethod
    def extract_ptr(host_array):
        """Returns cffi pointer to host_array.
        """
        arr = getattr(host_array, "__array_interface__", None)
        if arr is not None:
            host_ptr = arr["data"][0]
        elif isinstance(host_array, CU):
            host_ptr = host_array.handle
        else:
            host_ptr = host_array
        return 0 if host_ptr is None else int(cu.ffi.cast("size_t", host_ptr))

    @staticmethod
    def extract_ptr_and_size(host_array, size):
        """Returns cffi pointer to host_array and its size.
        """
        arr = getattr(host_array, "__array_interface__", None)
        if arr is not None:
            host_ptr = arr["data"][0]
            if size is None:
                size = host_array.nbytes
        elif isinstance(host_array, CU):
            host_ptr = host_array.handle
            if size is None:
                size = host_array.size
        else:
            host_ptr = host_array
            if size is None:
                raise ValueError("size should be set "
                                 "in case of non-numpy host_array")
        return (0 if host_ptr is None
                else int(cu.ffi.cast("size_t", host_ptr)), size)


class Memory(CU):
    """Manages host-device memory.

    Attributes:
        handle: cffi handle to the allocation (int).
        context: Context to hold the reference.
        size: size of the allocation.
        flags: flags of the allocation.
    """
    def __init__(self, context, size_or_ndarray, flags=0):
        super(Memory, self).__init__()
        context._add_ref(self)
        self._context = context
        size = getattr(size_or_ndarray, "nbytes", size_or_ndarray)
        self._size = size
        self._flags = flags
        self._device_alloc()
        if size is not size_or_ndarray:
            self.to_device(size_or_ndarray)

    def _device_alloc(self):
        """Call CUDA api to allocate memory here.
        """
        raise NotImplementedError()

    @property
    def context(self):
        return self._context

    @property
    def size(self):
        """Size of the allocation.
        """
        return self._size

    @property
    def flags(self):
        """Flags supplied for this allocation.
        """
        return self._flags

    def to_host(self, host_array, offs=0, size=None):
        """Copies memory from device to host.

        The function will block until completion.

        Parameters:
            host_array: host array to copy to (numpy, cffi handle or int).
            offs: offset from the device memory base in bytes.
            size: size of the memory to copy in bytes.
        """
        ptr, size = CU.extract_ptr_and_size(host_array, size)
        err = self._lib.cuMemcpyDtoH_v2(ptr, self.handle + offs, size)
        if err:
            raise CU.error("cuMemcpyDtoH_v2", err)

    def to_device(self, host_array, offs=0, size=None):
        """Copies memory from host to device.

        The function will block until completion.

        Parameters:
            host_array: host array to copy from (numpy, cffi handle or int).
            offs: offset from the device memory base in bytes.
            size: size of the memory to copy in bytes.
        """
        ptr, size = CU.extract_ptr_and_size(host_array, size)
        err = self._lib.cuMemcpyHtoD_v2(self.handle + offs, ptr, size)
        if err:
            raise CU.error("cuMemcpyHtoD_v2", err)

    def to_device_async(self, host_array, offs=0, size=None, stream=None):
        """Copies memory from host to device.

        The function will NOT block.

        Parameters:
            host_array: host array to copy from (numpy, cffi handle or int).
            offs: offset from the device memory base in bytes.
            size: size of the memory to copy in bytes.
            stream: compute stream.
        """
        ptr, size = CU.extract_ptr_and_size(host_array, size)
        err = self._lib.cuMemcpyHtoDAsync_v2(
            self.handle + offs, ptr, size,
            0 if stream is None else stream)
        if err:
            raise CU.error("cuMemcpyHtoDAsync_v2", err)

    def from_device_async(self, src, dst_offs=0, size=None, stream=None):
        """Copies memory from device to device.

        The function will NOT block.

        Parameters:
            src: source device buffer to copy memory from (Memory, int).
            dst_offs: offset from this device memory base in bytes.
            size: size of the memory to copy in bytes
                  (defaults to this buffer size - dst_offs).
            stream: compute stream.
        """
        err = self._lib.cuMemcpyDtoDAsync_v2(
            self.handle + dst_offs, int(src),
            self.size - dst_offs if size is None else size,
            0 if stream is None else stream)
        if err:
            raise CU.error("cuMemcpyDtoDAsync_v2", err)

    def memset32_async(self, value=0, offs=0, size=None, stream=None):
        """Sets memory object with 32-bit integer value.

        The function will NOT block.

        Parameters:
            value: value to set with.
            offs: offset from the device memory base in 32-bit elements.
            size: size of device memory to set in 32-bit elements.
            stream: compute stream.
        """
        err = self._lib.cuMemsetD32Async(
            self.handle + (offs << 2), value,
            (self.size >> 2) - offs if size is None else size,
            0 if stream is None else stream)
        if err:
            raise CU.error("cuMemsetD32Async", err)

    def memcpy_3d_async(self, src_origin, dst_origin, region,
                        src_pitch=0, src_height=0,
                        dst_pitch=0, dst_height=0,
                        src=None, dst=None, stream=None):
        """Copies memory for 3D arrays.

        The function will NOT block.

        Parameters:
            src_origin: (src_x_in_bytes, src_y, src_z).
            dst_origin: (dst_x_in_bytes, dst_y, dst_z).
            region: (width_in_bytes, height, depth) of the region to copy.
            src_pitch: the length of each source row in bytes.
            src_height: the height of each source 2D slice.
            dst_pitch: the length of each destination row in bytes.
            dst_height: the height of each destination 2D slice.
            src: source:
                None - use self as the source,
                convertible to int - use as the device buffer address,
                numpy array - use as the host buffer address.
            dst: destination:
                None - use self as the destination,
                convertible to int - use as the device buffer address,
                numpy array - use as the host buffer address.
            stream: compute stream.
        """
        p_copy = cu.ffi.new("CUDA_MEMCPY3D *")

        p_copy.WidthInBytes = region[0]
        p_copy.Height = region[1]
        p_copy.Depth = region[2]

        p_copy.srcXInBytes = src_origin[0]
        p_copy.srcY = src_origin[1]
        p_copy.srcZ = src_origin[2]

        p_copy.dstXInBytes = dst_origin[0]
        p_copy.dstY = dst_origin[1]
        p_copy.dstZ = dst_origin[2]

        p_copy.srcPitch = src_pitch if src_pitch else src_origin[0] + region[0]
        p_copy.srcHeight = (src_height if src_height
                            else src_origin[1] + region[1])

        p_copy.dstPitch = dst_pitch if dst_pitch else dst_origin[0] + region[0]
        p_copy.dstHeight = (dst_height if dst_height
                            else dst_origin[1] + region[1])

        if src is None:
            p_copy.srcDevice = self.handle
            p_copy.srcMemoryType = cu.CU_MEMORYTYPE_DEVICE
        else:
            arr = getattr(src, "__array_interface__", None)
            if arr is None:
                p_copy.srcDevice = int(src)
                p_copy.srcMemoryType = cu.CU_MEMORYTYPE_DEVICE
            else:
                p_copy.srcHost = arr["data"][0]
                p_copy.srcMemoryType = cu.CU_MEMORYTYPE_HOST

        if dst is None:
            p_copy.dstDevice = self.handle
            p_copy.dstMemoryType = cu.CU_MEMORYTYPE_DEVICE
        else:
            arr = getattr(dst, "__array_interface__", None)
            if arr is None:
                p_copy.dstDevice = int(dst)
                p_copy.dstMemoryType = cu.CU_MEMORYTYPE_DEVICE
            else:
                p_copy.dstHost = arr["data"][0]
                p_copy.dstMemoryType = cu.CU_MEMORYTYPE_HOST

        err = self._lib.cuMemcpy3DAsync_v2(
            p_copy, 0 if stream is None else stream)
        if err:
            raise CU.error("cuMemcpy3DAsync_v2", err)

    def _release_mem(self):
        """Do actual memory release in child class.

        self.handle garanted to be not None.
        """
        raise NotImplementedError()

    def _release(self):
        """Releases allocation.
        """
        if self.handle is not None:
            self._release_mem()
            self._handle = None

    def __del__(self):
        if self.context.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self._release()
        self.context._del_ref(self)


class MemPtr(Memory):
    """Provides memory functions over arbitrary address.

    Attributes:
        owner: holds the reference to the parent object
               for correct destructor call order.
    """
    def __init__(self, context, ptr, owner=None, size=0):
        """Associates pointer to memory object.

        Parameters:
            context: the device context.
            ptr: pointer to the GPU memory.
            owner: optional owner to hold the reference.
            size: optional size of the memory region.
        """
        super(MemPtr, self).__init__(context, size)
        self._handle = int(ptr)
        self.owner = owner

    def _device_alloc(self):
        """Does nothing by design.
        """
        pass

    def _release_mem(self):
        """Does nothing by design.
        """
        pass


class MemAlloc(Memory):
    """Allocates memory via cuMemAlloc.

    Attributes:
        handle: pointer in the device address space (int).
    """
    def _device_alloc(self):
        ptr = cu.ffi.new("CUdeviceptr *")
        with self.context:
            err = self._lib.cuMemAlloc_v2(ptr, self.size)
        if err:
            raise CU.error("cuMemAlloc_v2", err)
        self._handle = int(ptr[0])

    def _release_mem(self):
        self._lib.cuMemFree_v2(self.handle)


class MemAllocManaged(Memory):
    """Allocated memory via cuMemAllocManaged.

    Attributes:
        handle: pointer in the unified address space (int).
    """
    def __init__(self, context, size_or_ndarray,
                 flags=cu.CU_MEM_ATTACH_GLOBAL):
        super(MemAllocManaged, self).__init__(context, size_or_ndarray, flags)

    def _device_alloc(self):
        ptr = cu.ffi.new("CUdeviceptr *")
        with self.context:
            err = self._lib.cuMemAllocManaged(ptr, self.size, self.flags)
        if err:
            raise CU.error("cuMemAllocManaged", err)
        self._handle = int(ptr[0])

    def _release_mem(self):
        self._lib.cuMemFree_v2(self.handle)


class MemHostAlloc(Memory):
    """Allocates memory via cuMemHostAlloc.

    Attributes:
        handle: pointer in the host address space (int).
    """
    def __init__(self, context, size_or_ndarray,
                 flags=(cu.CU_MEMHOSTALLOC_PORTABLE |
                        cu.CU_MEMHOSTALLOC_DEVICEMAP)):
        super(MemHostAlloc, self).__init__(context, size_or_ndarray, flags)

    def _device_alloc(self):
        pp = cu.ffi.new("size_t *")
        with self.context:
            err = self._lib.cuMemHostAlloc(pp, self.size, self.flags)
        if err:
            raise CU.error("cuMemHostAlloc", err)
        self._handle = int(pp[0])

    @property
    def device_pointer(self):
        """Returns device pointer in case of non-unified address.
        """
        dptr = cu.ffi.new("CUdeviceptr *")
        err = cu.lib.cuMemHostGetDevicePointer_v2(dptr, self.handle, 0)
        if err:
            raise CU.error("cuMemHostGetDevicePointer_v2", err)
        return int(dptr[0])

    @property
    def buffer(self):
        return cu.ffi.buffer(cu.ffi.cast("void *", self.handle), self.size)

    def _release_mem(self):
        self._lib.cuMemFreeHost(self.handle)


class skip(object):
    """For skipping arguments when passed to set_args.
    """
    def __init__(self, amount=1):
        self.amount = amount


class Function(CU):
    """Holds cffi handle to CUDA function.
    """
    def __init__(self, module, name):
        super(Function, self).__init__()
        self._module = module
        func = cu.ffi.new("CUfunction *")
        with module.context:
            err = self._lib.cuModuleGetFunction(
                func, module.handle,
                cu.ffi.new("char[]", name.encode("utf-8")))
        if err:
            raise CU.error("cuModuleGetFunction", err)
        self._handle = int(func[0])
        # Holds references to the original python objects
        self._refs = []
        # Holds cffi data, copied from the original python objects
        self._args = []
        # Holds pointers to the cffi data
        self._params = None

    @property
    def module(self):
        return self._module

    def max_active_blocks_per_multiprocessor(self, block_size,
                                             dynamic_smem_size=0):
        """Calculates occupancy of a function.

        Parameters:
            block_size: block size the kernel is intended to be launched with.
            dynamic_smem_size: per-block dynamic shared memory usage intended,
                               in bytes.

        Returns:
            num_blocks: the number of the maximum active blocks
                        per streaming multiprocessor.
        """
        num_blocks = cu.ffi.new("int *")
        err = self._lib.cuOccupancyMaxActiveBlocksPerMultiprocessor(
            num_blocks, self.handle, block_size, dynamic_smem_size)
        if err:
            raise CU.error("cuOccupancyMaxActiveBlocksPerMultiprocessor", err)
        return int(num_blocks[0])

    def max_potential_block_size(self, block_size_to_dynamic_smem_size=None,
                                 dynamic_smem_size=0, block_size_limit=0):
        """Suggests a launch configuration with reasonable occupancy.

        Parameters:
            block_size_to_dynamic_smem_size:
                callback that returns dynamic shared memory size in bytes
                for a given block size, for example: lambda x: x ** 2
            dynamic_smem_size: dynamic shared memory usage intended, in bytes.
            block_size_limit: the maximum block size the kernel
                              is designed to handle.

        Returns:
            min_grid_size, block_size:
                minimum grid size needed to achieve the maximum occupancy,
                maximum block size that can achieve the maximum occupancy.
        """
        min_grid_size = cu.ffi.new("int *")
        block_size = cu.ffi.new("int *")
        err = self._lib.cuOccupancyMaxPotentialBlockSize(
            min_grid_size, block_size, self.handle,
            cu.ffi.NULL if block_size_to_dynamic_smem_size is None else
            cu.ffi.callback("size_t(int)", block_size_to_dynamic_smem_size),
            dynamic_smem_size, block_size_limit)
        if err:
            raise CU.error("cuOccupancyMaxPotentialBlockSize", err)
        return int(min_grid_size[0]), int(block_size[0])

    def set_args(self, *args):
        self._params = None
        i = 0
        for arg in args:
            if arg is skip:
                i += 1
                continue
            elif isinstance(arg, skip):
                i += arg.amount
                continue
            self.set_arg(i, arg)
            i += 1

    def set_arg(self, i, arg):
        self._params = None
        while len(self._args) <= i:
            ptr = cu.ffi.new("size_t *")
            self._args.append(ptr)
            self._refs.append(None)
        self._refs[i] = arg
        ptr = cu.ffi.new("size_t *")
        if isinstance(arg, CU):
            ptr[0] = cu.ffi.cast("size_t", arg.handle)
            self._args[i] = ptr
            return
        arr = getattr(arg, "__array_interface__", None)
        if arr is not None:  # save address to the contents of the numpy array
            self._args[i] = cu.ffi.cast("size_t *", arr["data"][0])
            return
        ptr[0] = cu.ffi.cast("size_t", 0 if arg is None else arg)
        self._args[i] = ptr

    def __call__(self, grid_dims, block_dims=(1, 1, 1), args_tuple=None,
                 shared_mem_bytes=0, stream=None):
        if args_tuple is not None:
            self.set_args(*args_tuple)
        if self._params is None:
            n = len(self._args)
            if n:
                self._params = cu.ffi.new("void*[]", n)
                self._params[0:n] = self._args[0:n]
            else:
                self._params = cu.ffi.NULL
        err = self._lib.cuLaunchKernel(
            self.handle, grid_dims[0], grid_dims[1], grid_dims[2],
            block_dims[0], block_dims[1], block_dims[2],
            shared_mem_bytes, 0 if stream is None else stream,
            self._params, cu.ffi.NULL)
        if err:
            raise CU.error("cuLaunchKernel", err)


class Module(CU):
    """Class for compiling CUDA module from source (nvcc is required)
    or linking from PTX/cubin/fatbin binary.
    """
    OPTIONS_PTX = ("-ptx",)
    OPTIONS_CUBLAS = ("-cubin", "-lcudadevrt", "-lcublas_device", "-dlink")

    def __init__(self, context, ptx=None, source=None, source_file=None,
                 nvcc_options=("-O3", "--ftz=true", "--fmad=true"),
                 nvcc_path="nvcc", include_dirs=(),
                 nvcc_options2=OPTIONS_PTX):
        """Calls cuModuleLoadData, invoking nvcc if ptx is not None.

        Parameters:
            context: Context instance.
            ptx: str or bytes of PTX, cubin or fatbin.
            source: kernel source code to invoke nvcc on.
            source_file: path to the file with kernel code.
            nvcc_options: general options for nvcc.
            nvcc_path: path to execute as nvcc.
            include_dirs: include directories for nvcc.
            nvcc_options2: more options for nvcc (defaults to ("-ptx",)),
                example: ("-cubin", "-lcudadevrt", "-lcublas_device", "-dlink")
        """
        super(Module, self).__init__()
        context._add_ref(self)
        self._context = context
        self._ptx = None
        self._stdout = None
        self._stderr = None

        if ptx is None:
            if source is None and source_file is None:
                raise ValueError("Either ptx, source or source_file "
                                 "should be provided")
            if source is not None:
                fout, source_file = tempfile.mkstemp(".cu")
                os.write(fout, source.encode("utf-8"))
                os.close(fout)

            fptx, ptx_file = tempfile.mkstemp(".ptx")
            os.close(fptx)

            nvcc_options = list(nvcc_options)
            for dirnme in include_dirs:
                if not len(dirnme):
                    continue
                nvcc_options.extend(("-I", dirnme))
            nvcc_options.append("-arch=sm_%d%d" %
                                context.device.compute_capability)
            nvcc_options.extend(nvcc_options2)
            nvcc_options.extend(("-o", ptx_file))
            nvcc_options.insert(0, nvcc_path)
            nvcc_options.insert(1, source_file)
            try:
                proc = subprocess.Popen(
                    nvcc_options, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                self._stdout, self._stderr = proc.communicate()
                err = proc.returncode
                with open(ptx_file, "rb") as fptx:
                    ptx = fptx.read()
            except OSError:
                raise RuntimeError("Could not execute %s" %
                                   " ".join(nvcc_options))
            finally:
                os.unlink(ptx_file)
                if source is not None:
                    os.unlink(source_file)
            if err:
                raise RuntimeError("nvcc returned %d with stderr:\n%s"
                                   "\nCommand line was:\n%s" %
                                   (err, self.stderr.decode("utf-8"),
                                    " ".join(nvcc_options)))
        self._ptx = ptx.encode("utf-8") if type(ptx) != type(b"") else ptx

        # Workaround to prevent deadlock in pypy when doing
        # garbage collection inside CFFI.new
        gc.collect()

        module = cu.ffi.new("CUmodule *")
        ptx = cu.ffi.new("unsigned char[]", self._ptx)
        with context:
            err = self._lib.cuModuleLoadData(module, ptx)
        if err:
            raise CU.error("cuModuleLoadData", err)
        self._handle = int(module[0])

    def create_function(self, name):
        return Function(self, name)

    @property
    def context(self):
        return self._context

    @property
    def ptx(self):
        return self._ptx

    @property
    def stderr(self):
        return self._stderr

    def get_func(self, name):
        """Returns function pointer.
        """
        return Function(self, name)

    def get_global(self, name):
        """Returns tuple (pointer, size).
        """
        ptr = cu.ffi.new("CUdeviceptr *")
        sz = cu.ffi.new("size_t *")
        with self.context:
            err = self._lib.cuModuleGetGlobal_v2(
                ptr, sz, self.handle,
                cu.ffi.new("char[]", name.encode("utf-8")))
        if err:
            raise CU.error("cuModuleGetGlobal_v2", err)
        return int(ptr[0]), int(sz[0])

    def _release(self):
        if self.handle is not None:
            with self.context:
                self.context.synchronize()
                self._lib.cuModuleUnload(self.handle)
            self._handle = None

    def __del__(self):
        if self.context.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self._release()
        self.context._del_ref(self)


class Context(CU):
    """Holds CUDA context associated with the selected Device.

    Attributes:
        _handle: cffi handle for CUDA context.
        _n_refs: reference count as a workaround for possible
                 incorrect destructor call order, see
                 http://bugs.python.org/issue23720
                 (weakrefs do not help here).
    """
    default_flags = cu.CU_CTX_SCHED_AUTO | cu.CU_CTX_MAP_HOST
    context_count = 0  # number of active contexts

    def __init__(self, device, flags=0, handle=None):
        """Initializes CUDA context.

        Parameters:
            device: Device instance or int.
            flags: context creation flags.
            handle: external context handle,
                    if set, device and flags are ignored, and
                    cuCtxDestroy_v2 will not be called in destructor.
        """
        super(Context, self).__init__()
        self._n_refs = 1
        if handle is None:
            ctx = cu.ffi.new("CUcontext *")
            err = self._lib.cuCtxCreate_v2(
                ctx, flags if flags else Context.default_flags, device)
            if err:
                raise CU.error("cuCtxCreate_v2", err)
            self._handle = int(ctx[0])
            self._own_handle = True
        else:
            self._handle = int(handle)
            self._own_handle = False
        self.device = device
        Context.context_count += 1

    def _add_ref(self, obj):
        self._n_refs += 1

    def _del_ref(self, obj):
        self._n_refs -= 1
        if self._n_refs <= 0:
            self._release()

    def synchronize(self):
        if self.handle is None:
            return
        err = self._lib.cuCtxSynchronize()
        if err:
            raise CU.error("cuCtxSynchronize", err)

    def mem_alloc(self, size_or_ndarray):
        return MemAlloc(self, size_or_ndarray)

    def mem_alloc_managed(self, size_or_ndarray,
                          flags=cu.CU_MEM_ATTACH_GLOBAL):
        return MemAllocManaged(self, size_or_ndarray, flags)

    def mem_host_alloc(self, size_or_ndarray,
                       flags=(cu.CU_MEMHOSTALLOC_PORTABLE |
                              cu.CU_MEMHOSTALLOC_DEVICEMAP)):
        return MemHostAlloc(self, size_or_ndarray, flags)

    def create_module(self, ptx=None, source=None, source_file=None,
                      nvcc_options=("-O3", "--ftz=true", "--fmad=true"),
                      nvcc_path="nvcc", include_dirs=(),
                      nvcc_options2=Module.OPTIONS_PTX):
        return Module(self, ptx, source, source_file,
                      nvcc_options, nvcc_path, include_dirs,
                      nvcc_options2)

    def set_current(self):
        err = self._lib.cuCtxSetCurrent(self.handle)
        if err:
            raise CU.error("cuCtxSetCurrent", err)

    @staticmethod
    def get_current():
        """Returns cffi handle to the current CUDA context.
        """
        ctx = cu.ffi.new("CUcontext *")
        err = cu.lib.cuCtxGetCurrent(ctx)
        if err:
            raise CU.error("cuCtxGetCurrent", err)
        return ctx[0]

    def push_current(self):
        err = self._lib.cuCtxPushCurrent_v2(self.handle)
        if err:
            raise CU.error("cuCtxPushCurrent_v2", err)

    @staticmethod
    def pop_current():
        ctx = cu.ffi.new("CUcontext *")
        err = cu.lib.cuCtxPopCurrent_v2(ctx)
        if err:
            raise CU.error("cuCtxPopCurrent_v2", err)
        return ctx[0]

    def __enter__(self):
        if self.handle is None:
            raise SystemError("Incorrect destructor call order detected")
        self.push_current()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.handle is not None:
            self._lib.cuCtxPopCurrent_v2(cu.ffi.new("CUcontext *"))

    def _release(self):
        if self.handle is not None:
            if self._own_handle:
                self._lib.cuCtxDestroy_v2(self.handle)
            self._handle = None
            Context.context_count -= 1

    def __del__(self):
        self._del_ref(self)


class Device(CU):
    """Holds device id and it's info.
    """
    def __init__(self, index):
        """Will get the device by index here.

        Parameters:
            index: device index or it's pci id.
        """
        super(Device, self).__init__()
        dev = cu.ffi.new("CUdevice *")
        try:
            nme = "cuDeviceGet"
            index = int(index)
            err = self._lib.cuDeviceGet(dev, index)
        except ValueError:
            nme = "cuDeviceGetByPCIBusId"
            err = self._lib.cuDeviceGetByPCIBusId(
                dev, cu.ffi.new("char[]", str(index).encode("utf-8")))
        if err:
            raise CU.error(nme, err)
        self._handle = int(dev[0])

    def create_context(self, flags=0):
        """Creates the context with the current Device.

        Parameters:
            flags: flags for the context (0 - use Context.default_flags).
        """
        return Context(self, flags)

    @property
    def name(self):
        name = cu.ffi.new("char[]", 1024)
        err = self._lib.cuDeviceGetName(name, cu.ffi.sizeof(name), self.handle)
        if err:
            raise CU.error("cuDeviceGetName", err)
        return cu.ffi.string(name).decode("utf-8")

    @property
    def total_mem(self):
        n = cu.ffi.new("size_t *")
        err = self._lib.cuDeviceTotalMem_v2(n, self.handle)
        if err:
            raise CU.error("cuDeviceTotalMem_v2", err)
        return int(n[0])

    @property
    def compute_capability(self):
        """Returns tuple (Major, Minor).
        """
        return (
            self._get_attr(cu.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
            self._get_attr(cu.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR))

    @property
    def pci_bus_id(self):
        pci_id = cu.ffi.new("char[]", 32)
        err = self._lib.cuDeviceGetPCIBusId(pci_id, cu.ffi.sizeof(pci_id),
                                            self.handle)
        if err:
            raise CU.error("cuDeviceGetPCIBusId", err)
        return cu.ffi.string(pci_id).decode("utf-8")

    @property
    def unified_addressing(self):
        return bool(self._get_attr(cu.CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING))

    @property
    def warp_size(self):
        return self._get_attr(cu.CU_DEVICE_ATTRIBUTE_WARP_SIZE)

    @property
    def max_threads_per_block(self):
        return self._get_attr(cu.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)

    @property
    def max_block_dims(self):
        return (self._get_attr(cu.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
                self._get_attr(cu.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
                self._get_attr(cu.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))

    @property
    def max_grid_dims(self):
        return (self._get_attr(cu.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X),
                self._get_attr(cu.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y),
                self._get_attr(cu.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z))

    @property
    def max_shared_memory_per_block(self):
        return self._get_attr(
            cu.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)

    @property
    def max_registers_per_block(self):
        return self._get_attr(cu.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)

    @property
    def total_constant_memory(self):
        return self._get_attr(cu.CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY)

    @property
    def multiprocessor_count(self):
        return self._get_attr(cu.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)

    @property
    def kernel_exec_timeout(self):
        return self._get_attr(cu.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT)

    @property
    def integrated(self):
        return bool(self._get_attr(cu.CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT))

    @property
    def can_map_host_memory(self):
        return bool(self._get_attr(cu.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY))

    @property
    def concurrent_kernels(self):
        return bool(self._get_attr(cu.CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS))

    @property
    def ecc_enabled(self):
        return bool(self._get_attr(cu.CU_DEVICE_ATTRIBUTE_ECC_ENABLED))

    @property
    def memory_bus_width(self):
        return self._get_attr(cu.CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)

    @property
    def l2_cache_size(self):
        return self._get_attr(cu.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)

    @property
    def max_threads_per_multiprocessor(self):
        return self._get_attr(
            cu.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)

    @property
    def async_engine_count(self):
        return self._get_attr(cu.CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT)

    @property
    def stream_priorities_supported(self):
        return bool(self._get_attr(
            cu.CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED))

    @property
    def global_l1_cache_supported(self):
        return bool(self._get_attr(
            cu.CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED))

    @property
    def local_l1_cache_supported(self):
        return bool(self._get_attr(
            cu.CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED))

    @property
    def max_shared_memory_per_multiprocessor(self):
        return self._get_attr(
            cu.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)

    @property
    def max_registers_per_multiprocessor(self):
        return self._get_attr(
            cu.CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)

    @property
    def managed_memory(self):
        return bool(self._get_attr(cu.CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY))

    @property
    def multi_gpu_board(self):
        return bool(self._get_attr(cu.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD))

    @property
    def multi_gpu_board_group_id(self):
        return self._get_attr(cu.CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID)

    @property
    def max_pitch(self):
        return self._get_attr(cu.CU_DEVICE_ATTRIBUTE_MAX_PITCH)

    @property
    def clock_rate(self):
        """Clock rate in kHz.
        """
        return self._get_attr(cu.CU_DEVICE_ATTRIBUTE_CLOCK_RATE)

    @property
    def memory_clock_rate(self):
        """Memory clock rate in kHz.
        """
        return self._get_attr(cu.CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)

    def _get_attr(self, attr):
        n = cu.ffi.new("int *")
        err = self._lib.cuDeviceGetAttribute(n, attr, self.handle)
        if err:
            CU.raise_error("cuDeviceGetAttribute", err)
        return int(n[0])


class Devices(CU):
    """Holds a list of available CUDA devices.
    """
    def __init__(self):
        cu.initialize()
        super(Devices, self).__init__()
        err = self._lib.cuInit(0)
        if err:
            raise CU.error("cuInit", err)
        n = cu.ffi.new("int *")
        err = self._lib.cuDeviceGetCount(n)
        if err:
            raise CU.error("cuDeviceGetCount", err)
        self._devices = list(Device(index)
                             for index in range(int(n[0])))

    def __len__(self):
        return len(self._devices)

    def __getitem__(self, index):
        return self._devices[index]

    def __iter__(self):
        return iter(self._devices)

    @property
    def devices(self):
        return self._devices

    def dump_devices(self):
        """Returns string with information about CUDA devices.
        """
        if not len(self.devices):
            return "No CUDA devices available."
        lines = []
        for i, device in enumerate(self.devices):
            lines.append(
                "\t%d: %s (%d Mb, compute_%d%d, pci %s)" %
                ((i, device.name, device.total_mem // 1048576) +
                 device.compute_capability + (device.pci_bus_id,)))
        return "\n".join(lines)

    def create_some_context(self):
        """Returns Context object with some CUDA device attached.

        If environment variable CUDA_DEVICE is set and not empty,
        gets context based on it, format is:
        <device number>

        If CUDA_DEVICE is not set and os.isatty(0) == True, then
        displays available devices and reads line from stdin in the same
        format as CUDA_DEVICE.

        Else chooses first device.
        """
        if len(self.devices) < 1:
            raise ValueError("No CUDA devices available")
        if len(self.devices) == 1:
            return self.devices[0].create_context()
        ctx = os.environ.get("CUDA_DEVICE")
        if ctx is None or not len(ctx):
            if os.isatty(0):
                sys.stdout.write(
                    "\nEnter "
                    "<device number> or "
                    "set CUDA_DEVICE environment variable.\n"
                    "\nCUDA devices available:\n\n%s\n\n" %
                    (self.dump_devices()))
                sys.stdout.flush()
                ctx = sys.stdin.readline().strip()
            else:
                ctx = "0"
        try:
            device_number = int(ctx)
        except ValueError:
            raise ValueError("Incorrect device number")
        try:
            device = self.devices[device_number]
        except IndexError:
            raise IndexError("Device number is out of range")
        return device.create_context()
