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
Tests some of the api in cuda4py package.
"""
import gc
import logging
import cuda4py as cu
try:
    import numpy
except ImportError:
    pass
import os
import threading
import unittest


class Test(unittest.TestCase):
    def setUp(self):
        self.old_env = os.environ.get("CUDA_DEVICE")
        if self.old_env is None:
            os.environ["CUDA_DEVICE"] = "0"
        self.path = os.path.dirname(__file__)
        if not len(self.path):
            self.path = "."

    def tearDown(self):
        if self.old_env is None:
            del os.environ["CUDA_DEVICE"]
        else:
            os.environ["CUDA_DEVICE"] = self.old_env
        del self.old_env
        gc.collect()

    def test_devices(self):
        logging.debug("ENTER: test_devices")
        devices = cu.Devices()
        logging.debug("Found %d CUDA device%s", len(devices),
                      "" if len(devices) <= 1 else "s")
        for i, device in enumerate(devices):
            logging.debug("%d: %s", i, device.name)
        if not len(devices):
            return
        logging.debug("Selecting device 0")
        d = devices[0]
        self.assertEqual(d.handle, int(d.handle))
        logging.debug("It's name is %s", d.name)
        logging.debug("It's total mem is %d", d.total_mem)
        logging.debug("It's compute capability is %d_%d",
                      *d.compute_capability)
        logging.debug("It's pci bus id: %s", d.pci_bus_id)
        logging.debug("Trying to get device by it's pci id")
        d2 = cu.Device(d.pci_bus_id)
        self.assertEqual(d2.handle, d.handle)
        logging.debug("Succeeded")
        logging.debug("EXIT: test_devices")

    def test_attributes(self):
        d = cu.Devices()[0]
        self.assertIsInstance(d.unified_addressing, bool)
        self.assertGreater(d.warp_size, 0)
        self.assertGreater(d.max_threads_per_block, 0)
        self.assertGreaterEqual(d.max_shared_memory_per_block, 0)
        xyz = d.max_block_dims
        self.assertIsInstance(xyz, tuple)
        self.assertEqual(len(xyz), 3)
        for x in xyz:
            self.assertGreater(x, 0)
        xyz = d.max_grid_dims
        self.assertIsInstance(xyz, tuple)
        self.assertEqual(len(xyz), 3)
        for x in xyz:
            self.assertGreater(x, 0)
        self.assertGreater(d.max_registers_per_block, 0)
        self.assertGreater(d.clock_rate, 0)
        self.assertGreater(d.memory_clock_rate, 0)
        self.assertGreaterEqual(d.total_constant_memory, 0)
        self.assertGreater(d.multiprocessor_count, 0)
        self.assertGreaterEqual(d.kernel_exec_timeout, 0)
        self.assertIsInstance(d.integrated, bool)
        self.assertIsInstance(d.can_map_host_memory, bool)
        self.assertIsInstance(d.concurrent_kernels, bool)
        self.assertIsInstance(d.ecc_enabled, bool)
        self.assertGreater(d.memory_bus_width, 0)
        self.assertGreaterEqual(d.l2_cache_size, 0)
        self.assertGreater(d.max_threads_per_multiprocessor, 0)
        self.assertGreaterEqual(d.async_engine_count, 0)
        self.assertIsInstance(d.stream_priorities_supported, bool)
        self.assertIsInstance(d.global_l1_cache_supported, bool)
        self.assertIsInstance(d.local_l1_cache_supported, bool)
        self.assertGreaterEqual(d.max_shared_memory_per_multiprocessor, 0)
        self.assertGreater(d.max_registers_per_multiprocessor, 0)
        self.assertIsInstance(d.managed_memory, bool)
        self.assertIsInstance(d.multi_gpu_board, bool)
        self.assertGreaterEqual(d.multi_gpu_board_group_id, 0)
        self.assertGreaterEqual(d.max_pitch, 0)

    def test_extract_ptr(self):
        a = numpy.zeros(127, dtype=numpy.float32)
        ptr = cu.CU.extract_ptr(a)
        self.assertEqual(ptr, int(a.__array_interface__["data"][0]))
        ptr2, sz = cu.CU.extract_ptr_and_size(a, None)
        self.assertEqual(ptr, ptr2)
        self.assertEqual(sz, a.nbytes)
        ptr = cu.CU.extract_ptr(None)
        self.assertEqual(ptr, 0)
        ptr2, sz = cu.CU.extract_ptr_and_size(None, 0)
        self.assertEqual(ptr2, 0)
        self.assertEqual(sz, 0)

    def test_dump_devices(self):
        logging.debug("ENTER: test_dump_devices")
        logging.debug("Available CUDA devices:\n%s",
                      cu.Devices().dump_devices())
        logging.debug("EXIT: test_dump_devices")

    def _fill_retval(self, retval, target, args):
        retval[0], retval[1] = target(*args)

    def _run_on_thread(self, target, args):
        retval = [None, None, None]
        thread = threading.Thread(target=self._fill_retval,
                                  args=(retval, target, args))
        thread.start()
        thread.join(5)
        if thread.is_alive():
            raise TimeoutError()
        return retval[0], retval[1]

    def _check_push_pop(self, ctx):
        ctx.push_current()
        h = cu.Context.get_current()
        ctx.pop_current()
        h0 = cu.Context.get_current()
        return h, h0

    def _check_with(self, ctx):
        with ctx:
            logging.debug("Inside with statement")
            h = cu.Context.get_current()
        h0 = cu.Context.get_current()
        return h, h0

    def _check_set_current(self, ctx):
        ctx.set_current()
        return cu.Context.get_current(), None

    def test_context(self):
        logging.debug("ENTER: test_context")
        ctx = cu.Devices().create_some_context()
        logging.debug("Context created")
        self.assertEqual(ctx.handle, cu.Context.get_current())

        h, h0 = self._run_on_thread(self._check_push_pop, (ctx,))
        self.assertEqual(h, ctx.handle)
        self.assertEqual(h0, 0)
        logging.debug("push/pop succeeded")

        h, h0 = self._run_on_thread(self._check_with, (ctx,))
        self.assertEqual(h, ctx.handle)
        self.assertEqual(h0, 0)
        logging.debug("with succeeded")

        self.assertEqual(
            self._run_on_thread(self._check_set_current, (ctx,))[0],
            ctx.handle)
        logging.debug("set_current succeeded")
        logging.debug("EXIT: test_context")

    def test_module(self):
        logging.debug("ENTER: test_module")
        ctx = cu.Devices().create_some_context()
        module = cu.Module(ctx, source_file="%s/test.cu" % self.path)
        self.assertIsNotNone(module.handle)
        self.assertIsNotNone(ctx.handle)
        logging.debug("nvcc compilation succeeded")
        logging.debug("Resulted ptx code is:\n%s", module.ptx.decode("utf-8"))

        logging.debug("Will try Context.create_module")
        module = ctx.create_module(source_file="%s/test.cu" % self.path)
        self.assertIsNotNone(module.handle)
        self.assertIsNotNone(ctx.handle)
        logging.debug("Succeeded")

        logging.debug("Will try to compile with includes")
        module = cu.Module(ctx, source_file="%s/inc.cu" % self.path,
                           include_dirs=("", self.path, ""))
        self.assertIsNotNone(module.handle)
        self.assertIsNotNone(ctx.handle)
        logging.debug("Succeeded")
        logging.debug("Will try to compile with source")
        module = cu.Module(ctx, source="#include \"inc.cu\"",
                           include_dirs=(self.path,))
        self.assertIsNotNone(module.handle)
        self.assertIsNotNone(ctx.handle)
        logging.debug("Succeeded")
        logging.debug("Testing get_func, get_global")
        with ctx:
            self.assertIsNotNone(module.get_func("test"))
            ptr, size = module.get_global("g_a")
            self.assertEqual(ptr, int(ptr))
            self.assertEqual(size, 4)
        logging.debug("Succeeded")
        logging.debug("EXIT: test_module")

    def _test_alloc(self, alloc, test=None):
        mem = alloc(4096)
        self.assertEqual(mem.handle, int(mem.handle))
        self.assertEqual(mem.handle, int(mem))
        self.assertEqual(mem.size, 4096)
        self.assertIsNotNone(mem.handle)
        if test is not None:
            test(mem)

        a = numpy.random.rand(4096).astype(numpy.float32)
        mem = alloc(a)
        b = numpy.zeros_like(a)
        mem.to_host(b)
        max_diff = float(numpy.fabs(a - b).max())
        self.assertEqual(max_diff, 0.0)
        if test is not None:
            test(mem)

    def test_mem_alloc(self):
        logging.debug("ENTER: test_mem_alloc")
        ctx = cu.Devices().create_some_context()
        self._test_alloc(lambda a: cu.MemAlloc(ctx, a))
        self._test_alloc(ctx.mem_alloc)
        logging.debug("MemAlloc succeeded")
        logging.debug("EXIT: test_mem_alloc")

    def test_mem_alloc_managed(self):
        logging.debug("ENTER: test_mem_alloc_managed")
        ctx = cu.Devices().create_some_context()
        self._test_alloc(lambda a: cu.MemAllocManaged(ctx, a))
        self._test_alloc(ctx.mem_alloc_managed)
        logging.debug("MemAllocManaged succeeded")
        logging.debug("EXIT: test_mem_alloc_managed")

    def test_mem_host_alloc(self):
        logging.debug("ENTER: test_mem_host_alloc")
        ctx = cu.Devices().create_some_context()

        def test(mem):
            devptr = mem.device_pointer
            self.assertEqual(devptr, int(devptr))
            if ctx.device.unified_addressing:
                self.assertEqual(devptr, mem.handle)
            self.assertIsNotNone(mem.buffer)

        self._test_alloc(lambda a: cu.MemHostAlloc(ctx, a), test)
        self._test_alloc(ctx.mem_host_alloc, test)

        logging.debug("MemHostAlloc succeeded")
        logging.debug("EXIT: test_mem_host_alloc")

    def test_launch_kernel(self):
        logging.debug("ENTER: test_launch_kernel")
        ctx = cu.Devices().create_some_context()
        logging.debug("Context created")
        N = 1024
        C = 0.75
        a = cu.MemHostAlloc(ctx, N * 4)
        b = cu.MemHostAlloc(ctx, N * 4)
        logging.debug("Memory allocated")
        module = cu.Module(ctx, source_file="%s/test.cu" % self.path)
        logging.debug("Program builded")
        f = module.get_func("test")
        logging.debug("Got function pointer")
        f.set_args(a, b, numpy.array([C], dtype=numpy.float32))
        f.set_args(a, cu.skip, numpy.array([C], dtype=numpy.float32))
        f.set_args(cu.skip(2), numpy.array([C], dtype=numpy.float32))
        f.set_args(a, b, cu.skip(1))
        f.set_args(cu.skip(3))
        f.set_arg(0, None)
        f.set_arg(0, a)
        logging.debug("Args set")
        a_host = numpy.random.rand(N).astype(numpy.float32)
        b_host = numpy.random.rand(N).astype(numpy.float32)
        gold = a_host.copy()
        for _ in range(10):
            gold += b_host * C
        a.to_device(a_host)
        b.to_device_async(b_host)
        for _ in range(10):
            f((N, 1, 1))
        logging.debug("Scheduled for execution")
        c_host = numpy.zeros(N, dtype=numpy.float32)
        a.to_host(c_host)
        logging.debug("Got results back")
        max_diff = numpy.fabs(c_host - gold).max()
        self.assertLess(max_diff, 0.0001)
        logging.debug("test_launch_kernel() succeeded")
        logging.debug("EXIT: test_launch_kernel")

    def test_memset(self):
        logging.debug("ENTER: test_memset")
        ctx = cu.Devices().create_some_context()
        mem = cu.MemAlloc(ctx, 4096)
        mem.memset32_async(123)
        mem.memset32_async(456, 1)
        mem.memset32_async(789, 2, 3)
        a = numpy.zeros(mem.size // 4, dtype=numpy.int32)
        mem.to_host(a)
        self.assertEqual(a[0], 123)
        self.assertEqual(a[1], 456)
        for i in range(2, 2 + 3):
            self.assertEqual(a[i], 789)
        for i in range(2 + 3, a.size):
            self.assertEqual(a[i], 456)
        logging.debug("EXIT: test_memset")

    def test_memcpy(self):
        logging.debug("ENTER: test_memcpy")
        ctx = cu.Devices().create_some_context()
        a = cu.MemAlloc(ctx, 4096)
        a.memset32_async(123)
        b = cu.MemAlloc(ctx, 4096)
        b.memset32_async(456)
        test = numpy.zeros(a.size // 4, dtype=numpy.int32)
        a.from_device_async(b)
        a.to_host(test)
        for x in test:
            self.assertEqual(x, 456)
        a.memset32_async(123)
        a.from_device_async(b, 12)
        a.to_host(test)
        for x in test[:3]:
            self.assertEqual(x, 123)
        for x in test[3:]:
            self.assertEqual(x, 456)
        a.memset32_async(123)
        a.from_device_async(b, 12, 64)
        a.to_host(test)
        for x in test[:3]:
            self.assertEqual(x, 123)
        for x in test[3:19]:
            self.assertEqual(x, 456)
        for x in test[19:]:
            self.assertEqual(x, 123)
        logging.debug("EXIT: test_memcpy")

    def test_occupancy(self):
        logging.debug("ENTER: test_occupancy")
        ctx = cu.Devices().create_some_context()
        logging.debug("Context created")
        module = cu.Module(ctx, source_file="%s/test.cu" % self.path)
        logging.debug("Program builded")
        f = module.get_func("test")
        logging.debug("Got function pointer")

        num_blocks = f.max_active_blocks_per_multiprocessor(1)
        self.assertEqual(num_blocks, int(num_blocks))
        self.assertGreater(num_blocks, 0)
        logging.debug("num_blocks = %d", num_blocks)
        logging.debug("Testing dynamic_smem_size parameter")
        num_blocks = f.max_active_blocks_per_multiprocessor(
            128, dynamic_smem_size=8192)
        self.assertEqual(num_blocks, int(num_blocks))
        self.assertGreater(num_blocks, 0)
        logging.debug("num_blocks = %d", num_blocks)

        min_grid_size, block_size = f.max_potential_block_size()
        self.assertEqual(min_grid_size, int(min_grid_size))
        self.assertEqual(block_size, int(block_size))
        self.assertGreater(min_grid_size, 0)
        self.assertGreater(block_size, 0)
        logging.debug("min_grid_size, block_size = %d, %d",
                      min_grid_size, block_size)
        logging.debug("Trying callback")
        min_grid_size, block_size = f.max_potential_block_size(
            lambda x: x ** 2)
        self.assertEqual(min_grid_size, int(min_grid_size))
        self.assertEqual(block_size, int(block_size))
        self.assertGreater(min_grid_size, 0)
        self.assertGreater(block_size, 0)
        logging.debug("min_grid_size, block_size = %d, %d",
                      min_grid_size, block_size)
        logging.debug("Testing block_size_limit parameter")
        min_grid_size, block_size = f.max_potential_block_size(
            block_size_limit=16)
        self.assertEqual(min_grid_size, int(min_grid_size))
        self.assertEqual(block_size, int(block_size))
        self.assertGreater(min_grid_size, 0)
        self.assertGreater(block_size, 0)
        self.assertLessEqual(block_size, 16)
        logging.debug("min_grid_size, block_size = %d, %d",
                      min_grid_size, block_size)
        logging.debug("Testing dynamic_smem_size parameter")
        min_grid_size, block_size = f.max_potential_block_size(
            dynamic_smem_size=8192)
        self.assertEqual(min_grid_size, int(min_grid_size))
        self.assertEqual(block_size, int(block_size))
        self.assertGreater(min_grid_size, 0)
        self.assertGreater(block_size, 0)
        logging.debug("min_grid_size, block_size = %d, %d",
                      min_grid_size, block_size)
        logging.debug("EXIT: test_occupancy")

    def test_memcpy_3d_async(self):
        logging.debug("ENTER: test_memcpy_3d_async")

        p_copy = cu.get_ffi().new("CUDA_MEMCPY3D *")
        self.assertEqual(cu.get_ffi().sizeof(p_copy[0]), 200)

        ctx = cu.Devices().create_some_context()
        logging.debug("Context created")

        # Create arrays with some values for testing
        a = numpy.arange(35 * 25 * 15, dtype=numpy.float32).reshape(35, 25, 15)
        b = numpy.arange(37 * 27 * 17, dtype=numpy.float32).reshape(37, 27, 17)
        b *= 0.5
        c = numpy.empty_like(b)
        c[:] = 1.0e30

        # Create buffers
        a_ = cu.MemAlloc(ctx, a)
        b_ = cu.MemAlloc(ctx, b)

        # Copy 3D rect from one device buffer to another
        logging.debug("Testing device -> device memcpy_3d_async")
        sz = a.itemsize
        a_.memcpy_3d_async(
            (3 * sz, 4, 5), (6 * sz, 7, 8), (5 * sz, 10, 20),
            a.shape[2] * sz, a.shape[1], b.shape[2] * sz, b.shape[1],
            dst=b_)
        b_.to_host(c)
        diff = numpy.fabs(c[8:28, 7:17, 6:11] - a[5:25, 4:14, 3:8]).max()
        self.assertEqual(diff, 0)

        # Copy 3D rect from host buffer to device buffer
        logging.debug("Testing host -> device memcpy_3d_async")
        sz = a.itemsize
        b_.memset32_async()
        b_.memcpy_3d_async(
            (3 * sz, 4, 5), (6 * sz, 7, 8), (5 * sz, 10, 20),
            a.shape[2] * sz, a.shape[1], b.shape[2] * sz, b.shape[1],
            src=a)
        c[:] = 1.0e30
        b_.to_host(c)
        diff = numpy.fabs(c[8:28, 7:17, 6:11] - a[5:25, 4:14, 3:8]).max()
        self.assertEqual(diff, 0)

        # Copy 3D rect from device buffer to host buffer
        logging.debug("Testing device -> host memcpy_3d_async")
        sz = a.itemsize
        c[:] = 1.0e30
        a_.memcpy_3d_async(
            (3 * sz, 4, 5), (6 * sz, 7, 8), (5 * sz, 10, 20),
            a.shape[2] * sz, a.shape[1], b.shape[2] * sz, b.shape[1],
            dst=c)
        ctx.synchronize()
        diff = numpy.fabs(c[8:28, 7:17, 6:11] - a[5:25, 4:14, 3:8]).max()
        self.assertEqual(diff, 0)

        logging.debug("EXIT: test_memcpy_3d_async")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
