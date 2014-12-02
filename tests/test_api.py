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
        self.assertTrue(type(d.handle) == int)
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
        self.assertEqual(h0, cu.NULL)
        logging.debug("push/pop succeeded")

        h, h0 = self._run_on_thread(self._check_with, (ctx,))
        self.assertEqual(h, ctx.handle)
        self.assertEqual(h0, cu.NULL)
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
            self.assertTrue(type(ptr) == int)
            self.assertEqual(size, 4)
        logging.debug("Succeeded")
        logging.debug("EXIT: test_module")

    def test_mem_alloc(self):
        logging.debug("ENTER: test_mem_alloc")
        ctx = cu.Devices().create_some_context()
        mem = cu.MemAlloc(ctx, 4096)
        self.assertTrue(type(mem.handle) == int)
        self.assertEqual(mem.size, 4096)
        self.assertIsNotNone(mem.handle)
        logging.debug("MemAlloc succeeded")
        logging.debug("EXIT: test_mem_alloc")

    def test_mem_alloc_managed(self):
        logging.debug("ENTER: test_mem_alloc_managed")
        ctx = cu.Devices().create_some_context()
        mem = cu.MemAllocManaged(ctx, 4096)
        self.assertTrue(type(mem.handle) == int)
        self.assertEqual(mem.size, 4096)
        self.assertIsNotNone(mem.handle)
        logging.debug("MemAllocManaged succeeded")
        logging.debug("EXIT: test_mem_alloc_managed")

    def test_mem_host_alloc(self):
        logging.debug("ENTER: test_mem_host_alloc")
        ctx = cu.Devices().create_some_context()
        mem = cu.MemHostAlloc(ctx, 4096)
        self.assertTrue(type(mem.handle) == int)
        self.assertEqual(mem.size, 4096)
        self.assertIsNotNone(mem.handle)
        devptr = mem.device_pointer
        self.assertTrue(type(devptr) == int)
        if ctx.device.unified_addressing:
            self.assertEqual(devptr, mem.handle)
        self.assertIsNotNone(mem.buffer)
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


if __name__ == "__main__":
    import os
    os.environ["CUDA_DEVICE"] = "3"
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
