cuda4py
=========

Python cffi CUDA bindings and helper classes.

Tested with Python 2.7, Python 3.4 and PyPy on Linux with CUDA 6.5.

To compile kernel code written in C++, nvcc is required and
exported functions should be marked as extern "C".
Functions in plain PTX can be used without nvcc.

To use CUBLAS, libcublas.so (cublas.dll) should be present
(see tests/test_cublas.py for example).

Not all CUDA api is currently covered.

To install the module run:
```bash
python setup.py install
```
or just copy src/cuda4py to any place where python
interpreter will be able to find it.

To run the tests, execute:

for Python 2.7:
```bash
PYTHONPATH=src nosetests -w tests
```

for Python 3.4:
```bash
PYTHONPATH=src nosetests3 -w tests
```

for PyPy:
```bash
PYTHONPATH=src pypy tests/test_api.py
```

Currently, PyPy numpy support may be incomplete,
so tests which use numpy arrays may fail.

Example usage:

```python
import cuda4py as cu
import logging
import numpy


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    ctx = cu.Devices().create_some_context()
    module = cu.Module(
        ctx, source=
        """
        extern "C"
        __global__ void test(const float *a, const float *b,
                             float *c, const float k) {
          size_t i = blockDim.x * blockIdx.x + threadIdx.x;
          c[i] = (a[i] + b[i]) * k;
        }
        """)
    test = cu.Function(module, "test")
    a = numpy.arange(1000000, dtype=numpy.float32)
    b = numpy.arange(1000000, dtype=numpy.float32)
    c = numpy.empty(1000000, dtype=numpy.float32)
    k = numpy.array([0.5], dtype=numpy.float32)
    a_buf = cu.MemAlloc(ctx, a.nbytes)
    b_buf = cu.MemAlloc(ctx, b.nbytes)
    c_buf = cu.MemAlloc(ctx, c.nbytes)
    a_buf.to_device_async(a)
    b_buf.to_device_async(b)
    test.set_args(a_buf, b_buf, c_buf, k)
    test((a.size, 1, 1))
    c_buf.to_host(c)
    max_diff = numpy.fabs(c - (a + b) * k[0]).max()
    logging.info("max_diff = %.6f", max_diff)
```

Released under Simplified BSD License.
Copyright (c) 2014, Samsung Electronics Co.,Ltd.
