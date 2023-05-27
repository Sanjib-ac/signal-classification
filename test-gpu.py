from timeit import default_timer as timer

# To run on CPU
import GPUtil
import numpy as np
import torch
from numba import jit, cuda

# import cudf
print(cuda.list_devices())
# import torch
print(GPUtil.getAvailable())


def func(a):
    for i in range(10000000):
        a[i] += 1


# To run on GPU
@jit
def func2(x):
    return x + 1


if __name__ == "__main__":
    n = 10000000
    a = np.ones(n, dtype=np.float64)
    start = timer()
    func(a)
    print("without GPU:", timer() - start)
    start = timer()
    func2(a)
    # numba.cuda.profile_stop()
    print("with GPU:", timer() - start)




# # x = torch.rand(5, 3)
# # print(x)
# n_real = np.array([[-0.96831675  -0.85321414  -0.71319101]])
# print(n_real)
# print(np.argmax(n_real))
# print(n_real[np.argmax(n_real)])


# print("Is cuda available?", torch.cuda.is_available())

# print(11.193214655546535 * 2.4e3)
# print(-12.27575768182757 * 2.4e3)
#
# print((11.193214655546535* 2.4e3)/ 0.000716238054670067)