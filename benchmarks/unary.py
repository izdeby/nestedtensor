import torch
import nestedtensor
import utils

import random

# Performance tanks hard for lots of small Tensors as expected
RAND_INTS = [random.randint(10, 30) for _ in range(2000)]
RAND_INTS = [random.randint(100, 300) for _ in range(20)]


def gen_t_cos():
    tensor = torch.cat([torch.rand(i, 2560).reshape(-1) for i in RAND_INTS])
    tensor = tensor.cuda()

    def t():
        tensor.cos_()
    return t


def gen_t_loop_cos():
    tensors = [torch.rand(i, 2560).cuda() for i in RAND_INTS]

    def t_loop():
        for t in tensors:
            t.cos_()
    return t_loop


def gen_nt_cos():
    nested_tensor = nestedtensor.nested_tensor(
        [torch.rand(i, 2560).cuda() for i in RAND_INTS])

    def nt():
        nested_tensor.cos_()
    return nt


if __name__ == "__main__":
    print(utils.benchmark_fn(gen_t_cos()))
    print(utils.benchmark_fn(gen_t_loop_cos()))
    print(utils.benchmark_fn(gen_nt_cos()))
