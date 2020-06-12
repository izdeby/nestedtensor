import torch
import nestedtensor
import utils

import random

# Performance tanks hard for lots of small Tensors as expected
RAND_INTS = [random.randint(10, 30) for _ in range(2000)]
RAND_INTS = [random.randint(100, 300) for _ in range(20)]


def gen_t_mul():
    tensor = torch.cat([torch.rand(i, 2560).reshape(-1) for i in RAND_INTS])
    tensor1 = tensor.cuda()
    tensor2 = tensor.cuda().clone()

    def t():
        tensor1.mul(tensor2)
    return t


def gen_t_loop_mul():
    tensors1 = [torch.rand(i, 2560).cuda() for i in RAND_INTS]
    tensors2 = [torch.rand(i, 2560).cuda() for i in RAND_INTS]

    def t_loop():
        for t1, t2 in zip(tensors1, tensors2):
            t1.mul(t2)
    return t_loop


def gen_nt_mul():
    nested_tensor1 = nestedtensor.nested_tensor(
        [torch.rand(i, 2560).cuda() for i in RAND_INTS])
    nested_tensor2 = nestedtensor.nested_tensor(
        [torch.rand(i, 2560).cuda() for i in RAND_INTS])

    def nt():
        nested_tensor1.mul(nested_tensor2)
    return nt


if __name__ == "__main__":
    print(utils.benchmark_fn(gen_t_mul()))
    print(utils.benchmark_fn(gen_t_loop_mul()))
    print(utils.benchmark_fn(gen_nt_mul()))
