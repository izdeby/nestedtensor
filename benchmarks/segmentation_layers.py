import torch
import nestedtensor
import utils

def relu_tensor():
    inputs = [
            torch.randn(3, 500, 600),
            torch.randn(3, 500, 600),
            torch.randn(3, 128, 128),
            torch.randn(3, 500, 600),
            torch.randn(3, 500, 600),
            torch.randn(3, 128, 128)
        ]

    def _relu_tensor():
        tensor_res = []
        for i in range(2):
                t_res = torch.nn.functional.relu(inputs[i].unsqueeze(0))
                tensor_res.append(t_res.squeeze(0))
    return _relu_tensor

def relu_nt_contiguous():
    inputs = [
            torch.randn(3, 500, 600),
            torch.randn(3, 500, 600),
            torch.randn(3, 128, 128),
            torch.randn(3, 500, 600),
            torch.randn(3, 500, 600),
            torch.randn(3, 128, 128)
        ]

    def _relu_nt_contiguous():
        nt = nestedtensor.nested_tensor(inputs)
        nt_res = torch.nn.functional.relu(nt)
    
    return _relu_nt_contiguous

if __name__ == "__main__":
    print(utils.benchmark_fn(relu_tensor()))
    print(utils.benchmark_fn(relu_nt_contiguous()))