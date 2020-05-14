import torch 
import nestedtensor
import time

inputs = []
big = 1


if big == 0:
    for i in range(50):
        input_i = torch.randn(256, 80, 80)
        inputs.append(input_i)

    bn = torch.nn.BatchNorm2d(256, 1e-05, 0.1)
    bn.eval()
else: 
    for i in range(3):
        input_i = torch.randn(256, 800, 800)
        inputs.append(input_i)

    bn = torch.nn.BatchNorm2d(256, 1e-05, 0.1)
    bn.eval()

t3 = time.time()
nt = nestedtensor.as_nested_tensor(inputs)
t3res = time.time() - t3
#print("constructor: ", t3res)

t2 = time.time()
res = bn(nt)
t2_res = time.time() - t2
print("bn time: ", t2_res)
print("\n\n\n")

bn.eval()
t1 = time.time() 
for t in inputs:
    bn(t.unsqueeze(0)).squeeze(0) # <-- why vision is not doing it?!
t1_res = time.time() - t1
print(t1_res)
