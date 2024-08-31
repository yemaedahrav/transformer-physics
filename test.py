import torch

A = torch.zeros((65, 2))
A[:,1] = 1
print(A)
B = A.view(-1)
print(B)