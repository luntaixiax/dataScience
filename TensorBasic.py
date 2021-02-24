import torch

#======================================================#
#                   Initializing Tensor
#======================================================#

device = "cuda" if torch.cuda.is_available() else "cpu"

my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, device=device, requires_grad=True)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.requires_grad)
print(my_tensor.data)
print(my_tensor.shape)

x = torch.empty(size=(3, 3))
y = torch.zeros((3, 3))
z = torch.rand((3, 3),) # normal distribution
a = torch.eye(5, 5)  # diagonal matrix
print(a)
b = torch.arange(start=0, end=5, step=1)
print(b)
c = torch.linspace(start=0.1, end=1, steps=10)
print(c)
d = torch.empty(size=(1, 5)).normal_(mean=5, std=2)
print(d)
e = torch.diag(torch.ones(3))
print(e)


#======================================================#
#                  Convert Tensor types
#======================================================#

tensor = torch.arange(4)
print(tensor.bool()) # convert to boolean
print(tensor.short())  # convert to short type
print(tensor.long())  # convert to int64 type
print(tensor.half())  # convert to float16
print(tensor.float())  # convert to float32
print(tensor.double())  # convert to float64

# convert from numpy to tensor
import numpy as np

np_arr = np.zeros((5, 5))
tensor2 = torch.from_numpy(np_arr)
np_arr_back = tensor2.numpy()


#======================================================#
#                  Tensor Math and operations
#======================================================#

x1 = torch.tensor([1, 2, 3])
x2 = torch.tensor([9, 8, 7])

# Addition/Subtraction
y1 = torch.add(x1, x2)
y2 = torch.sub(x1, x2)
print(y1, y2)

# Division
y3 = torch.true_divide(x1, x2)
print(y3)

# exponentiation
y4 = x1.pow(2)

#======================================================#
#                  Inplace Operation (with _ at end)
#======================================================#

x1 = torch.tensor([1, 2, 3])
x2 = torch.tensor([9, 8, 7])

x2.add_(x1)  # inplace addition
print(x2)


#======================================================#
#                  Matirx Operation
#======================================================#

m1 = torch.rand((2, 5))
m2 = torch.rand((5, 3))

z1 = torch.mm(m1, m2)
z2 = m1.mm(m2)
print(z1)

# transpose
zt = z1.t()
print(zt)

# matrix exponentiation
m3 = torch.rand((5, 5))
mat_exp = m3.matrix_power(3)
print(mat_exp)

# element wise mult
v1 = torch.rand(3)
v2 = torch.rand(3)

z3 = torch.dot(v1, v2)
z4 = v1 * v2
print(z3, z4)

# batch matirx mult
batch = 32
n = 10
m = 20
p = 30

t1 = torch.rand((batch, n, m))
t2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(t1, t2)  # (batch, n, p)
print(out_bmm)

# example of braodcasting
b1 = torch.rand((5, 5))
b2 = torch.rand((1, 5))

bb = b1 - b2


#======================================================#
#                  Useful Operation
#======================================================#
m1 = torch.rand((2, 5))

sum_x1 = torch.sum(m1, dim = 0)
sum_x2 = torch.sum(m1, dim = 1)
print(sum_x1, sum_x2)

sorted_m, indices = torch.sort(m1, dim=0, descending=False)
print(sorted_m, indices)

relu_m = torch.clamp(m1, min=0)


#======================================================#
#                  Indexing
#======================================================#

i1 = torch.randn(5)
i2 = i1[[0, 3, 4]]
print(i1)
print(i2)

i3 = torch.randn((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
print(i3)
print(i3[rows, cols])  # take i3[1, 4] and i3[0, 0]


#======================================================#
#                  Reshape
#======================================================#

r1 = torch.arange(9)
r2 = r1.view(3, 3)  # performance better
r3 = r1.reshape(3, 3)  # safer (work under all scenario)

r4 = r2.t().contiguous().view(9)  # .contiguous to ensure safety, otherwise runtime error
r5 = r2.t().reshape(9)  # reshape is safer
print(r4)

# flat
r6 = r3.view(-1)
print(r6)

batch = 64
h = torch.rand((batch, 2, 5))
print(h.view(batch, -1).shape)
print(h.permute(0, 2, 1).shape)  # transpose

s1 = torch.arange(10)
print(s1.unsqueeze(0).shape)  # unsqueeze dim 0  torch.Size([1, 10])
print(s1.unsqueeze(1).shape)  # unsqueeze dim 1  torch.Size([10, 1])

s2 = torch.arange(10).unsqueeze(0).unsqueeze(1)  # 1x1x10
print(s2.squeeze(0).shape)  # torch.Size([1, 10])
print(s2.squeeze(1).shape)  # torch.Size([1, 10])

c1 = torch.rand((2, 5))
c2 = torch.rand((2, 5))
print(torch.cat((c1, c2), dim = 0))
print(torch.cat((c1, c2), dim = 1))


