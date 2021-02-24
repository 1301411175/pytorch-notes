## TENSORS ——张量
> 摘自 
> https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html
> (内容为手译)

Tensors 是一种与数组、矩阵非常相似的数据结构。在 Pytorch中，我们用tensors来编码模型的输入、输出以及各参数。
Tensors 与 NumPy 中的 ndarrays 十分相似，不同的是 tensors 可以运行在 GPU 或其他可以加速计算的硬件上。如果你对 ndarrays 熟悉的话，那你也可以轻松驾驭 Tensor API.
但要是你对 ndarrays 不是很熟悉的话，请跟着下面的操作演练一遍。
```python
import torch
import numpy as np
```
---

#### Tensor Initialization —— Tensor 的初始化
Tensors 可以用多种方式进行初始化。请看下面的例子:
- **Directly from data  直接从数据加载**
  ```python
  data = [[1, 2], [3, 4]]
  x_data = torch.tensor(data)
  ```
- **From a NumPy array 从NumPy的array中加载**
  ```python
  np_array = np.array(data)
  x_np = torch.from_numpy(np_array)
  ```
- **Form another tensor 从其他tensor中加载**
  除非明确说明覆盖，否则新的tensor将保留参数tensor的属性(维度形状shape、数据类型datatype)
  ```python
  x_ones = torch.ones_like(x_data) # retains the properties of x_data
  print(f"Ones Tensor: \n {x_ones} \n")

  x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
  print(f"Random Tensor: \n {x_rand} \n")
  ```
  输出:
  ```python
  Ones Tensor:
  tensor([[1, 1],
        [1, 1]])

  Random Tensor:
  tensor([[0.0018, 0.3099],
        [0.3142, 0.0861]])
  ```
- **With random or constant values 用随机值或常量初始化**
  shape 是 tensor 保存维度的元组，在下面的函数中，它规定了输出 tensor 的维度。
  ```python
  shape = (2,3,)
  rand_tensor = torch.rand(shape)
  ones_tensor = torch.ones(shape)
  zeros_tensor = torch.zeros(shape)

  print(f"Random Tensor: \n {rand_tensor} \n")
  print(f"Ones Tensor: \n {ones_tensor} \n")
  print(f"Zeros Tensor: \n {zeros_tensor}")
  ```
  *输出：*
  ```python
  Random Tensor:
  tensor([[0.8766, 0.9551, 0.8060],
        [0.7050, 0.4493, 0.6373]])

  Ones Tensor:
  tensor([[1., 1., 1.],
        [1., 1., 1.]])

  Zeros Tensor:
  tensor([[0., 0., 0.],
        [0., 0., 0.]])
  ```

---
#### Tensor Attributes —— Tensor 的属性
Tensor 的属性描述了其维度形状shape、数据类型datatype以及存储tensor的设备。
```python
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```
*输出：*
```
Shape of tensor: torch.Size([3, 4])
Datatype of tensor: torch.float32
Device tensor is stored on: cpu
```

---
#### Tensor Operations —— tensor的操作
超过 100 种tensor的操作，包括：转置transposing, indexing, 切片slicing, 数学操作mathematical operations, 线性代数linear algebra, 随机取样random sampling  等等都完全地说明在[这个网页](https://pytorch.org/docs/stable/torch.html)。
所有的这些操作都可以运行在GPU上(通常比CPU的速度更快)。如果你使用的是Colab，可以通过    Edit > Notebook Settings 的方法来分配使用GPU。( *为防止翻译错误，我把原文贴上来了：*   If you’re using Colab, allocate a GPU by going to Edit > Notebook Settings.)
```python
# 在GPU条件允许的情况下，我们把tensor移动到GPU上
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
```
试用下面列出的操作。要是你熟悉 NumPy API 操作的话，你会觉得这些都是小菜一碟。
- **Standard numpy-like indexing and slicing 标准的 numpy 风格的索引和切片操作**
  ```python
  tensor = torch.ones(4, 4)
  tensor[:,1] = 0
  print(tensor)
  ```
  *输出：*
  ```python
  tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
  ```
- **Joining tensors 连接tensors**
  你可以使用 ```torch.cat``` 从指定维度连接一组 tensors。```torch.stack```是一个与 ```torch.cat``` 有细微区别但也可以连接 tensors 的函数。
  ```python
  t1 = torch.cat([tensor, tensor, tensor], dim=1)
  print(t1)
  ```
  *输出：*
  ```python
  tensor([[1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.],
        [1., 0., 1., 1., 1., 0., 1., 1., 1., 0., 1., 1.]])
  ```
- **Multiplying tensors 张量(tensor)相乘**
  按各元素相乘
  ```python
  # 两个 tensor 按各元素相乘
  print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
  # 另一可替代语法:
  print(f"tensor * tensor \n {tensor * tensor}")
  ```
  *输出：*
  ```python
  tensor.mul(tensor)
  tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

  tensor * tensor
  tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])
  ```
  按矩阵相乘
  ```python
  # 按矩阵相乘
  print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
  # 另一可替代语法:
  print(f"tensor @ tensor.T \n {tensor @ tensor.T}")
  ```
  *输出：*
  ```python
  tensor.matmul(tensor.T)
  tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])

  tensor @ tensor.T
  tensor([[3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.],
        [3., 3., 3., 3.]])
  ```

- **In-place operations  就地操作**
  操作符后面加一个 ```_``` 后缀就是 *in-place就地操作*。
  ```python
  print(tensor, "\n")
  tensor.add_(5)
  print(tensor)
  ```
  *输出：*
  ```python
  tensor([[1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.],
        [1., 0., 1., 1.]])

  tensor([[6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.],
        [6., 5., 6., 6.]])
  ```
  > in-place operations (就地操作)可能会节约一些内存，但是在计算导数时可能会因为立即丢失历史记录而出现问题。因此**谨慎使用**。

---
#### Bridge with NumPy —— 与Numpy的联系
在CPU里的 tensor 可以和 Numpy array 共享底层的内存分配，如果改变一个那另一个也会随之改变。
- **Tensor to NumPy array  从tensor到NumPy array**
  ```python
  t = torch.ones(5)
  print(f"t: {t}")
  n = t.numpy()
  print(f"n: {n}")
  ```
  *输出：*
  ```
  t: tensor([1., 1., 1., 1., 1.])
  n: [1. 1. 1. 1. 1.]
  ```
  在 tensor 中的改变会反映到 Numpy array中。
  ```python
  t.add_(1)
  print(f"t: {t}")
  print(f"n: {n}")
  ```
  *输出：*
  ```python
  t: tensor([2., 2., 2., 2., 2.])
  n: [2. 2. 2. 2. 2.]
  ```
- **Numpy array to Tensor 从Numpy array到 tensor**
  ```python
  n = np.ones(5)
  t = torch.from_numpy(n)
  ```
  在 NumPy array 中的改变会反映到 tensor 中。
  ```python
  np.add(n, 1, out=n)
  print(f"t: {t}")
  print(f"n: {n}")
  ```
  *输出：*
  ```
  t: tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
  n: [2. 2. 2. 2. 2.]
  ```



  

