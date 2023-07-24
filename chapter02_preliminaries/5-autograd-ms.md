# 自动微分
:label:`sec_autograd`

正如 :numref:`sec_calculus`中所说，求导是几乎所有深度学习优化算法的关键步骤。
虽然求导的计算很简单，只需要一些基本的微积分。
但对于复杂的模型，手工进行更新是一件很痛苦的事情（而且经常容易出错）。

深度学习框架通过自动计算导数，即*自动微分*（automatic differentiation）来加快求导。
实际中，根据设计好的模型，系统会构建一个*计算图*（computational graph），
来跟踪计算是哪些数据通过哪些操作组合起来产生输出。
自动微分使系统能够随后反向传播梯度。
这里，*反向传播*（backpropagate）意味着跟踪整个计算图，填充关于每个参数的偏导数。

## 一个简单的例子

作为一个演示例子，(**假设我们想对函数$y=2\mathbf{x}^{\top}\mathbf{x}$关于列向量$\mathbf{x}$求导**)。
首先，我们创建变量`x`并为其分配一个初始值。



```python
import mindspore as ms
from mindspore import ops

x = ops.arange(4.0)
x
```




    Tensor(shape=[4], dtype=Float32, value= [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00,  3.00000000e+00])



[**在我们计算$y$关于$\mathbf{x}$的梯度之前，需要一个地方来存储梯度。**]
重要的是，我们不会在每次对一个参数求导时都分配新的内存。
因为我们经常会成千上万次地更新相同的参数，每次都分配新的内存可能很快就会将内存耗尽。
注意，一个标量函数关于向量$\mathbf{x}$的梯度是向量，并且与$\mathbf{x}$具有相同的形状。



```python
from mindspore import Parameter
```


```python
x = Parameter(x)
```


```python
x.requires_grad = True  # 等价于x=torch.arange(4.0,requires_grad=True)
# x.grad  # 默认值是None
```

(**现在计算$y$。**)



```python
y = 2 * ops.tensor_dot(x, x, axes=1)
y
```




    Tensor(shape=[], dtype=Float32, value= 28)



`x`是一个长度为4的向量，计算`x`和`x`的点积，得到了我们赋值给`y`的标量输出。
接下来，[**通过调用反向传播函数来自动计算`y`关于`x`每个分量的梯度**]，并打印这些梯度。



```python
from mindspore import grad
```


```python
def func(x):
    y = 2 * ops.tensor_dot(x, x, axes=1)
    return y
grad_fn = ms.grad(func, grad_position=0)
grad_x = grad_fn(x)
print(grad_x)
```

    [ 0.  4.  8. 12.]


函数$y=2\mathbf{x}^{\top}\mathbf{x}$关于$\mathbf{x}$的梯度应为$4\mathbf{x}$。
让我们快速验证这个梯度是否计算正确。



```python
grad_x == 4 * x
```




    Tensor(shape=[4], dtype=Bool, value= [ True,  True,  True,  True])



[**现在计算`x`的另一个函数。**]



```python
# 在默认情况下，MindSpore不像PyTorch会累积梯度，我们不需要清除之前的值。

def func(x):
    y = x.sum()
    return y
grad_fn = ms.grad(func, grad_position=0)
grad_x = grad_fn(x)
print(grad_x)
```

    [1. 1. 1. 1.]


## 非标量变量的反向传播

当`y`不是标量时，向量`y`关于向量`x`的导数的最自然解释是一个矩阵。
对于高阶和高维的`y`和`x`，求导的结果可以是一个高阶张量。

然而，虽然这些更奇特的对象确实出现在高级机器学习中（包括[**深度学习中**]），
但当调用向量的反向计算时，我们通常会试图计算一批训练样本中每个组成部分的损失函数的导数。
这里(**，我们的目的不是计算微分矩阵，而是单独计算批量中每个样本的偏导数之和。**)



```python
# 在默认情况下，MindSpore不像PyTorch会累积梯度，我们不需要清除之前的值。
def func(x):
    y = x * x
    y = y.sum()
    return y
grad_fn = ms.grad(func, grad_position=0)
grad_x = grad_fn(x)
print(grad_x)
```

    [0. 2. 4. 6.]


## 分离计算

有时，我们希望[**将某些计算移动到记录的计算图之外**]。
例如，假设`y`是作为`x`的函数计算的，而`z`则是作为`y`和`x`的函数计算的。
想象一下，我们想计算`z`关于`x`的梯度，但由于某种原因，希望将`y`视为一个常数，
并且只考虑到`x`在`y`被计算后发挥的作用。

这里可以分离`y`来返回一个新变量`u`，该变量与`y`具有相同的值，
但丢弃计算图中如何计算`y`的任何信息。
换句话说，梯度不会向后流经`u`到`x`。
因此，下面的反向传播函数计算`z=u*x`关于`x`的偏导数，同时将`u`作为常数处理，
而不是`z=x*x*x`关于`x`的偏导数。


请观察如下的求解过程


```python
from copy import deepcopy
```


```python
y = Parameter(x * x)
u = deepcopy(y)
u.requires_grad = False
def func(x):
    z = u * x
    z = z.sum()
    return z
grad_fn = ms.grad(func, grad_position=0)
grad_x = grad_fn(x)
grad_x == u
```




    Tensor(shape=[4], dtype=Bool, value= [ True,  True,  True,  True])




```python
def func(x):
    y = Parameter(x * x)
    u = deepcopy(y)
    u.requires_grad = False
    z = u * x
    z = z.sum()
    return z
grad_fn = ms.grad(func, grad_position=0)
grad_x = grad_fn(x)
grad_x == u
```




    Tensor(shape=[4], dtype=Bool, value= [ True,  True,  True,  True])




```python
def func(x):
    y = Parameter(x * x)
    u = deepcopy(y)
    u.requires_grad = True
    # u.requires_grad = False
    z = u * x
    z = z.sum()
    return z
grad_fn = ms.grad(func, grad_position=0)
grad_x = grad_fn(x)
grad_x == u
```




    Tensor(shape=[4], dtype=Bool, value= [ True,  True,  True,  True])




```python
def func(x):
    y = x * x
    u = deepcopy(y)
    # u.requires_grad = True
    # u.requires_grad = False
    z = u * x
    z = z.sum()
    return z
grad_fn = ms.grad(func, grad_position=0)
grad_x = grad_fn(x)
grad_x == u
```




    Tensor(shape=[4], dtype=Bool, value= [ True, False, False, False])



由于记录了`y`的计算结果，我们可以随后在`y`上调用反向传播，
得到`y=x*x`关于的`x`的导数，即`2*x`。


此处发现了一个bug，MindSpore不能对含有Parameter的变量进行复合函数梯度求导，也就是第一步计算出来的y被视作了一个标量。


```python
def func(x):
    y = Parameter(x * x)
    y = y.sum()
    return y
grad_fn = ms.grad(func, grad_position=0)
grad_x = grad_fn(x)
grad_x == 2 * x
```




    Tensor(shape=[4], dtype=Bool, value= [ True, False, False, False])




```python
def func(x):
    y = x * x
    y = y.sum()
    return y
grad_fn = ms.grad(func, grad_position=0)
grad_x = grad_fn(x)
grad_x == 2 * x
```




    Tensor(shape=[4], dtype=Bool, value= [ True,  True,  True,  True])



## Python控制流的梯度计算

使用自动微分的一个好处是：
[**即使构建函数的计算图需要通过Python控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度**]。
在下面的代码中，`while`循环的迭代次数和`if`语句的结果都取决于输入`a`的值。



```python
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

让我们计算梯度。



```python
a = ops.randn(())
grad_fn = ms.grad(f, grad_position=0)
grad_a = grad_fn(a)
```

    [WARNING] KERNEL(331276,7ff6aaffd640,python):2023-07-16-19:02:10.308.959 [mindspore/ccsrc/kernel/kernel.h:518] CheckShapeNull] For 'Cast', the shape of input cannot contain zero, but got (0)


我们现在可以分析上面定义的`f`函数。
请注意，它在其输入`a`中是分段线性的。
换言之，对于任何`a`，存在某个常量标量`k`，使得`f(a)=k*a`，其中`k`的值取决于输入`a`，因此可以用`d/a`验证梯度是否正确。



```python
grad_a
```




    Tensor(shape=[], dtype=Float32, value= 204800)




```python
f(a) / a
```




    Tensor(shape=[], dtype=Float32, value= 204800)



这一点就比较奇怪，有时候明明是相等的，结果却不同。这也是我在MindSpore中发现的一个奇异现象。


```python
grad_a == f(a) / a
```




    Tensor(shape=[], dtype=Bool, value= False)



## 小结

* 深度学习框架可以自动计算导数：我们首先将梯度附加到想要对其计算偏导数的变量上，然后记录目标值的计算，执行它的反向传播函数，并访问得到的梯度。

## 练习

1. 为什么计算二阶导数比一阶导数的开销要更大？
1. 在运行反向传播函数之后，立即再次运行它，看看会发生什么。
1. 在控制流的例子中，我们计算`d`关于`a`的导数，如果将变量`a`更改为随机向量或矩阵，会发生什么？
1. 重新设计一个求控制流梯度的例子，运行并分析结果。
1. 使$f(x)=\sin(x)$，绘制$f(x)$和$\frac{df(x)}{dx}$的图像，其中后者不使用$f'(x)=\cos(x)$。


[Discussions](https://discuss.d2l.ai/t/1759)

