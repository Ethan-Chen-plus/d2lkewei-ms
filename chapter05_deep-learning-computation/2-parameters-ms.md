# 参数管理

在选择了架构并设置了超参数后，我们就进入了训练阶段。
此时，我们的目标是找到使损失函数最小化的模型参数值。
经过训练后，我们将需要使用这些参数来做出未来的预测。
此外，有时我们希望提取参数，以便在其他环境中复用它们，
将模型保存下来，以便它可以在其他软件中执行，
或者为了获得科学的理解而进行检查。

之前的介绍中，我们只依靠深度学习框架来完成训练的工作，
而忽略了操作参数的具体细节。
本节，我们将介绍以下内容：

* 访问参数，用于调试、诊断和可视化；
* 参数初始化；
* 在不同模型组件间共享参数。

(**我们首先看一下具有单隐藏层的多层感知机。**)



```python
import mindspore as ms
from mindspore import nn, ops

net = nn.SequentialCell(nn.Dense(4, 8), nn.ReLU(), nn.Dense(8, 1))
X = ops.rand((2, 4))
net(X)
```




    Tensor(shape=[2, 1], dtype=Float32, value=
    [[-4.33961391e-01],
     [-3.03957075e-01]])



## [**参数访问**]

我们从已有模型中访问参数。
当通过`Sequential`类定义模型时，
我们可以通过索引来访问模型的任意层。
这就像模型是一个列表一样，每层的参数都在其属性中。
如下所示，我们可以检查第二个全连接层的参数。



```python
print(net[2].parameters_dict())
```

    OrderedDict([('2.weight', Parameter (name=2.weight, shape=(1, 8), dtype=Float32, requires_grad=True)), ('2.bias', Parameter (name=2.bias, shape=(1,), dtype=Float32, requires_grad=True))])


输出的结果告诉我们一些重要的事情：
首先，这个全连接层包含两个参数，分别是该层的权重和偏置。
两者都存储为单精度浮点数（float32）。
注意，参数名称允许唯一标识每个参数，即使在包含数百个层的网络中也是如此。

### [**目标参数**]

注意，每个参数都表示为参数类的一个实例。
要对参数执行任何操作，首先我们需要访问底层的数值。
有几种方法可以做到这一点。有些比较简单，而另一些则比较通用。
下面的代码从第二个全连接层（即第三个神经网络层）提取偏置，
提取后返回的是一个参数类实例，并进一步访问该参数的值。



```python
print(type(net[2].bias))
print(net[2].bias)
print(net[2].bias.value())
```

    <class 'abc.Parameter'>
    Parameter (name=2.bias, shape=(1,), dtype=Float32, requires_grad=True)
    [-0.3341447]


参数是复合的对象，包含值、梯度和额外信息。
这就是我们需要显式参数值的原因。
除了值之外，我们还可以访问每个参数的梯度。
在上面这个网络中，由于我们还没有调用反向传播，所以参数的梯度处于初始状态。
MindSpore默认是没有梯度这个属性的。

### [**一次性访问所有参数**]

当我们需要对所有参数执行操作时，逐个访问它们可能会很麻烦。
当我们处理更复杂的块（例如，嵌套块）时，情况可能会变得特别复杂，
因为我们需要递归整个树来提取每个子块的参数。
下面，我们将通过演示来比较访问第一个全连接层的参数和访问所有层。



```python
print(*[(name, param.shape) for name, param in net[0].parameters_dict().items()])
print(*[(name, param.shape) for name, param in net.parameters_dict().items()])
```

    ('0.weight', (8, 4)) ('0.bias', (8,))
    ('0.weight', (8, 4)) ('0.bias', (8,)) ('2.weight', (1, 8)) ('2.bias', (1,))


这为我们提供了另一种访问网络参数的方式，如下所示。



```python
net.parameters_dict()['2.bias'].value()
```




    Tensor(shape=[1], dtype=Float32, value= [-3.34144711e-01])



### [**从嵌套块收集参数**]

让我们看看，如果我们将多个块相互嵌套，参数命名约定是如何工作的。
我们首先定义一个生成块的函数（可以说是“块工厂”），然后将这些块组合到更大的块中。



```python
def block1():
    return nn.SequentialCell(nn.Dense(4, 8), nn.ReLU(),
                             nn.Dense(8, 4), nn.ReLU())

def block2():
    net = nn.SequentialCell()
    for i in range(4):
        # 在这里嵌套
        net.insert_child_to_cell(f'block {i}', block1())
    return net

rgnet = nn.SequentialCell(block2(), nn.Dense(4, 1))
rgnet(X)
```




    Tensor(shape=[2, 1], dtype=Float32, value=
    [[-6.19896591e-01],
     [-5.72349489e-01]])



[**设计了网络后，我们看看它是如何工作的。**]



```python
print(rgnet)
```

    SequentialCell<
      (0): SequentialCell<
        (block 0): SequentialCell<
          (0): Dense<input_channels=4, output_channels=8, has_bias=True>
          (1): ReLU<>
          (2): Dense<input_channels=8, output_channels=4, has_bias=True>
          (3): ReLU<>
          >
        (block 1): SequentialCell<
          (0): Dense<input_channels=4, output_channels=8, has_bias=True>
          (1): ReLU<>
          (2): Dense<input_channels=8, output_channels=4, has_bias=True>
          (3): ReLU<>
          >
        (block 2): SequentialCell<
          (0): Dense<input_channels=4, output_channels=8, has_bias=True>
          (1): ReLU<>
          (2): Dense<input_channels=8, output_channels=4, has_bias=True>
          (3): ReLU<>
          >
        (block 3): SequentialCell<
          (0): Dense<input_channels=4, output_channels=8, has_bias=True>
          (1): ReLU<>
          (2): Dense<input_channels=8, output_channels=4, has_bias=True>
          (3): ReLU<>
          >
        >
      (1): Dense<input_channels=4, output_channels=1, has_bias=True>
      >


因为层是分层嵌套的，所以我们也可以像通过嵌套列表索引一样访问它们。
下面，我们访问第一个主要的块中、第二个子块的第一层的偏置项。



```python
rgnet[0][1][0].bias.value()
```




    Tensor(shape=[8], dtype=Float32, value= [-2.60119308e-02,  1.20149732e-01,  5.43249175e-02, -3.95016611e-01,  3.61491054e-01, -3.70439142e-01,  8.23884737e-03,  2.11492881e-01])



## 参数初始化

知道了如何访问参数后，现在我们看看如何正确地初始化参数。
我们在 :numref:`sec_numerical_stability`中讨论了良好初始化的必要性。
深度学习框架提供默认随机初始化，
也允许我们创建自定义初始化方法，
满足我们通过其他规则实现初始化权重。


默认情况下，PyTorch会根据一个范围均匀地初始化权重和偏置矩阵，
这个范围是根据输入和输出维度计算出的。
PyTorch的`nn.init`模块提供了多种预置初始化方法。


### [**内置初始化**]

让我们首先调用内置的初始化器。
下面的代码将所有权重参数初始化为标准差为0.01的高斯随机变量，
且将偏置参数设置为0。



```python
from mindspore.common.initializer import initializer, Normal, Constant, XavierUniform, Uniform
```


```python
def init_normal(m):
    if type(m) == nn.Dense:
        m.weight.set_data(initializer(Normal(sigma=0.01, mean=0), m.weight.shape, m.weight.dtype))
        m.bias.set_data(initializer(Constant(0.0),m.bias.shape,m.bias.dtype))
net.apply(init_normal)
net[0].weight.data[0], net[0].bias.data[0]
```




    (Tensor(shape=[4], dtype=Float32, value= [-4.31267265e-03, -5.35854883e-03,  5.15463871e-05, -7.43840635e-03]),
     Tensor(shape=[], dtype=Float32, value= 0))



我们还可以将所有参数初始化为给定的常数，比如初始化为1。



```python
def init_constant(m):
    if type(m) == nn.Dense:
        m.weight.set_data(initializer(Constant(1), m.weight.shape, m.weight.dtype))
        m.bias.set_data(initializer(Constant(0),m.bias.shape,m.bias.dtype))
        
net.apply(init_constant)
net[0].weight.data[0], net[0].bias.data[0]
```




    (Tensor(shape=[4], dtype=Float32, value= [ 1.00000000e+00,  1.00000000e+00,  1.00000000e+00,  1.00000000e+00]),
     Tensor(shape=[], dtype=Float32, value= 0))



我们还可以[**对某些块应用不同的初始化方法**]。
例如，下面我们使用Xavier初始化方法初始化第一个神经网络层，
然后将第三个神经网络层初始化为常量值42。



```python
def init_xavier(m):
    if type(m) == nn.Dense:
        m.weight.set_data(initializer(XavierUniform(), m.weight.shape, m.weight.dtype))
def init_42(m):
    if type(m) == nn.Dense:
        m.weight.set_data(initializer(Constant(42), m.weight.shape, m.weight.dtype))

net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data[0])
```

    [ 0.57022125 -0.21588863  0.41397578  0.01421497]
    [42. 42. 42. 42. 42. 42. 42. 42.]


### [**自定义初始化**]

有时，深度学习框架没有提供我们需要的初始化方法。
在下面的例子中，我们使用以下的分布为任意权重参数$w$定义初始化方法：

$$
\begin{aligned}
    w \sim \begin{cases}
        U(5, 10) & \text{ 可能性 } \frac{1}{4} \\
            0    & \text{ 可能性 } \frac{1}{2} \\
        U(-10, -5) & \text{ 可能性 } \frac{1}{4}
    \end{cases}
\end{aligned}
$$



```python
net[0].parameters_dict()
```




    OrderedDict([('0.weight',
                  Parameter (name=0.weight, shape=(8, 4), dtype=Float32, requires_grad=True)),
                 ('0.bias',
                  Parameter (name=0.bias, shape=(8,), dtype=Float32, requires_grad=True))])



同样，我们实现了一个`my_init`函数来应用到`net`。



```python
def my_init(m):
    if type(m) == nn.Dense:
        print("Init", *[(name, param.shape)
                        for name, param in m.parameters_dict().items()][0])
        m.weight.set_data(initializer(Uniform(scale=10), m.weight.shape, m.weight.dtype))
        # print(m.weight.data.abs())
        # print(m.weight.data.abs()>=5)
        # print(m.weight.value()*(m.weight.data.abs() >= 5))
        m.weight.set_data(m.weight.value()*(m.weight.data.abs() >= 5))

net.apply(my_init)
net[0].weight[:2]
```

    Init 0.weight (8, 4)
    Init 2.weight (1, 8)





    Tensor(shape=[2, 4], dtype=Float32, value=
    [[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -8.66300011e+00],
     [ 6.78226042e+00, -0.00000000e+00, -6.63471270e+00, -6.80070257e+00]])



注意，我们始终可以直接设置参数。



```python
net[0].weight.data[:] += 1
net[0].weight.data[0, 0] = 42
net[0].weight.data[0]
```




    Tensor(shape=[4], dtype=Float32, value= [ 4.20000000e+01,  1.00000000e+00,  1.00000000e+00, -7.66300011e+00])



## [**参数绑定**]

有时我们希望在多个层间共享参数：
我们可以定义一个稠密层，然后使用它的参数来设置另一个层的参数。



```python
# 我们需要给共享层一个名称，以便可以引用它的参数
shared = nn.Dense(8, 8)
net = nn.SequentialCell(nn.Dense(4, 8), nn.ReLU(),
                        shared, nn.ReLU(),
                        shared, nn.ReLU(),
                        nn.Dense(8, 1))
net(X)
# 检查参数是否相同
print(net[2].weight.data[0] == net[4].weight.data[0])
net[2].weight.data[0, 0] = 100
# 确保它们实际上是同一个对象，而不只是有相同的值
print(net[2].weight.data[0] == net[4].weight.data[0])
```

    [ True  True  True  True  True  True  True  True]
    [ True  True  True  True  True  True  True  True]


这个例子表明第三个和第五个神经网络层的参数是绑定的。
它们不仅值相等，而且由相同的张量表示。
因此，如果我们改变其中一个参数，另一个参数也会改变。
这里有一个问题：当参数绑定时，梯度会发生什么情况？
答案是由于模型参数包含梯度，因此在反向传播期间第二个隐藏层
（即第三个神经网络层）和第三个隐藏层（即第五个神经网络层）的梯度会加在一起。


## 小结

* 我们有几种方法可以访问、初始化和绑定模型参数。
* 我们可以使用自定义初始化方法。

## 练习

1. 使用 :numref:`sec_model_construction` 中定义的`FancyMLP`模型，访问各个层的参数。
1. 查看初始化模块文档以了解不同的初始化方法。
1. 构建包含共享参数层的多层感知机并对其进行训练。在训练过程中，观察模型各层的参数和梯度。
1. 为什么共享参数是个好主意？


[Discussions](https://discuss.d2l.ai/t/1829)

