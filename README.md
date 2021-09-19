# ComputerVision_Interview
计算机视觉算法岗-知识点整理

本答案全部来源于网络，刚开始整理的时候，没有写好出处，实在抱歉

[toc]

## 跟进前沿遇到的知识点

### 1. 匈牙利算法(Hungarian algorithm) or KM(Kuhn-Munkres) 算法的流程

[hungarian-algorithm](https://github.com/benchaplin/hungarian-algorithm)

or

> 1. https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.linear_sum_assignment.html
> 2. https://brc2.com/the-algorithm-workshop/
> 3. https://github.com/bmc/munkres
> 4. https://zhuanlan.zhihu.com/p/89380238 为什么有效
> 5. https://arxiv.org/pdf/0911.1269.pdf 为什么有效(Hall’s theorem)
> A generalization of Hungarian method and Hall’s theorem with applications in wireless sensor networks

```python
cost = np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]])
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(cost)
###col_ind
#array([1, 0, 2])
###cost[row_ind, col_ind].sum()
#5
```

### 2. Gale-Shapley算法--寻找稳定婚配

```python
from collections import deque
def find_free_partner(boys, girls, sort_boy_to_girl, sort_girl_to_boy):
    # 当前选择的舞伴
    current_boys = dict(zip(boys, [None]*len(boys)))
    current_girls = dict(zip(girls, [None]*len(girls)))
    # current_boys = {boys[0]:None, boys[1]:None, boys[2]:None, boys[3]:None}
    # current_girls = {girls[0]:None, girls[1]:None, girls[2]:None, boys[3]:None}
    count = len(boys)

    # 建立队列，男孩下一次选择的女孩
    next_select = dict(zip(boys, [None]*len(boys)))
    for i in range(count):
        temp = [girls[m-1] for m in sort_boy_to_girl[i]]
        next_select[boys[i]] = deque(temp)

    # 女孩选择男孩字典
    sort_girl = dict(zip(girls, [None]*len(boys)))
    for i in range(count):
        # 通过题目给出的sort_girl_to_boy字典,排在前面的名字好感度比较高
        temp = [[boys[m-1], 4-ind] for ind, m in enumerate(sort_girl_to_boy[i])]
        name, love = [], []
        for t in temp:
            name.append(t[0])
            love.append(t[1])
        sort_girl[girls[i]] = dict(zip(name, love))

    while None in current_boys.values():
        for i in range(count):
            bid = boys[i]
            if current_boys[bid]:
                # 男孩有对象，跳过
                continue
            else:
                # 优先选择的女孩
                select = next_select[bid][0]
                if current_girls[select] == None:
                    # 女孩没对象，两者结合
                    current_boys[bid] = select
                    current_girls[select] = bid
                    next_select[bid].popleft()
                else:
                    # 和女孩的对象好感度对比,如果对现任的好感度,大于第三者,不动
                    if sort_girl[select][current_girls[select]] > sort_girl[select][bid]:
                        next_select[bid].popleft()
                    # 如果与上面相反
                    # 现任男孩失恋,第三者男孩选择了当前女孩,当前女孩选择了第三者男孩
                    # 第三者男孩失去对当前女孩的追求权(本算法不能对同一女士追求两次)
                    else:
                        current_boys[current_girls[select]] = None
                        current_boys[bid] = select
                        current_girls[select] = bid
                        next_select[bid].popleft()
    return current_boys

## 初始化
boys = ["Alex", "David", "Bob", "Chris"]
girls = ["Ada", "Becky", "Cindy", "Diana"]

# 偏爱列表
sort_boy_to_girl = [[1, 4, 3, 2], [3, 1, 2, 4],
                    [1, 2, 3, 4], [2, 4, 3, 1]]
sort_girl_to_boy = [[4, 1, 3, 2], [1, 2, 4, 3],
                    [3, 2, 4, 1], [2, 3, 1, 4]]

print(find_free_partner(boys, girls, sort_boy_to_girl, sort_girl_to_boy))
```

### 3. Sinkhorn算法

> 1. https://michielstock.github.io/posts/2017/2017-11-5-OptimalTransport/
>
> 2. https://blog.csdn.net/zsfcg/article/details/112510577 翻译
>
> 3. https://dfdazac.github.io/sinkhorn.html
>
> 4. https://www.jiqizhixin.com/articles/19031102 机器之心翻译
>
> 5. https://zhuanlan.zhihu.com/p/257069018?utm_source=wechat_session 量子位
>
>    https://github.com/lucidrains/sinkhorn-transformer
>
> 6. https://zhuanlan.zhihu.com/p/45980364 机器之心
>
> 7. https://baijiahao.baidu.com/s?id=1690211191638922715&wfr=spider&for=pc 机器之心
>
>    Wasserstein GAN
>    Generative Sinkhorn Modeling
>
> 8. https://baijiahao.baidu.com/s?id=1705967954519550438&wfr=spider&for=pc 机器之心 **YOLOX**
>
>    Sinkhorn-Knopp  替换为  **SimOTA**
>
> 9. https://baijiahao.baidu.com/s?id=1697633320792229739&wfr=spider&for=pc 机器之心 Sinkhorn distance

### 4. Physarum Dynamics算法(AAAI 2021)

> ​	[Physarum Powered Differentiable Linear Programming Layers and Applications](https://arxiv.org/abs/2004.14539)
>
> ​	该算法

## 深度学习相关问题

### 1. BN(Batch Normalization)层

### 2. dropout

### 3. 防止过拟合的方法

### 4. 权重初始化的方法

### 5. 优化算法(optimizer)

### 6. 深度学习主干网络backboon编年史

### 7. 目标检测NMS，numpy实现

### 8. 目标检测IOU计算，numpy实现

### 9. 一阶段目标检测和二阶段目标检测对比

### 10. focal-loss(RetinaNet)

### 11. Anchor free

### 12. L1-Norm和L2-Norm的区别

### 13. IOU扩展-旋转bbox的iou计算(即与坐标轴不平行的bbox)

### 14. attention原理(SK-Net SE-Net)

### 15. 感受野的计算

### 16. 交叉熵损失函数(sigmoid,softmax)

### 17. 局部极小值的定义

### 18. 大型网络为什么不容易陷入局部极小值

### 19. RPN(RegionProposal Network)细节

### 20. Adam优化器细节

### 21. 单机多卡训练时候，超参数如何变化，比如学习率lr，迭代次数iteration

### 22. BN层的numpy实现(forward， backward)

### 23. 数据增广的方法(Classification， ObjectDetection)

### 24. 网络不收敛的解决方法

### 25. 轻量化网络

### 26. merge_bn原理(bn在inference阶段的推理加速)

### 27. Triplet-loss 和 Softmax-loss 的区别

### 28. resnet解决了什么样的问题

### 29. RPN(RegionProposal Network)如何选择候选框

### 30. yolo-v4中的数据增强方式Mosaic

### 31. 数据增强方式 CutMix CutOut区别

### 32. yolo-v4的损失函数

### 33. label-smoothing原理

### 34. 难例样本挖掘(SSD)

### 35. IOU扩展(多边形IOU的计算问题)

### 36. 机器学习深度学习归一化的方式有哪些

### 37. LSTM相对于RNN的改进

### 38. softmax求导，sigmoid求导

### 39. nms soft-nms softer-nms

### 39. 如何尝试解决目标检测框遮挡问题

### 40. 如何尝试解决目标检测框重叠问题

### 41. smooth-L1loss,为什么用这个回归bbox

### 42. KL散度是什么意思

## pytorch相关问题

### 1. nn.module nn.Function的区别

   nn.Module实现的layer是由class Layer(nn.Module)定义的特殊类，会自动提取可学习参数nn.Parameter
   nn.functional中的函数更像是纯函数，由def function(input)定义。

   对于激活函数和池化层，由于没有可学习参数，一般使用nn.functional完成，其他的有学习参数的部分则使用类。但是Droupout由于在训练和测试时操作不同，所以建议使用nn.Module实现，它能够通过model.eval加以区分。

   搭配使用：

   ```python
   import torch as t
   import torch.nn as nn
   import torch.nn.functional as F
     
   class LeNet(nn.Module):
       def __init__(self):
           super(LeNet,self).__init__()
           self.conv1 = nn.Conv2d(3, 6, 5)
           self.conv2 = nn.Conv2d(6,16,5)
           self.fc1 = nn.Linear(16*5*5,120)
           self.fc2 = nn.Linear(120,84)
           self.fc3 = nn.Linear(84,10)
     
       def forward(self,x):
           x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
           x = F.max_pool2d(F.relu(self.conv2(x)),2)
           x = x.view(x.size()[0], -1)
           x = F.relu(self.fc1(x))
           x = F.relu(self.fc2(x))
           x = self.fc3(x)
           return x
   ```

   nn.functional是函数接口，而nn.Module是nn.functional的类封装，并且nn.Module都继承于一个共同祖先nn.Module。这一点导致nn.Module除了具有nn.functional功能之外，内部附带了nn.Module相关的属性和方法，例如train(), eval(),load_state_dict, state_dict 等。

   1)两者的调用方式不同
   nn.Module 需要先实例化并传入参数，然后以函数调用的方式调用实例化的对象并传入输入数据。

   ```python
   inputs = torch.rand(64, 3, 244, 244)
   conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
   out = conv(inputs)
   ```

   nn.functional 同时传入输入数据和weight, bias等其他参数 。

   ```python
   weight = torch.rand(64,3,3,3)
   bias = torch.rand(64) 
   out = nn.functional.conv2d(inputs, weight, bias, padding=1)
   ```

   2)nn.Module继承于nn.Module， 能够很好的与nn.Sequential结合使用， 而nn.functional无法与nn.Sequential结合使用。

   ```python
   fm_layer = nn.Sequential(
               nn.Conv2d(3, 64, kernel_size=3, padding=1),
               nn.BatchNorm2d(num_features=64),
               nn.ReLU(),
               nn.MaxPool2d(kernel_size=2),
               nn.Dropout(0.2)
     )
   ```

   3)nn.Module不需要你自己定义和管理weight；而nn.functional需要你自己定义weight，每次调用的时候都需要手动传入weight, 不利于代码复用。

   使用`nn.Module`定义一个CNN 。

   ```python
   class CNN(nn.Module):
       def __init__(self):
           super(CNN, self).__init__()
           
           self.cnn1 = nn.Conv2d(in_channels=1,  out_channels=16, kernel_size=5,padding=0)
           self.relu1 = nn.ReLU()
           self.maxpool1 = nn.MaxPool2d(kernel_size=2)
           
           self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,  padding=0)
           self.relu2 = nn.ReLU()
           self.maxpool2 = nn.MaxPool2d(kernel_size=2)
           
           self.linear1 = nn.Linear(4 * 4 * 32, 10)
           
       def forward(self, x):
           x = x.view(x.size(0), -1)
           out = self.maxpool1(self.relu1(self.cnn1(x)))
           out = self.maxpool2(self.relu2(self.cnn2(out)))
           out = self.linear1(out.view(x.size(0), -1))
           return out
   ```

   使用`nn.function`定义一个与上面相同的CNN。

   ```python
   class CNN(nn.Module):
       def __init__(self):
           super(CNN, self).__init__()
           
           self.cnn1_weight = nn.Parameter(torch.rand(16, 1, 5, 5))
           self.bias1_weight = nn.Parameter(torch.rand(16))
           
           self.cnn2_weight = nn.Parameter(torch.rand(32, 16, 5, 5))
           self.bias2_weight = nn.Parameter(torch.rand(32))
           
           self.linear1_weight = nn.Parameter(torch.rand(4 * 4 * 32, 10))
           self.bias3_weight = nn.Parameter(torch.rand(10))
           
       def forward(self, x):
           x = x.view(x.size(0), -1)
           out = F.conv2d(x, self.cnn1_weight, self.bias1_weight)
           out = F.relu(out)
           out = F.max_pool2d(out)
           
           out = F.conv2d(x, self.cnn2_weight, self.bias2_weight)
           out = F.relu(out)
           out = F.max_pool2d(out)
           
           out = F.linear(x, self.linear1_weight, self.bias3_weight)
           return out
   ```

   上面两种定义方式得到CNN功能都是相同的，至于喜欢哪一种方式，是个人口味问题，但PyTorch官方推荐：具有学习参数的（例如，conv2d, linear, batch_norm)采用`nn.Module`方式，没有学习参数的（例如，maxpool, loss func, activation func）等根据个人选择使用`nn.function`或者`nn.Module`方式。但关于dropout，个人强烈推荐使用`nn.Module`方式，因为一般情况下只有训练阶段才进行dropout，在eval阶段都不会进行dropout。使用`nn.Module`方式定义dropout，在调用`model.eval()`之后，model中所有的dropout layer都关闭，但以`nn.function.dropout`方式定义dropout，在调用`model.eval()`之后并不能关闭dropout。

### 2. ModuleList 和 Sequential的区别

   `nn.ModuleList`

   nn.ModuleList 这个类，你可以把任意 nn.Module 的子类 (比如 nn.Conv2d, nn.Linear 之类的) 加到这个 list 里面，方法和 Python 自带的 list 一样，无非是 extend，append 等操作。但不同于一般的 list，加入到 nn.ModuleList 里面的 module 是会自动注册到整个网络上的，同时 module 的 parameters 也会自动添加到整个网络中。描述看起来很枯燥，我们来看几个例子。

   `nn.Sequential`

   第一个网络，我们先来看看使用 nn.ModuleList 来构建一个小型网络，包括3个全连接层：

   ```python
   class net1(nn.Module):
       def __init__(self):
           super(net1, self).__init__()
           self.linears = nn.ModuleList([nn.Linear(10,10) for i in range(2)])
       def forward(self, x):
           for m in self.linears:
               x = m(x)
           return x
   
   net = net1()
   print(net)
   # net1(
   #   (modules): ModuleList(
   #     (0): Linear(in_features=10, out_features=10, bias=True)
   #     (1): Linear(in_features=10, out_features=10, bias=True)
   #   )
   # )
   
   for param in net.parameters():
       print(type(param.data), param.size())
   # <class 'torch.Tensor'> torch.Size([10, 10])
   # <class 'torch.Tensor'> torch.Size([10])
   # <class 'torch.Tensor'> torch.Size([10, 10])
   # <class 'torch.Tensor'> torch.Size([10])
   ```

   我们可以看到，这个网络包含两个全连接层，他们的权重 (weithgs) 和偏置 (bias) 都在这个网络之内。接下来我们看看第二个网络，它使用 Python 自带的 list：

   ```python
   class net2(nn.Module):
       def __init__(self):
           super(net2, self).__init__()
           self.linears = [nn.Linear(10,10) for i in range(2)]
       def forward(self, x):
           for m in self.linears:
               x = m(x)
           return x
   
   net = net2()
   print(net)
   # net2()
   print(list(net.parameters()))
   # []
   ```

   显然，使用 Python 的 list 添加的全连接层和它们的 parameters 并没有自动注册到我们的网络中。当然，我们还是可以使用 forward 来计算输出结果。但是如果用 net2 实例化的网络进行训练的时候，因为这些层的 parameters 不在整个网络之中，所以其网络参数也不会被更新，也就是无法训练。

   好，看到这里，我们大致明白了 nn.ModuleList 是干什么的了：它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。但是，我们需要注意到，nn.ModuleList 并没有定义一个网络，它只是将不同的模块储存在一起，这些模块之间并没有什么先后顺序可言，比如：

   ```python
   class net3(nn.Module):
       def __init__(self):
           super(net3, self).__init__()
           self.linears = nn.ModuleList([nn.Linear(10,20), nn.Linear(20,30), nn.Linear(5,10)])
       def forward(self, x):
           x = self.linears[2](x)
           x = self.linears[0](x)
           x = self.linears[1](x) 
           return x
   
   net = net3()
   print(net)
   # net3(
   #   (linears): ModuleList(
   #     (0): Linear(in_features=10, out_features=20, bias=True)
   #     (1): Linear(in_features=20, out_features=30, bias=True)
   #     (2): Linear(in_features=5, out_features=10, bias=True)
   #   )
   # )
   input = torch.randn(32, 5)
   print(net(input).shape)
   # torch.Size([32, 30])
   ```

   根据 net3 的结果，我们可以看出来这个 ModuleList 里面的顺序并不能决定什么，网络的执行顺序是根据 forward 函数来决定的。如果你非要 ModuleList 和 forward 中的顺序不一样， PyTorch 表示它无所谓，但以后 review 你代码的人可能会意见比较大。

   我们再考虑另外一种情况，既然这个 ModuleList 可以根据序号来调用，那么一个模块是否可以在 forward 函数中被调用多次呢？答案当然是可以的，但是，被调用多次的模块，是使用同一组 parameters 的，也就是它们的参数是共享的，无论之后怎么更新。例子如下，虽然在 forward 中我们用了 nn.Linear(10,10) 两次，但是它们只有一组参数。这么做有什么用处呢，比如可以定义孪生网络，如siamese-LSTM，就用到了参数共享；参数共享大幅减少计算量

   ```python
   class net4(nn.Module):
       def __init__(self):
           super(net4, self).__init__()
           self.linears = nn.ModuleList([nn.Linear(5, 10), nn.Linear(10, 10)])
       def forward(self, x):
           x = self.linears[0](x)
           x = self.linears[1](x)
           x = self.linears[1](x)
           return x
   
   net = net4()
   print(net)
   # net4(
   #   (linears): ModuleList(
   #     (0): Linear(in_features=5, out_features=10, bias=True)
   #     (1): Linear(in_features=10, out_features=10, bias=True)
   #   )
   # )
   for name, param in net.named_parameters():
       print(name, param.size())
   # linears.0.weight torch.Size([10, 5])
   # linears.0.bias torch.Size([10])
   # linears.1.weight torch.Size([10, 10])
   # linears.1.bias torch.Size([10])
   ```

`nn.Sequential`

现在我们来研究一下 nn.Sequential，不同于 nn.ModuleList，它已经实现了内部的 forward 函数，而且里面的模块必须是按照顺序进行排列的，所以我们必须确保前一个模块的输出大小和下一个模块的输入大小是一致的，如下面的例子所示：

```python
class net5(nn.Module):
    def __init__(self):
        super(net5, self).__init__()
        self.block = nn.Sequential(nn.Conv2d(1,20,5),
                                    nn.ReLU(),
                                    nn.Conv2d(20,64,5),
                                    nn.ReLU())
    def forward(self, x):
        x = self.block(x)
        return x

net = net5()
print(net)
# net5(
#   (block): Sequential(
#     (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#     (1): ReLU()
#     (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#     (3): ReLU()
#   )
# )
```

下面给出了两个 nn.Sequential 初始化的例子，来自于 [官网教程](https://link.zhihu.com/?target=https%3A//pytorch.org/docs/stable/nn.html%23sequential)。在第二个初始化中我们用到了 OrderedDict 来指定每个 module 的名字，而不是采用默认的命名方式 (按序号 0,1,2,3...) 。

```python
# Example of using Sequential
model1 = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
print(model1)
# Sequential(
#   (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#   (1): ReLU()
#   (2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#   (3): ReLU()
# )

# Example of using Sequential with OrderedDict
import collections
model2 = nn.Sequential(collections.OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
print(model2)
# Sequential(
#   (conv1): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
#   (relu1): ReLU()
#   (conv2): Conv2d(20, 64, kernel_size=(5, 5), stride=(1, 1))
#   (relu2): ReLU()
# )
```

有同学可能发现了，诶，你这个 model1 和 从类 net5 实例化来的 net 有什么区别吗？是没有的。这两个网络是相同的，因为 nn.Sequential 就是一个 nn.Module 的子类，也就是 nn.Module 所有的方法 (method) 它都有。并且直接使用 nn.Sequential 不用写 forward 函数，因为它内部已经帮你写好了。

这时候有同学该说了，既然 nn.Sequential 这么好，我以后都直接用它了。如果你确定 nn.Sequential 里面的顺序是你想要的，而且不需要再添加一些其他处理的函数 (比如 nn.functional 里面的函数，[nn 与 nn.functional 有什么区别?](https://www.zhihu.com/question/66782101/answer/579393790) )，那么完全可以直接用 nn.Sequential。这么做的代价就是失去了部分灵活性，毕竟不能自己去定制 forward 函数里面的内容了。

一般情况下 nn.Sequential 的用法是来组成卷积块 (block)，然后像拼积木一样把不同的 block 拼成整个网络，让代码更简洁，更加结构化。

`nn.ModuleList 和 nn.Sequential: 到底该用哪个`

**场景一**，有的时候网络中有很多相似或者重复的层，我们一般会考虑用 for 循环来创建它们，而不是一行一行地写，比如：

```python
layers = [nn.Linear(10, 10) for i in range(5)]
```

这个时候，很自然而然地，我们会想到使用 ModuleList，像这样：

```python
class net6(nn.Module):
    def __init__(self):
        super(net6, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(3)])

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x

net = net6()
print(net)
# net6(
#   (linears): ModuleList(
#     (0): Linear(in_features=10, out_features=10, bias=True)
#     (1): Linear(in_features=10, out_features=10, bias=True)
#     (2): Linear(in_features=10, out_features=10, bias=True)
#   )
# )
```

这个是比较一般的方法，但如果不想这么麻烦，我们也可以用 Sequential 来实现，如 net7 所示！注意 `*` 这个操作符，它可以把一个 list 拆开成一个个独立的元素。但是，请注意这个 list 里面的模块必须是按照想要的顺序来进行排列的。在 **场景一** 中，我个人觉得使用 net7 这种方法比较方便和整洁。

```python
class net7(nn.Module):
    def __init__(self):
        super(net7, self).__init__()
        self.linear_list = [nn.Linear(10, 10) for i in range(3)]
        self.linears = nn.Sequential(*self.linears_list)

    def forward(self, x):
        self.x = self.linears(x)
        return x

net = net7()
print(net)
# net7(
#   (linears): Sequential(
#     (0): Linear(in_features=10, out_features=10, bias=True)
#     (1): Linear(in_features=10, out_features=10, bias=True)
#     (2): Linear(in_features=10, out_features=10, bias=True)
#   )
# )
```

**场景二**，当我们需要之前层的信息的时候，比如 ResNets 中的 shortcut 结构，或者是像 FCN 中用到的 skip architecture 之类的，当前层的结果需要和之前层中的结果进行融合，一般使用 ModuleList 比较方便，一个非常简单的例子如下：

```python
class net8(nn.Module):
    def __init__(self):
        super(net8, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 20), nn.Linear(20, 30), nn.Linear(30, 50)])
        self.trace = []

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
            self.trace.append(x)
        return x

net = net8()
input  = torch.randn(32, 10) # input batch size: 32
output = net(input)
for each in net.trace:
    print(each.shape)
# torch.Size([32, 20])
# torch.Size([32, 30])
# torch.Size([32, 50])
```

我们使用了一个 trace 的列表来储存网络每层的输出结果，这样如果以后的层要用的话，就可以很方便地调用了。

**总结**

本文中我们通过一些实例学习了 ModuleList 和 Sequential 这两种 nn containers，ModuleList 就是一个储存各种模块的 list，这些模块之间没有联系，没有实现 forward 功能，但相比于普通的 Python list，ModuleList 可以把添加到其中的模块和参数自动注册到网络上。而Sequential 内的模块需要按照顺序排列，要保证相邻层的输入输出大小相匹配，内部 forward 功能已经实现，可以使代码更加整洁。在不同场景中，如果二者都适用，那就看个人偏好了。非常推荐大家看一下 PyTorch 官方的 [TorchVision](https://link.zhihu.com/?target=https%3A//github.com/pytorch/vision/tree/master/torchvision/models) 下面模型实现的代码，能学到很多构建网络的技巧。

### 3. pytorch反向传播前为什么清零

### 4. pytorch中detach的作用

### 5. pytorch的中断恢复，是在恢复什么东西

### 6. model.eval()  model.train()的作用

### 7. pytorch的中断恢复，Adadm优化器会恢复什么东西

### 8. pytorch的Dataloader

## python相关问题

### 1. python深拷贝, 浅拷贝

### 2. python函数的引用传递，值传递区别

### 3. 多进程，多线程相关问题


### 4. 元组touple和列表list的区别

### 5. 列表可以当字典的的key吗

## git相关问题
### 1. 待续

### 2. 待续

### 3. 待续

## docker相关问题
### 1. 待续

### 2. 待续

### 3. 待续

## shell相关问题

### 1. 待续

### 2. 待续

### 3. 待续

## Anaconda相关问题

### 1. 待续

### 2. 待续

### 3. 待续


## Leetcode相关问题

### 1. 岛屿问题---进阶版如果8个方向都可以呢(POJ-2386)

### 2. 待续

### 3. 待续