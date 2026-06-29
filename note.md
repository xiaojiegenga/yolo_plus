# Bottlenneck类 block.py:457
```python
class Bottleneck(nn.Module):
    """Standard bottleneck."""
    def __init__(
        self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5
    ):
        """Initialize a standard bottleneck module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            shortcut (bool): Whether to use shortcut connection.
            g (int): Groups for convolutions.
            k (tuple): Kernel sizes for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
```

- Bottleneck 是一个“先降维、再提特征、可选残差”的两层卷积块，主要用于 Backbone/Neck 里做特征提取与融合，而不是在 Head 里直接输出预测。它的核心目的：减少计算量、保留表达能力、让梯度更容易传播。
- `shortcut` 参数控制是否使用残差连接（输入直接加到输出），这有助于训练更深的网络。`e` 控制隐藏层的宽度，通常设置为 0.5 表示压缩一半的通道数。  使用的add残差连接，输入和输出通道数必须相同（`c1 == c2`），否则就不能直接相加。

# C2F类 block.py:288

```python
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        """Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
```
- C2f = “先用1×1卷积生成两份特征 → 一份直通，一份经过 n 个 Bottleneck 深加工 → 拼起来 → 再压回输出通道”

# C3k2类 block.py:1069
```python
class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
```

- 它只比 C2f 多一个开关 —— YAML 里 C3k2 [256, False, 0.25] 第二个参数 False 就是这个开关。
重点看：c3k=True 用更复杂的 C3k，c3k=False 退化成 Bottleneck
- 回看 YAML 第 24 行 C3k2 [256, False, 0.25] —— c3k=False，所以用 Bottleneck；第 28 行 C3k2 [512, True] —— c3k=True，用 C3k。浅层用简单的，深层用复杂的.