```pycon
import torch
from einops import rearrange

def mean_flat(tensor: torch.Tensor, mask=None):
    """
    计算张量在非批次维度上的均值。

    Args:
        tensor (torch.Tensor): 输入张量。
        mask (torch.Tensor, optional): 掩码张量，用于加权平均计算。默认为None。

    Returns:
        torch.Tensor: 在非批次维度上计算的均值。
    """
    if mask is None:
        # 如果没有掩码，直接计算所有非批次维度的均值
        return tensor.mean(dim=list(range(1, len(tensor.shape))))
    else:
        # 检查输入张量和掩码的维度是否符合要求
        assert tensor.dim() == 5
        assert tensor.shape[2] == mask.shape[1]
        
        # 重新排列张量以适应后续计算
        tensor = rearrange(tensor, "b c t h w -> b t (c h w)")
        
        # 计算分母，即掩码的总和乘以最后一个维度的大小
        denom = mask.sum(dim=1) * tensor.shape[-1]
        
        # 计算加权平均值
        loss = (tensor * mask.unsqueeze(2)).sum(dim=1).sum(dim=1) / denom
        
        return loss

# 创建一个形状为 (3, 2, 4, 5, 5) 的随机张量
tensor = torch.randn(3, 2, 4, 5, 5)

# 创建一个形状为 (3, 4) 的随机掩码张量，值为0或1
mask = torch.randint(0, 2, (3, 4))

# 调用 mean_flat 函数
result = mean_flat(tensor, mask)

print("输入张量:")
print(tensor)
print("\n掩码张量:")
print(mask)
print("\n计算结果:")
print(result)

```


