import numpy as np
import torch

"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

# vae 支持 重参数采样 和 kl约束
class DiagonalGaussianDistribution(object):
    def __init__(
        self,
        parameters,
        deterministic=False,
    ):
        self.parameters = parameters
        # 按列，在中间砍一刀，分为两份
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # # 将logvar的值限制在-30到20之间，以避免数值不稳定的问题
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            # 确定性模式：在确定性模式下，模型的行为应该是完全可预测的，不受随机因素的影响。
            # 方差 标准差 为0
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device, dtype=self.mean.dtype)

    def sample(self): #重参数化采样，使得可微分
        # torch.randn: standard normal distribution
        # 生成一个与 self.mean 形状相同的张量 x
        # 使用正态分布（均值为0，标准差为1）生成随机噪声，进行参数重采样，支持 可微分
        # 将随机噪声乘以 self.std 并加到 self.mean 上
        # 将结果转换为与 self.parameters 相同的设备和数据类型
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device, dtype=self.mean.dtype)
        return x

    def kl(self, other=None): # 约束code的分布符合 （0,1）高斯分布，防止 重建过程，code变得单一
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:  # SCH: assumes other is a standard normal distribution
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3, 4])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3, 4],
                )

    def nll(self, sample, dims=[1, 2, 3, 4]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean
