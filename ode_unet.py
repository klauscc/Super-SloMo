# -*- coding: utf-8 -*-
#================================================================
#   God Bless You.
#
#   author: klaus
#   email: chengfeng2333@gmail.com
#   created date: 2020/04/30
#   description:
#
#================================================================

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint as odeint

from model import down, up


class ConcatConv2d(nn.Module):

    def __init__(self,
                 dim_in,
                 dim_out,
                 ksize=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(dim_in + 1,
                             dim_out,
                             kernel_size=ksize,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             groups=groups,
                             bias=bias)

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        # self.norm2 = norm(dim)
        # self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        # self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        # out = self.norm2(out)
        # out = self.relu(out)
        # out = self.conv2(t, out)
        # out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc, method):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc

        self.nfe = 0
        self.tol = 1e-3
        self.method = method

    def forward(self, ts, x):
        """TODO: Docstring for forward.

        Args:
            ts (array): time to integrate from 0. shape: (bs,) 
            x (tensor): Input to odenet. Shape: (bs, C, H, W). 

        Returns: The same shape as input.

        """
        outs = []
        self.nfe = []
        for i, t in enumerate(ts):
            integration_time = torch.tensor([0, t]).float().type_as(x)
            out = odeint(self.odefunc,
                         x[i:i + 1, ...],
                         integration_time,
                         rtol=self.tol,
                         atol=self.tol,
                         method=self.method,
                         options={"step_size": 0.0625})
            outs.append(out[1])

            self.nfe.append(self.odefunc.nfe)
            self.odefunc.nfe = 0
        return torch.cat(outs, dim=0)

    # @property
    # def nfe(self):
    # return self.nfe

    # @nfe.setter
    # def nfe(self, value):
    # self.nfe = value


class ODE_UNet(nn.Module):
    """ODE-UNet"""

    def __init__(self, cin, cout, ode_method):
        super(ODE_UNet, self).__init__()

        self.conv1 = nn.Conv2d(cin, 32, 7, stride=1, padding=3)
        self.conv2 = nn.Conv2d(32, 32, 7, stride=1, padding=3)
        self.down1 = down(32, 64, 5)
        self.down2 = down(64, 128, 3)
        self.down3 = down(128, 256, 3)
        self.down4 = down(256, 512, 3)
        self.down5 = down(512, 512, 3)
        self.up1 = up(512, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.up5 = up(64, 32)
        self.conv3 = nn.Conv2d(32, cout, 3, stride=1, padding=1)

        self.flows = nn.ModuleList(
            [ODEBlock(ODEfunc(dim=c), method=ode_method) for c in [32, 64, 128, 256, 512, 512]])

    def flow(self, t, pyramid_features):
        """
        Args:
            t (array): time to integrate from 0. shape: (bs,) 
            pyramid_features (List): features at different level: [s1, s2, s3, s4, s5, bridge] 

        Returns:
            The flowed features. The same shape as pyramid_features.
        """
        assert len(self.flows) == len(pyramid_features)
        outs = []
        for i, ode_block in enumerate(self.flows):
            out = ode_block(t, pyramid_features[i])
            outs.append(out)
        return outs

    def forward(self, t, x):
        """

        Args:
            t (float): The time integrated from 0 to.
            x (tensor): The input to the NN block. shape: (bs, cin, h, w)

        Returns: tensor. The output of the NN block.

        """
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        s1 = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        s2 = self.down1(s1)
        s3 = self.down2(s2)
        s4 = self.down3(s3)
        s5 = self.down4(s4)
        x = self.down5(s5)    # c: 512

        s1, s2, s3, s4, s5, x = self.flow(t, [s1, s2, s3, s4, s5, x])

        x = self.up1(x, s5)
        x = self.up2(x, s4)
        x = self.up3(x, s3)
        x = self.up4(x, s2)
        x = self.up5(x, s1)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.1)
        return x
