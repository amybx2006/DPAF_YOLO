import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x


class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
         super().__init__()

         self.skip = conv_block(in_features=in_features,
                                out_features=filters,
                                kernel_size=(1, 1),
                                padding=(0, 0),
                                norm_type='bn',
                                activation=False)
         self.c1 = conv_block(in_features=in_features,
                                out_features=filters,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                norm_type='bn',
                                activation=True)
         self.c2 = conv_block(in_features=filters,
                                out_features=filters,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                norm_type='bn',
                                activation=True)
         self.c3 = conv_block(in_features=filters,
                                out_features=filters,
                                kernel_size=(3, 3),
                                padding=(1, 1),
                                norm_type='bn',
                                activation=True)
         self.sa = SpatialAttentionModule()
         self.cn = EMA(filters)
         self.lga2 = LocalGlobalAttention(filters, 2)
         self.lga4 = LocalGlobalAttention(filters, 4)

         self.bn1 = nn.BatchNorm2d(filters)
         self.drop = nn.Dropout2d(0.1)
         self.relu = nn.ReLU()

         self.gelu = nn.GELU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor  # 分组率
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)  # Softmax
        self.agp = nn.AdaptiveAvgPool2d((1, 1))  # 平均池化层
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # x平均池化层 h=1
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # y平均池化层 w=1
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)  # 分组操作
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1,
                                 padding=0)  # 1×1卷积分支
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1,
                                 padding=1)  # 3×3卷积分支

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

class DASI(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        self.bag = Bag()

        self.tail_conv = conv_block(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=None,
            activation=False
        )

        self.conv = conv_block(
            in_features=out_features // 2,
            out_features=out_features // 4,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=None,
            activation=False
        )

        self.bns = nn.BatchNorm2d(out_features)

        self.skips = conv_block(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=None,
            activation=False
        )

        self.relu = nn.ReLU()

    def align(self, feat, target):

        if feat is None:
            return None

        # 对齐空间尺寸
        if feat.shape[2:] != target.shape[2:]:

            feat = F.interpolate(
                feat,
                size=target.shape[2:],
                mode="bilinear",
                align_corners=True
            )

        # 对齐通道
        c_tar = target.shape[1]
        c = feat.shape[1]

        if c > c_tar:
            feat = feat[:, :c_tar]

        elif c < c_tar:

            feat = F.pad(
                feat,
                (0, 0, 0, 0, 0, c_tar - c)
            )

        return feat

    def forward(self, inputs):

        x = inputs[0]
        x_low = inputs[1]
        x_high = inputs[2]

        x_low = self.align(x_low, x)
        x_high = self.align(x_high, x)

        x_skip = self.skips(x)

        x = self.skips(x)

        x = torch.chunk(x, 4, dim=1)

        if x_low is not None:
            x_low = torch.chunk(x_low, 4, dim=1)

        if x_high is not None:
            x_high = torch.chunk(x_high, 4, dim=1)

        if x_high is None:

            x0 = self.conv(torch.cat([x[0], x_low[0]], 1))
            x1 = self.conv(torch.cat([x[1], x_low[1]], 1))
            x2 = self.conv(torch.cat([x[2], x_low[2]], 1))
            x3 = self.conv(torch.cat([x[3], x_low[3]], 1))

        elif x_low is None:

            x0 = self.conv(torch.cat([x[0], x_high[0]], 1))
            x1 = self.conv(torch.cat([x[1], x_high[1]], 1))
            x2 = self.conv(torch.cat([x[2], x_high[2]], 1))
            x3 = self.conv(torch.cat([x[3], x_high[3]], 1))

        else:

            x0 = self.bag(x_low[0], x_high[0], x[0])
            x1 = self.bag(x_low[1], x_high[1], x[1])
            x2 = self.bag(x_low[2], x_high[2], x[2])
            x3 = self.bag(x_low[3], x_high[3], x[3])

        x = torch.cat(
            [x0, x1, x2, x3],
            dim=1
        )

        x = self.tail_conv(x)

        x = x + x_skip

        x = self.bns(x)

        x = self.relu(x)

        return x
    
class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True))
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output

class Bag(nn.Module):

    def __init__(self):
        super().__init__()

    def _align(self, feat, ref):
        """
        将 feat 对齐到 ref：
        1. 空间尺寸一致
        2. 通道一致
        """

        # 尺寸对齐
        if feat.shape[2:] != ref.shape[2:]:
            feat = F.interpolate(
                feat,
                size=ref.shape[2:],
                mode="bilinear",
                align_corners=True
            )

        # 通道对齐
        c_ref = ref.shape[1]
        c_feat = feat.shape[1]

        if c_feat > c_ref:
            feat = feat[:, :c_ref]

        elif c_feat < c_ref:
            feat = F.pad(
                feat,
                (0, 0, 0, 0, 0, c_ref - c_feat)
            )

        return feat

    def forward(self, p, i, d):

        # 全部对齐到 d
        p = self._align(p, d)
        i = self._align(i, d)

        edge_att = torch.sigmoid(d)

        edge_att = self._align(edge_att, p)

        out = edge_att * p + (1 - edge_att) * i

        return out


class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x


class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3),
                 stride=(1, 1),
                 padding=(1, 1),
                 dilation=(1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups = 1
                 ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups = groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            # self.relu = nn.GELU()
            self.relu = nn.ReLU(inplace=False)


    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x


class MDCR(nn.Module):
    def __init__(self, in_features, out_features, norm_type='bn', activation=True, rate=[1, 6, 12, 18]):
        super().__init__()

        self.block1 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[0],
            dilation=rate[0],
            norm_type=norm_type,
            activation=activation,
            groups = 128
            )
        self.block2 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[1],
            dilation=rate[1],
            norm_type=norm_type,
            activation=activation,
            groups=128
            )
        self.block3 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[2],
            dilation=rate[2],
            norm_type=norm_type,
            activation=activation,
            groups=128
            )
        self.block4 = conv_block(
            in_features=in_features//4,
            out_features=out_features//4,
            padding=rate[3],
            dilation=rate[3],
            norm_type=norm_type,
            activation=activation,
            groups=128
            )
        self.out_s = conv_block(
            in_features=4,
            out_features=4,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
        )
        self.out = conv_block(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(1, 1),
            padding=(0, 0),
            norm_type=norm_type,
            activation=activation,
            )

    def forward(self, x):
        split_tensors = []
        x = torch.chunk(x, 4, dim=1)
        x1 = self.block1(x[0])
        x2 = self.block2(x[1])
        x3 = self.block3(x[2])
        x4 = self.block4(x[3])
        for channel in range(x1.size(1)):
            channel_tensors = [tensor[:, channel:channel + 1, :, :] for tensor in [x1, x2, x3, x4]]
            concatenated_channel = self.out_s(torch.cat(channel_tensors, dim=1))  # 拼接在 batch_size 维度上
            split_tensors.append(concatenated_channel)
        x = torch.cat(split_tensors, dim=1)
        x = self.out(x)
        return x
