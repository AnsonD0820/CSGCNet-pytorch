import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..models.mobilenetv3 import mobilenetv3_small as MobileNetV3
from torch.nn import Softmax


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)



BatchNorm2d = nn.BatchNorm2d
BatchNorm1d = nn.BatchNorm1d

class _DWConv(nn.Module):
	"""Depthwise Convolutions"""
	def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
		super(_DWConv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(True)
		)

	def forward(self, x):
		return self.conv(x)


class _DSConv(nn.Module):
	"""Depthwise Separable Convolutions"""
	def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
		super(_DSConv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
			nn.BatchNorm2d(dw_channels),
			nn.ReLU(True),
			nn.Conv2d(dw_channels, out_channels, 1, bias=False),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(True)
		)

	def forward(self, x):
		return self.conv(x)

class ConvBNReLU(nn.Module):
	def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1, groups = 1,*args, **kwargs):
		super(ConvBNReLU, self).__init__()
		self.conv = nn.Conv2d(in_chan,
				out_chan,
				kernel_size = kernel_size,
				stride = stride,
				padding = padding,
				bias = False,
				groups=groups)
		#self.bn = nn.BatchNorm2d(out_chan)
		self.bn = torch.nn.GroupNorm(out_chan,out_chan, eps=1e-05, affine=True, device=None, dtype=None)
		self.relu = nn.ReLU()
		self.init_weight()

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x

	def init_weight(self):
		for ly in self.children():
			if isinstance(ly, nn.Conv2d):
				nn.init.kaiming_normal_(ly.weight, a=1)
				if not ly.bias is None: nn.init.constant_(ly.bias, 0)


	def get_params(self):
		wd_params, nowd_params = [], []
		for name, module in self.named_modules():
			if isinstance(module, (nn.Linear, nn.Conv2d)):
				wd_params.append(module.weight)
				if not module.bias is None:
					nowd_params.append(module.bias)
			elif isinstance(module, nn.BatchNorm2d):
				nowd_params += list(module.parameters())
		return wd_params, nowd_params


class AttentionBranch(nn.Module):
	def __init__(self, inplanes, interplanes, outplanes, num_classes):
		super(AttentionBranch, self).__init__()
		self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
								   nn.BatchNorm2d(interplanes),
								   nn.ReLU(interplanes))
		#self.a2block = ContextAggregationBlock(interplanes, interplanes//2)
		#self.a2block = SpatialGCN(plane=interplanes,dilation=3)
		self.a2block = CrissCrossAttention(in_dim=interplanes,group=interplanes//16)
		self.convb = nn.Conv2d(interplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=True)

		self.b1 = nn.Conv2d(inplanes + interplanes, outplanes, kernel_size=3, padding=1, bias=False)
		self.b2 = nn.BatchNorm2d(interplanes)
		self.b3 = nn.ReLU(interplanes)
		self.b4 = nn.Conv2d(interplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
		self.init_weight()

	def forward(self, x):
		#print(x.size())
		output = self.conva(x)
		#print(output.size())
		#output = self.a2block(output)
		output = self.convb(output)
		output = torch.cat([x, output], 1)
		output = self.b1(output)
		output = self.b2(output)
		output = self.b3(output)
		output_final = self.b4(output)
		return output, output_final

	def init_weight(self):
		for ly in self.children():
			if isinstance(ly, nn.Conv2d):
				nn.init.kaiming_normal_(ly.weight, a=1)
				if not ly.bias is None: nn.init.constant_(ly.bias, 0)

	def get_params(self):
		wd_params, nowd_params = [], []
		for name, module in self.named_modules():
			if isinstance(module, (nn.Linear, nn.Conv2d)):
				wd_params.append(module.weight)
				if not module.bias is None:
					nowd_params.append(module.bias)
			elif isinstance(module, nn.BatchNorm2d):
				nowd_params += list(module.parameters())
		return wd_params, nowd_params


class SpatialBranch(nn.Module):
	def __init__(self, *args, **kwargs):
		super(SpatialBranch, self).__init__()
		self.conv1 = ConvBNReLU(3, 16, kernel_size=7, stride=2, padding=3)
		self.conv2 = ConvBNReLU(16, 32, kernel_size=3, stride=2, padding=1,group=16)
		self.conv3 = ConvBNReLU(32, 64, kernel_size=3, stride=2, padding=1,group=16)
		self.conv_out = ConvBNReLU(64, 128, kernel_size=1, stride=1, padding=0,groups=1)

		self.bn1 = nn.BatchNorm2d(3)
		self.bn2 = nn.BatchNorm2d(64)
		self.bn = torch.nn.GroupNorm(1,64, eps=1e-05, affine=True, device=None, dtype=None)
		self.relu = nn.ReLU()
		self.init_weight()

	def forward(self, x):
		feat = self.conv1(x)
		feat = channel_shuffle(feat,groups=16)
		feat = self.conv2(feat)
		feat = channel_shuffle(feat,groups=16)
		feat = self.conv3(feat)
		feat = channel_shuffle(feat,groups=16)
		feat = self.conv_out(feat)

		return feat

	def init_weight(self):
		for ly in self.children():
			if isinstance(ly, nn.Conv2d):
				nn.init.kaiming_normal_(ly.weight, a=1)
				if not ly.bias is None: nn.init.constant_(ly.bias, 0)

	def get_params(self):
		wd_params, nowd_params = [], []
		for name, module in self.named_modules():
			if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
				wd_params.append(module.weight)
				if not module.bias is None:
					nowd_params.append(module.bias)
			elif isinstance(module, nn.BatchNorm2d):
				nowd_params += list(module.parameters())
		return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
	def __init__(self, in_chan, out_chan, *args, **kwargs):
		super(FeatureFusionModule, self).__init__()
		self.convblk = ConvBNReLU(in_chan, out_chan, kernel_size=1, stride=1, padding=0)
		self.conv1 = nn.Conv2d(out_chan,
				out_chan//4,
				kernel_size = 1,
				stride = 1,
				padding = 0,
				bias = False)
		self.conv2 = nn.Conv2d(out_chan//4,
				out_chan,
				kernel_size = 1,
				stride = 1,
				padding = 0,
				bias = False)
		self.relu = nn.ReLU(inplace=True)
		self.sigmoid = nn.Softmax(dim=-1)
		self.init_weight()

	def forward(self, fsp, fcp):
		fcat = torch.cat([fsp, fcp], dim=1)
		feat = self.convblk(fcat)
		atten = F.avg_pool2d(feat, feat.size()[2:])
		atten = self.conv1(atten)
		atten = self.relu(atten)
		atten = self.conv2(atten)
		atten = self.sigmoid(atten)
		feat_atten = torch.mul(feat, atten)
		feat_out = feat_atten + feat
		return feat_out

	def init_weight(self):
		for ly in self.children():
			if isinstance(ly, nn.Conv2d):
				nn.init.kaiming_normal_(ly.weight, a=1)
				if not ly.bias is None: nn.init.constant_(ly.bias, 0)

	def get_params(self):
		wd_params, nowd_params = [], []
		for name, module in self.named_modules():
			if isinstance(module, (nn.Linear, nn.Conv2d)):
				wd_params.append(module.weight)
				if not module.bias is None:
					nowd_params.append(module.bias)
			elif isinstance(module, nn.BatchNorm2d):
				nowd_params += list(module.parameters())
		return wd_params, nowd_params



class CSGCNetOutput(nn.Module):
	def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
		super(CSGCNetOutput, self).__init__()
		self.conv = ConvBNReLU(in_chan, mid_chan, kernel_size=3, stride=1, padding=1)
		self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
		self.init_weight()

	def forward(self, x):
		x = self.conv(x)
		x = self.conv_out(x)
		return x

	def init_weight(self):
		for ly in self.children():
			if isinstance(ly, nn.Conv2d):
				nn.init.kaiming_normal_(ly.weight, a=1)
				if not ly.bias is None: nn.init.constant_(ly.bias, 0)

	def get_params(self):
		wd_params, nowd_params = [], []
		for name, module in self.named_modules():
			if isinstance(module, (nn.Linear, nn.Conv2d)):
				wd_params.append(module.weight)
				if not module.bias is None:
					nowd_params.append(module.bias)
			elif isinstance(module, nn.BatchNorm2d):
				nowd_params += list(module.parameters())
		return wd_params, nowd_params


class AttentionFusion(nn.Module):
	def __init__(self, input_channels, output_channels, *args, **kwargs):
		super(AttentionFusion, self).__init__()
		self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
		self.init_weight()

	def forward(self, features1, features2):
		feat_concat = torch.cat([features1, features2], dim=1)
		feat_new = self.conv(feat_concat)
		return feat_new

	def init_weight(self):
		for ly in self.children():
			if isinstance(ly, nn.Conv2d):
				nn.init.kaiming_normal_(ly.weight, a=1)
				if not ly.bias is None: nn.init.constant_(ly.bias, 0)

	def get_params(self):
		wd_params, nowd_params = [], []
		for name, module in self.named_modules():
			if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
				wd_params.append(module.weight)
				if not module.bias is None:
					nowd_params.append(module.bias)
			elif isinstance(module, nn.BatchNorm2d):
				nowd_params += list(module.parameters())
		return wd_params, nowd_params


class CSGCNet(nn.Module):
	def __init__(self, n_classes, backbone_weights=None, pretrained=True,*args, **kwargs):
		super(CSGCNet, self).__init__()
		self.mobile = MobileNetV3(pretrained=pretrained, width_mult=1., weights=backbone_weights)
		self.ab = AttentionBranch(576, 256, 256, n_classes)
		self.sb = SpatialBranch()
		self.ffm = FeatureFusionModule(384, 256)
		self.conv_out = CSGCNetOutput(256, 256, n_classes)
		self.init_weight()

	def forward(self, x):
		H, W = x.size()[2:]
		feat_sb = self.sb(x)
		mobile_feat = self.mobile(x)

		feat_ab, feat_ab_final = self.ab(mobile_feat)
		
		feat_ab = F.interpolate(feat_ab, (feat_sb.size()[2:]), mode='bilinear', align_corners=True)
		feat_ab_final = F.interpolate(feat_ab_final, (feat_sb.size()[2:]), mode='bilinear', align_corners=True)

		feat_fuse = self.ffm(feat_sb, feat_ab)
		feat_out = self.conv_out(feat_fuse)

		feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
		feat_con = F.interpolate(feat_ab_final, (H, W), mode='bilinear', align_corners=True)
		return feat_out, feat_con

	def init_weight(self):
		for ly in self.children():
			if isinstance(ly, nn.Conv2d):
				nn.init.kaiming_normal_(ly.weight, a=1)
				if not ly.bias is None: nn.init.constant_(ly.bias, 0)

	def get_params(self):
		wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
		for name, child in self.named_children():
			child_wd_params, child_nowd_params = child.get_params()
			if isinstance(child, (FeatureFusionModule, CSGCNetOutput)):
				lr_mul_wd_params += child_wd_params
				lr_mul_nowd_params += child_nowd_params
			else:
				wd_params += child_wd_params
				nowd_params += child_nowd_params
		return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


class SpatialGCN(nn.Module):		# 空间GCN
	def __init__(self, plane, dilation):
		super(SpatialGCN, self).__init__()
		inter_plane = plane // 2  #//代表除法向下取整
		#  // c c   self
		self.node_k = nn.Conv2d(plane, inter_plane, kernel_size=1,groups=inter_plane)
		self.node_v = nn.Conv2d(plane, inter_plane, kernel_size=1,groups=inter_plane)
		self.node_q = nn.Conv2d(plane, inter_plane, kernel_size=1,groups=inter_plane)


		self.conv_wg = nn.Conv1d(inter_plane, inter_plane, kernel_size=1, bias=False , groups=inter_plane,dilation=dilation)
		#self.bn_wg = BatchNorm1d(inter_plane)
		self.bn_wg = torch.nn.GroupNorm(inter_plane, inter_plane, eps=1e-05, affine=True, device=None, dtype=None)
		self.softmax = nn.Softmax(dim=2)

		self.out = nn.Sequential(nn.Conv2d(inter_plane, plane, kernel_size=1),
								 BatchNorm2d(plane))

	def forward(self, x):
		# b, c, h, w = x.size()
		node_k = self.node_k(x)#torch.Size([4, 256, 17, 17])
		node_v = self.node_v(x)#torch.Size([4, 256, 17, 17])
		node_q = self.node_q(x)#torch.Size([4, 256, 17, 17])

		b,c,h,w = node_k.size() #size()函数主要是用来统计矩阵元素个数，或矩阵某一维上的元素个数的函数
		node_k = node_k.view(b, c, -1).permute(0, 2, 1)#permute 将tensor的维度换位
		node_q = node_q.view(b, c, -1)
		node_v = node_v.view(b, c, -1)#.permute(0, 2, 1)
		# A = k * q
		# AV = k * q * v
		# AVW = k *(q *v) * w
		AV = torch.bmm(node_k,node_q)# torch.bmm()是tensor中的一个相乘操作，类似于矩阵中的A*B。torch.Size([4, 256, 256])
		AV = self.softmax(AV)
		#print(AV.size())
		AV = torch.bmm(node_v, AV)
		AVW = self.conv_wg(AV)
		AVW = self.bn_wg(AVW)
		AVW = AVW.view(b, c, h, -1)
		AVW = AVW.permute(0, 1, 3, 2)
		#print(AVW.size())
		AVW = AVW.reshape(b, c, -1)
		AVW = self.conv_wg(AVW)
		AVW = self.bn_wg(AVW)
		AVW = AVW.view(b, c, w, -1)
		AVW = AVW.permute(0, 1, 3, 2)
		out = F.relu_(self.out(AVW)+x)
		return out


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim,group = 1):
        super(CrissCrossAttention,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1,groups=group)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1,groups=group)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1,groups=group)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))
        self.conv_1d = nn.Conv1d(in_dim,in_dim, kernel_size=1, bias=False,groups=group,dilation=5)
        self.bn_1d = BatchNorm1d(in_dim)
    def forward(self, x):
        m_batchsize, c , height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        out = self.gamma*(out_H + out_W)
        out =out.reshape(m_batchsize,c,-1)
        out = self.conv_1d(out)
        out = self.bn_1d(out)
        out = out.reshape(m_batchsize,c,height,width)
        out = F.relu_(out+x)

        return out
def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    # transpose
    # - contiguous() required if transpose() is used before view().
    #   See https://github.com/pytorch/pytorch/issues/764
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


if __name__ == "__main__":
	path = Path("pretrained_backbones")
	weights_path = (path / "mobilenetv3-small-55df8e1f.pth").resolve()
	net = CSGCNet(6, backbone_weights=weights_path)
	net.eval()
	in_ten = torch.randn(1, 3, 512, 512)
	start = time.time()
	out, out1 = net(in_ten)
	print(out.shape)
	print(out1.shape)
	# print(net.mobile.features[:4])
	# end = time.time()
	# print(out.shape)
	# print('TIME in ms: ', (end - start)*1000)