##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 整个网络的结构
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


import numpy as np
import torch.nn as nn
import Modules.Blockmodule as bm
import torch
import pdb
import Modules.channel_attention as channel
import Modules.grid_attention as grid
import Modules.scale1 as scale
import Modules.non_local as sa1
import Modules.modules as modules
from  Modules.HRViT import HRViT as HRtransformer
import Modules.HRViT as HR
import ml_collections
from dcnv3 import DCNv3 as DCNTransformer
# 二维切片的参数，2s张和单张的参数
params = {'num_filters': 64,
            'kernel_h': 5,
            'kernel_w': 5,
            'kernel_c': 1,
            'stride_conv': 1,
            'pool': 2,
            'stride_pool': 2,
          # Valid options : NONE, CSE, SSE, CSSE
            'se_block': "CSSE",
            'drop_out': 0.1}


# transformer下采样
def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config
def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config

class HCTN(nn.Module):
    def __init__(self, out_channels=1, num_slices=5, se_loss=True,**kwargs):

        super(HCTN, self).__init__()

        # 三个双门控空间注意力
        self.attentionblock1 = grid.MultiAttentionBlock(
            in_size=64, gate_size=64, inter_size=64,
            nonlocal_mode='concatenation',
            sub_sample_factor=(1, 1))
        self.attentionblock2 = grid.MultiAttentionBlock(
            in_size=64, gate_size=64, inter_size=64,
            nonlocal_mode='concatenation',
            sub_sample_factor=(1, 1))
        self.attentionblock3 = grid.MultiAttentionBlock(
            in_size=64, gate_size=64, inter_size=64,
            nonlocal_mode='concatenation',
            sub_sample_factor=(1, 1))

        # 单个non-local注意力
        self.sa1 = sa1.NLCA(in_channels=64, inter_channels=64 // 4)
        # self.sa1 = sa1.NonLocal2d_nowd(in_channels=64, inter_channels=64 // 4)

        # 四个编码与解码的拼接，注意这是解码的低层维度上采样之后与编码的特征拼接
        self.up_contact4 = modules.UpCat(64, 64, is_deconv=True)
        self.up_contact3 = modules.UpCat(64, 64, is_deconv=True)
        self.up_contact2 = modules.UpCat(64, 64, is_deconv=True)
        self.up_contact1 = modules.UpCat(64, 64, is_deconv=True)

        # 四个通道注意力
        self.CA4 = channel.RCA(64, 64, drop_out=True)
        self.CA3 = channel.RCA(64, 64)
        self.CA2 = channel.RCA(64, 64)
        self.CA1 = channel.RCA(64, 64)

        # 四个尺度注意力之前的四个不同尺度特征上采样
        self.dsv4 = modules.UnetDsv3(64, 16, scale_factor=(256, 256))
        self.dsv3 = modules.UnetDsv3(64, 16, scale_factor=(256, 256))
        self.dsv2 = modules.UnetDsv3(64, 16, scale_factor=(256, 256))
        self.dsv1 = modules.UnetDsv3(64, 16, scale_factor=(256, 256))

        # 一个尺度注意力
        self.scale_att = scale.scale_atten_convblock(64, 4)

        self.conv0 = DCNTransformer(channels=128)
        # self.conv1 = HR.MixConv2d(in_channels=128, out_channels=64, kernel_size=5, padding=2, stride=1,groups=64)
        # self.conv2 = HR.MixConv2d(in_channels=128, out_channels=64, kernel_size=5, padding=2, stride=1,groups=64)
        # self.conv3 = HR.MixConv2d(in_channels=128, out_channels=64, kernel_size=5, padding=2, stride=1,groups=64)
        # self.conv4 = HR.MixConv2d(in_channels=128, out_channels=64, kernel_size=5, padding=2, stride=1,groups=64)
        self.conv1 = DCNTransformer(channels=128)
        self.conv2 = DCNTransformer(channels=128)
        self.conv3 = DCNTransformer(channels=128)
        self.conv4 = DCNTransformer(channels=128)

        #
        self.conv12 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(5,5),padding=(2,2),stride=1)
        self.conv22 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(5,5),padding=(2,2),stride=1)
        self.conv32 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(5,5),padding=(2,2),stride=1)
        self.conv42 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(5,5),padding=(2,2),stride=1)
        self.conv52 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=(5,5),padding=(2,2),stride=1)

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(64, out_channels, 1))  # label output

        self.transformer = HRtransformer(in_channels=3,stride=4,channels=64,
        channel_list=(
            (32,),
            (32, 64),
            (32, 64, 128),
            (32, 64, 128),
        ),
        block_list=(
            (1,),
            (1, 1),
            (1, 1, 6),
            (1, 1, 6),
        ),
        dim_head=32,
        ws_list=(1, 2, 7),
        proj_dropout=0.0,
        # drop_path_rate=0.1,
        mlp_ratio_list=(4, 4, 4),
        dropout=0.0,
        # head_dropout=0.1,
        **kwargs,)



    def forward(self, input2D):
        """
        :param inputs: X
       :return: probabiliy map
        """
        skips = []
        x = input2D
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        trans_feature,conv_features = self.transformer(x)

        fusion12 = conv_features[0]
        fusion22 = conv_features[1]
        fusion32 = trans_feature[0]
        fusion42 = trans_feature[1]
        fusion52 = trans_feature[2]

        fusion12 = self.conv12(fusion12)
        fusion22 = self.conv22(fusion22)
        fusion32 = self.conv32(fusion32)
        fusion42 = self.conv42(fusion42)
        fusion52 = self.conv52(fusion52)

        # todo 完成编码器部分的改造
        #fusion52 （64,16,16）
        #fusion42 （64,32,32）
        #fusion32 （64,64,64）
        #fusion22 （64,128,128）
        #fusion12 （64,256,256）

        # image decoder

        up4 = self.up_contact4(fusion42, fusion52)
        up4 = self.conv4(up4)
        g_conv4 = self.sa1(up4)

        up4, attw4 = self.CA4(g_conv4)
        g_conv3, att3 = self.attentionblock3(fusion32, up4)
        up3 = self.up_contact3(g_conv3, up4)

        up3 = self.conv3(up3)
        up3, attw3 = self.CA3(up3)
        g_conv2, att2 = self.attentionblock2(fusion22, up3)
        up2 = self.up_contact2(g_conv2, up3)

        up2 = self.conv2(up2)
        up2, attw2 = self.CA2(up2)
        g_conv1, att1 = self.attentionblock1(fusion12, up2)
        up1 = self.up_contact1(g_conv1, up2)

        up1 = self.conv0(up1)
        up1, attw1 = self.CA1(up1)


        # 将不同尺度特征进行上采样，上采样的尺度特征与输入相同
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)

        # 拼接四个上采样后的特征
        dsv_cat = torch.cat([dsv1, dsv2, dsv3, dsv4], dim=1)
        # 进行尺度注意力运算
        outimage = self.scale_att(dsv_cat)
        # 最后经过卷积获取输出结果

        out_label = self.conv6(outimage)  # n 28 256 256
        # 返回最终结果
        return out_label
def get_network():
    return HCTN