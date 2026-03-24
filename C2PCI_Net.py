
import argparse
import torch.nn as nn
import torch
import math
from torch import nn
import torch.nn.functional as F
import scipy.io
from scipy.io import loadmat
import numpy as np
import torch.nn.init as init

from DataLoader import DataLoader
from config import Config
args = Config(dataset_name='DEAP')


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
#有挺多模块我看后面没有用到，但是原作者保留，我也没删，有数字编号的是用到的
#1.SE模块
#2.SDLN，对psd拓扑图处理
#3.位置编码
#4.eeg统计特征编码
#5.外周特征编码
#6.多头注意力融合
#7.总体：首先psd图用SDLN做处理，peri用peri backbone，eeg统计用eegbackbone，然后进入HF_ICMA
# 输出（bs，512×3）为x_s1，这个输出经训练头得到z_t，然后经过stage2_module得到x_s2,
# x_s2再经投影头得到z_c，x_s2经情感头得到e。总共模型输出者五个量。

#权重初始化设置
def weights_init(m):
    if isinstance(m,nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif isinstance(m,nn.Linear):
        nn.init.normal_(m.weight,0.0,0.01)
        if m.bias != None:
            nn.init.zeros_(m.bias)    

def custom_weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
#全连接神经网络MLP
class FC_Backbone(nn.Module):
    def __init__(self, input_size, layer_sizes):
        super(FC_Backbone, self).__init__()
        layers = []
        layer_sizes = [input_size] + layer_sizes
        for l_id in range(len(layer_sizes) - 1):
            if l_id == len(layer_sizes) - 2:
                layers.append(nn.Sequential(
                    #nn.BatchNorm1d(num_features=layer_sizes[l_id], affine=False),
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id+1]),
                ))
            else:
                layers.append(nn.Sequential(
                    nn.Linear(layer_sizes[l_id], layer_sizes[l_id+1]),
                    nn.Sigmoid(),
                    #nn.BatchNorm1d(num_features=layer_sizes[l_id+1], affine=False),
                ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def init_weights(self):
        for layer in self.layers:
            for m in layer.modules():
                weights_init(m)

#卷积
class BasicConv2d(nn.Module):  
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x
#1.通道注意力（Squeeze&excitation）：先全局平均池化展平再压缩，后面卷积会用到
class SE_Module(nn.Module):
    def __init__(self):
        super(SE_Module, self).__init__()
        self.features = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 16),
            nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Linear(16, 64), 
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        output = x.view(x.size(0), 64, 1, 1)
        return output
#2.用于对psd拓扑图进行卷积处理
class SDLN(nn.Module):
    def __init__(self): 
        super(SDLN, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(5, 16, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(16, affine=True),
            BasicConv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicConv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
        )
        self.SE_Module = SE_Module()
        self.features_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            # nn.Linear(64 * 4 * 4, 1024),
            # nn.ReLU(),
            # nn.Dropout(0.4),
            nn.Linear(1024, 256 * 2 * 2),
        )

    def forward(self, x):
        x = self.features(x)
        weight = self.SE_Module(x)
        x = weight * x
        x = self.features_2(x)
        x = x.view(-1, 256, 2, 2)  #  [batch_size, 256, 2, 2]
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

#3.位置编码，按照公式写的，后面eeg统计特征和peri特征都会用到        
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
    
#4.eeg统计特征编码，backbone_hidden=256,送入transformer（LN-8头注意力-dropout=0.1-残差连接
#-LN-FFN-dropout=0.1-残差连接-输出）
class Channel_Wise_TransformerEncoder_Backbon_eeg_stat(nn.Module):
    def __init__(self,args):
        super().__init__()  # Change this line

        self.channel_feat_dim=args.eeg_stat_channel_feat_dim#7
        self.num_channel=args.EEG_channel#32

        if args.positional_encoding:
            self.positional_encoding=PositionalEncoding(d_model=args.backbone_hidden,dropout=0)
        else:
            self.positional_encoding=None
        self.attn_layer=nn.TransformerEncoderLayer(d_model=args.backbone_hidden,nhead=8,dim_feedforward=1024,batch_first=True)
        self.encoder=nn.TransformerEncoder(encoder_layer=self.attn_layer,num_layers=args.num_layer)
        self.dim_up=nn.Linear(self.channel_feat_dim,args.backbone_hidden)
        # self.dim_down = None


        if args.fusion_mothod == 'DFAF':
            self.dim_down = None
        elif args.fusion_mothod == 'ATTEN':
            self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)
        elif args.fusion_mothod == 'HF_ICMA':
            self.dim_down = None
        elif args.fusion_mothod == 'MS-MDA':
            self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)
        elif args.fusion_mothod == 'DCCA':
            self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)
        elif args.fusion_mothod == 'RGNN':
            self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)
        elif args.fusion_mothod == 'HKE_model':
            self.dim_down = None
        elif args.fusion == 'AITST':
            self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)
        elif args.fusion == 'CGRU-MDGN':
            self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)
        elif args.fusion == 'MS-ERM':
            self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)
        else:
            self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)

    def forward(self,x):
          
            assert len(x.shape)==3 and x.shape[1]==self.num_channel and x.shape[2]==self.channel_feat_dim
            x=self.dim_up(x) 
            if self.positional_encoding!=None:
                x=self.positional_encoding(x)
            x=self.encoder(x) 

            # if use HF_ICMA:
            if self.dim_down!=None:
                x=self.dim_down(x.view(x.shape[0],-1)) # (bs,fusion_hidden)
            
            # if use concat:
            # if hasattr(self, 'dim_down') and self.dim_down is not None:
            #     x = self.dim_down(x.view(x.shape[0], -1))  # (bs, fusion_hidden)
    
            return x

#5.外周特征编码，backbone_hidden=256,送入transformer（LN-8头注意力-dropout=0.1-残差连接
#-LN-FFN-dropout=0.1-残差连接-输出），还做了自相关  
class Channel_Wise_TransformerEncoder_Backbon_peri(nn.Module):
    def __init__(self, args):
        super(Channel_Wise_TransformerEncoder_Backbon_peri, self).__init__()
        self.channel_feat_dim = args.eeg_peri_channel_feat_dim  # 1
        self.num_channel = args.peri_feat_dim#55

        if args.positional_encoding:
            self.positional_encoding = PositionalEncoding(d_model=args.backbone_hidden, dropout=0)
        else:
            self.positional_encoding = None
        self.attn_layer = nn.TransformerEncoderLayer(d_model=args.backbone_hidden, nhead=8, dim_feedforward=1024, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=self.attn_layer, num_layers=args.num_layer)
        self.dim_up = nn.Linear(self.channel_feat_dim, args.backbone_hidden)
        self.dim_up1 = nn.Linear(self.num_channel, args.backbone_hidden)

        if args.fusion_mothod == 'DFAF':
            self.dim_down = None
        elif args.fusion_mothod == 'ATTEN':
            self.dim_down = nn.Linear(self.num_channel * args.backbone_hidden, args.backbone_hidden)
        elif args.fusion_mothod == 'HF_ICMA':
            self.dim_down = None
        elif args.fusion_mothod == 'MS-MDA':
            self.dim_down = nn.Linear(self.num_channel * args.backbone_hidden, args.backbone_hidden)
        elif args.fusion_mothod == 'DCCA':
            self.dim_down = nn.Linear(self.num_channel * args.backbone_hidden, args.backbone_hidden)
        elif args.fusion_mothod == 'RGNN':
            self.dim_down = nn.Linear(self.num_channel * args.backbone_hidden, args.backbone_hidden)
        elif args.fusion_mothod == 'HKE_model':
            self.dim_down = None
        elif args.fusion_mothod == 'AITST':
            self.dim_down = nn.Linear(self.num_channel * args.backbone_hidden, args.backbone_hidden)
        elif args.fusion_mothod == 'CGRU-MDGN':
            self.dim_down = nn.Linear(self.num_channel * args.backbone_hidden, args.backbone_hidden)
        elif args.fusion_mothod == 'MS-ERM':
            self.dim_down = nn.Linear(self.num_channel * args.backbone_hidden, args.backbone_hidden)
        else:
            self.dim_down = nn.Linear(self.num_channel * args.backbone_hidden, args.backbone_hidden)

    def forward(self, x):
        assert len(x.shape) == 3 and x.shape[1] == self.num_channel and x.shape[2] == self.channel_feat_dim
        x1 = self.dim_up(x) 
        if self.positional_encoding is not None:
            x1 = self.positional_encoding(x1)
        x1 = self.encoder(x1)  #transformer处理

        if self.dim_down is not None:
            x1 = self.dim_down(x1.view(x1.shape[0], -1))  # (bs,fusion_hidden)

        # Parallel multiplication operation
        x2 = x
        x2 = torch.bmm(x2, x2.transpose(1, 2))  # x2 * x2.T
        x2 = torch.bmm(x2, x2)  # x2 * (x2 * x2.T)
        x2 = self.dim_up1(x2)  # Ensure x2 has the same dimensions as x1 after processing

        if self.positional_encoding is not None:
            x2 = self.positional_encoding(x2)
        x2 = self.encoder(x2)#transformer处理

        if self.dim_down is not None:
            x2 = self.dim_down(x2.view(x2.shape[0], -1))

        # Combine x1 and x2
        x = x1 + x2

        return x
#交叉网络
class CrossNet(torch.nn.Module):
    def __init__(self, in_features, layer_num, device, batch_size):
        super(CrossNet, self).__init__()
        self.layer_num = layer_num
        self.batch = batch_size
        self.in_features = in_features
        self.kernels = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_features, 1))) for i in range(self.layer_num)])
        self.bias = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(in_features, 1))) for i in range(self.layer_num)])
        self.to(device)
        # self.linear = torch.nn.Linear(in_features, 10, bias=True)
        self.linear = torch.nn.Linear(in_features, 1, bias=True)
        self.Sigmoid = torch.nn.Sigmoid()

    def forward(self, inputs):
        # x_0 = inputs.unsqueeze(2).float()
        # print('input的size:{}'.format(x_0.shape))
        # x_0 = inputs.unsqueeze(2).float()
        x_0 = inputs.float()
        x_l = x_0
        # print('input的size:{}'.format(x_l.shape))
        # print(self.kernels[1].shape)
        # print(x_l)
        # print(x_l.shape)
        for i in range(self.layer_num):
            xl_w = torch.tensordot(x_l, self.kernels[i], dims=([1], [0]))
            dot_ = torch.matmul(x_0, xl_w)
            x_l = dot_ + self.bias[i] + x_l
            x_l = self.Sigmoid(x_l)
        x_l = torch.squeeze(x_l, dim=2)
        # print('output的size:{}'.format(x_l.shape))
        # output = x_l
        output = self.linear(x_l)
        # print('output的size:{}'.format(x_l.shape))
        # print(output)
        return output
#注意力融合
class AttentionFusion(nn.Module):
    def __init__(self, fused_dim):
        super(AttentionFusion, self).__init__()
        self.fused_dim = fused_dim

        self.attention_weights = nn.Parameter(torch.randn(self.fused_dim, requires_grad=True))

    def forward(self, x1, x2):
        # calculate weigths for all input samples
        row, _ = x1.shape
        fused_tensor = torch.empty_like(x1)
        alpha = []
        for i in range(row):
            tmp1 = torch.dot(x1[i,:], self.attention_weights) # scalar
            tmp2 = torch.dot(x2[i,:], self.attention_weights) # scalar
            alpha_1 = torch.exp(tmp1) / (torch.exp(tmp1) + torch.exp(tmp2))
            alpha_2 = 1 - alpha_1
            alpha.append((alpha_1.detach().cpu().numpy(), alpha_2.detach().cpu().numpy()))
            fused_tensor[i, :] = alpha_1 * x1[i,:] + alpha_2 * x2[i, :]
        return fused_tensor, alpha

    def init_weights(self):
        nn.init.normal_(self.attention_weights,0.0,0.01)

#6.多头注意力融合
class HF_ICMA(nn.Module):
    def __init__(self,args):
        super(HF_ICMA, self).__init__()

        # self.eeg_feature_map_dim_change=nn.Linear(512,args.fusion_hidden) #  (512,256) !!!!! exactly 512 !!!!!
        self.eeg_feature_map_dim_change=nn.Linear(256,args.fusion_hidden)
        self.eeg_feature_dim_change=nn.Linear(args.backbone_hidden,args.fusion_hidden) #  (256,512)
        self.peri_dim_change=nn.Linear(args.backbone_hidden,args.fusion_hidden) #  (256,512)
        #512维，8头，batch_first=Ture
        self.eeg_feature_map_attn=nn.MultiheadAttention(embed_dim=args.fusion_hidden,num_heads=8,batch_first=True)
        self.eeg_feature_attn1=nn.MultiheadAttention(embed_dim=args.fusion_hidden,num_heads=8,batch_first=True)
        self.eeg_feature_attn2=nn.MultiheadAttention(embed_dim=args.fusion_hidden,num_heads=8,batch_first=True)
        self.peri_attn=nn.MultiheadAttention(embed_dim=args.fusion_hidden,num_heads=8,batch_first=True)
        #降维，统一到512
        self.eeg_feature_map_dim_down=nn.Linear(4*args.fusion_hidden,args.fusion_hidden)
        self.eeg_feature_dim_down=nn.Linear(args.EEG_channel*args.fusion_hidden,args.fusion_hidden)
        self.peri_dim_down=nn.Linear(args.peri_feat_dim*args.fusion_hidden,args.fusion_hidden)

        # # 
        # self.classifier = nn.Sequential(
        #     nn.Linear(3*args.fusion_hidden, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.4),
        #     nn.Linear(512,2),
        #     nn.Softmax(dim=1)
        # )

    def forward(self,eeg_map,eeg,peri):
        
        eeg_map=eeg_map.view(eeg_map.shape[0],eeg_map.shape[1],-1)
        eeg_map=eeg_map.transpose(1,2) 

        eeg_map=self.eeg_feature_map_dim_change(eeg_map)#（bs，4,512）
        eeg=self.eeg_feature_dim_change(eeg) #（bs，7,512）
        peri=self.peri_dim_change(peri) #（bs，55,512）
        #eeg查eeg map
        eeg_map_attn_out,_=self.eeg_feature_map_attn(eeg_map,eeg,eeg) 
        #eeg map查eeg
        eeg_attn1_out,_=self.eeg_feature_attn1(eeg,eeg_map,eeg_map) 
        #外周查eeg（attn1）
        eeg_attn2_out,cross_model_aeeention_value=self.eeg_feature_attn2(eeg_attn1_out,peri,peri) 

        # residual
        eeg_attn2_out=eeg_attn2_out+eeg_attn1_out
        #eeg attn1 查外周
        peri_attn_out,_=self.peri_attn(peri,eeg_attn1_out,eeg_attn1_out)
        # residual
        peri_attn_out=peri_attn_out+peri

        eeg_map_attn_out=eeg_map_attn_out.contiguous().view(eeg_map_attn_out.shape[0],-1)

        eeg_map_out=self.eeg_feature_map_dim_down(eeg_map_attn_out) # (bs,h)

        eeg_attn2_out=eeg_attn2_out.contiguous().view(eeg_attn2_out.shape[0],-1)
        eeg_out=self.eeg_feature_dim_down(eeg_attn2_out) # (bs,h)

        peri_attn_out=peri_attn_out.contiguous().view(peri_attn_out.shape[0],-1)
        peri_out=self.peri_dim_down(peri_attn_out) # (bs,h)

        fused_tensor=torch.cat([eeg_map_out,eeg_out,peri_out],dim=-1) # (bs,3h)  h=512

        return fused_tensor,cross_model_aeeention_value



    def init_weights(self):
        def init_weights(self):
            init.xavier_uniform_(self.eeg_feature_map_attn.weight)
            init.xavier_uniform_(self.eeg_feature_attn1.weight)
            init.xavier_uniform_(self.eeg_feature_attn2.weight)
            init.xavier_uniform_(self.peri_attn.weight)

            init.xavier_uniform_(self.eeg_feature_map_dim_change.weight)
            init.xavier_uniform_(self.eeg_feature_dim_change.weight)
            init.xavier_uniform_(self.peri_dim_change.weight)

            init.xavier_uniform_(self.eeg_feature_map_dim_down.weight)
            init.xavier_uniform_(self.eeg_feature_dim_down.weight)
            init.xavier_uniform_(self.peri_dim_down.weight)

    # def init_weights(self):
    #     def init_weights(self):
    #         init.xavier_normal_(self.eeg_feature_map_dim_change.weight)
    #         init.xavier_normal_(self.eeg_feature_dim_change.weight)
    #         init.xavier_normal_(self.peri_dim_change.weight)

    #         init.xavier_normal_(self.eeg_feature_map_attn.in_proj_weight)
    #         init.xavier_normal_(self.eeg_feature_attn1.in_proj_weight)
    #         init.xavier_normal_(self.eeg_feature_attn2.in_proj_weight)

    #         init.xavier_normal_(self.peri_attn.in_proj_weight)
    #         init.xavier_normal_(self.eeg_feature_map_dim_down.weight)
    #         init.xavier_normal_(self.eeg_feature_dim_down.weight)
    #         init.xavier_normal_(self.peri_dim_down.weight)
#7.总体：首先psd图用SDLN做处理，peri用peri backbone，eeg统计用eegbackbone，然后进入HF_ICMA
# 输出（bs，512×3）为x_s1，这个输出经训练头得到z_t，然后经过stage2_module得到x_s2,
# x_s2再经投影头得到z_c，x_s2经情感头得到e。总共模型输出者五个量。
class Fusion_Model(nn.Module):
    def __init__(self,args):
        super(Fusion_Model, self).__init__()
        self.args=args
        self.hf_icma = HF_ICMA(args)
        hidden=args.hidden
        self.switch=args.backbone_switch
        self.fusion=args.fusion_mothod
        if args.backbone_switch!=[1,1,1]:
            #
            if args.fusion_mothod == 'DFAF':
                self.dim_down = None
                merge_dim = args.fusion_hidden
                self.merge_dim=merge_dim
            elif args.fusion_mothod == 'ATTEN':
                self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)
                merge_dim = args.fusion_hidden
                self.merge_dim=merge_dim
            elif args.fusion_mothod == 'HF_ICMA':
                self.dim_down = None
                merge_dim = args.fusion_hidden
                self.merge_dim=merge_dim
            elif args.fusion_mothod == 'MS-MDA':
                self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)
                merge_dim = args.fusion_hidden
                self.merge_dim=merge_dim
            elif args.fusion_mothod == 'DCCA':
                self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)
                merge_dim = args.fusion_hidden
                self.merge_dim=merge_dim
            elif args.fusion_mothod == 'RGNN':
                self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)
                merge_dim = args.fusion_hidden
                self.merge_dim=merge_dim
            elif args.fusion_mothod == 'HKE_model':
                self.dim_down = None
                merge_dim = args.fusion_hidden
                self.merge_dim=merge_dim
            elif args.fusion_mothod == 'AITST':
                self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)
                merge_dim = args.fusion_hidden
                self.merge_dim=merge_dim
            elif args.fusion_mothod == 'CGRU-MDGN':
                self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)
                merge_dim = args.fusion_hidden
                self.merge_dim=merge_dim
            elif args.fusion_mothod == 'MS-ERM':
                self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)
                merge_dim = args.fusion_hidden
                self.merge_dim=merge_dim
            else:
                self.dim_down = nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)
                merge_dim = args.fusion_hidden*3
                self.merge_dim=merge_dim
        #     self.fusion=args.fusion_mothod
        # else:
        #     self.fusion=args.fusion_mothod

        self.device=torch.device(args.device)       

        self.PSD_map_backbone = SDLN() 
        self.peri_backbone = Channel_Wise_TransformerEncoder_Backbon_peri(args=args)
        self.eeg_backbone = Channel_Wise_TransformerEncoder_Backbon_eeg_stat(args=args)

        # merge_dim=sum(self.switch)*args.backbone_hidden

        merge_dim = args.fusion_hidden*3
        self.merge_dim=merge_dim

        self.stage2_module=nn.Sequential(
            nn.Linear(merge_dim,hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden,hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden,hidden),
        )

        self.proj_head=nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), \
                                     nn.Linear(hidden, hidden))
        self.trial_head=nn.Sequential(nn.Linear(merge_dim, hidden), nn.ReLU(inplace=True), \
                                      nn.Linear(hidden, args.trial_cnt))
        
        if args.dataset_name == 'DEAP':
            self.emo_head=nn.Sequential(nn.Linear(hidden,hidden),nn.Linear(hidden,2),\
                                        nn.Sigmoid())
        elif args.dataset_name == 'HCI':
            self.emo_head=nn.Sequential(nn.Linear(hidden, hidden), nn.Linear(hidden,2),\
                                        nn.Sigmoid())
        elif args.dataset_name == 'SEED-IV':
            self.emo_head=nn.Sequential(nn.Linear(hidden, hidden), \
                                        nn.ReLU(inplace=True), \
                                    nn.Linear(hidden, 4))
        elif args.dataset_name == 'SEED-V':
            self.emo_head=nn.Sequential(nn.Linear(hidden, hidden), \
                                        nn.ReLU(inplace=True), \
                                    nn.Linear(hidden, 5))
        else:
            self.emo_head=nn.Sequential(nn.Linear(hidden, hidden), \
                                        nn.ReLU(inplace=True), \
                                    nn.Linear(hidden, args.emo_categories))


    def forward(self, eeg_map_data, eeg_stat_data, peri_data):

        eeg_map_data = eeg_map_data.float()
        eeg_stat_data = eeg_stat_data.float()
        peri_data = peri_data.float()

        # save the output of the first Conv1
         
        # eeg_map_out_conv1 = self.PSD_map_backbone.features[0](eeg_map_data)
        # eeg_map_out_conv1 = self.PSD_map_backbone.features[0](eeg_map_data.float())


        # 
        eeg_map_out = self.PSD_map_backbone(eeg_map_data)
        stat_out = self.eeg_backbone(eeg_stat_data)
        peri_out = self.peri_backbone(peri_data)


        # 

        # if self.fusion_mothod == 'DFAF':
        #         eeg_map_out = eeg_map.view(eeg_map.shape[0],eeg_map.shape[1],-1) #(bs,512,4)
        #         eeg_map=eeg_map.view(eeg_map.shape[0],eeg_map.shape[1],-1) #(bs,512,4)
        #         eeg_map=eeg_map.transpose(1,2) # (bs,4,512)
    
        # 
        fused_x, cross_model_attention_value = self.hf_icma(eeg_map_out, stat_out, peri_out)

        
        # eeg_map_out=eeg_map_out.view(eeg_map_out.shape[0],eeg_map_out.shape[1],-1) #(bs,512,4)
        # eeg_map_out=eeg_map_out.transpose(1,2) # (bs,4,512)
        # x_s1 = torch.concat([eeg_map_out, stat_out, peri_out], dim=2) # x_s1 shape is (bs,hidden*3)

        # gain the output of other model
        x_s1 = fused_x
        z_t = self.trial_head(x_s1)
        x_s2 = self.stage2_module(x_s1)
        z_c = self.proj_head(x_s2)
        e = self.emo_head(x_s2)

        return x_s1, z_t, x_s2, z_c, e
        # return eeg_map_out_conv1, x_s1, z_t, z_c, e 
    
print('e has been load !!!  -----> e 已经加载完毕！！！！！！！！ ')
