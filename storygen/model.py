import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from .config import cfg
from torch.autograd import Variable
#from .recurrent import BertEncoderWithMemory, BertEmbeddings, NonRecurTransformer, BertEncoderWithMemoryForTree
from easydict import EasyDict as edict
#from .layers import DynamicFilterLayer1D as DynamicFilterLayer
#from .GLAttention import GLAttentionGeneral as ATT_NET
#from .cross_attention import LxmertCrossAttentionLayer as CrossAttn
from torchvision import models
import numpy as np
import os
from copy import copy


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True))
    return block

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

# remind me of what the configs are
base_config = edict(
    hidden_size=768,
    vocab_size=None,  # get from word2idx
    video_feature_size=2048,
    max_position_embeddings=None,  # get from max_seq_len
    max_v_len=100,  # max length of the videos
    max_t_len=30,  # max length of the text
    n_memory_cells=10,  # memory size will be (n_memory_cells, D)
    type_vocab_size=2,
    layer_norm_eps=1e-12,  # bert layernorm
    hidden_dropout_prob=0.1,  # applies everywhere except attention
    num_hidden_layers=2,  # number of transformer layers
    attention_probs_dropout_prob=0.1,  # applies only to self attention
    intermediate_size=768,  # after each self attention
    num_attention_heads=12,
    memory_dropout_prob=0.1
)

if torch.cuda.is_available():
    T = torch.cuda
else:
    T = torch
    
class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION * cfg.VIDEO_LEN
        self.c_dim = cfg.GAN.CONDITION_DIM
        self.fc = nn.Linear(self.t_dim, self.c_dim * 2, bias=True)
        self.relu = nn.ReLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.c_dim]
        logvar = x[:, self.c_dim:]
        print("mu:" + mu + "\n")
        print("logvar" + logvar + "\n")
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar

# ############# Networks for stageI GAN #############
class StoryGAN(nn.Module):
    def __init__(self, cfg, video_len):
        super(StoryGAN, self).__init__()
        self.cfg = cfg
        self.gf_dim = cfg.GAN.GF_DIM * 8
        self.motion_dim = cfg.TEXT.DIMENSION + cfg.LABEL_NUM
        self.content_dim = cfg.GAN.CONDITION_DIM  # encoded text dim
        self.noise_dim = cfg.GAN.Z_DIM  # noise
        self.recurrent = nn.GRUCell(self.noise_dim + self.motion_dim, self.motion_dim)
        self.mocornn = nn.GRUCell(self.motion_dim, self.content_dim)
        self.video_len = video_len
        self.n_channels = 3
        self.filter_num = 3
        self.filter_size = 21
        self.image_size = 124
        self.out_num = 1
        self.define_module()
    

    def define_module(self):
        ninput = self.motion_dim + self.content_dim + self.image_size
        ngf = self.gf_dim
        print("ngf " + ngf)

        self.ca_net = CA_NET()
        # -> ngf x 4 x 4
        self.fc = nn.Sequential(
            nn.Linear(ninput, ngf * 4 * 4, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4),
            nn.ReLU(True))

        self.filter_net = nn.Sequential(
            nn.Linear(self.content_dim, self.filter_size * self.filter_num * self.out_num),
            nn.BatchNorm1d(self.filter_size * self.filter_num * self.out_num))

        self.image_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.image_size * self.filter_num),
            nn.BatchNorm1d(self.image_size * self.filter_num),
            nn.Tanh())

        # ngf x 4 x 4 -> ngf/2 x 8 x 8
        self.upsample1 = upBlock(ngf, ngf // 2)
        # -> ngf/4 x 16 x 16
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # -> ngf/8 x 32 x 32
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        # -> ngf/16 x 64 x 64
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        # -> 3 x 64 x 64
        self.img = nn.Sequential(
            conv3x3(ngf // 16, 3),
            nn.Tanh())

        self.m_net = nn.Sequential(
            nn.Linear(self.motion_dim, self.motion_dim),
            nn.BatchNorm1d(self.motion_dim))

        self.c_net = nn.Sequential(
            nn.Linear(self.content_dim, self.content_dim),
            nn.BatchNorm1d(self.content_dim))

        self.dfn_layer = DynamicFilterLayer(self.filter_size,
                                            pad=self.filter_size // 2)

    def get_iteration_input(self, motion_input):
        num_samples = motion_input.shape[0]
        noise = T.FloatTensor(num_samples, self.noise_dim).normal_(0, 1)
        return torch.cat((noise, motion_input), dim=1)

    def get_gru_initial_state(self, num_samples):
        return Variable(T.FloatTensor(num_samples, self.motion_dim).normal_(0, 1))

    def sample_z_motion(self, motion_input):
        video_len = 1 if len(motion_input.shape) == 2 else self.video_len
        num_samples = motion_input.shape[0]
        h_t = [self.m_net(self.get_gru_initial_state(num_samples))]
        for frame_num in range(video_len):
            if len(motion_input.shape) == 2:
                e_t = self.get_iteration_input(motion_input)
            else:
                e_t = self.get_iteration_input(motion_input[:, frame_num, :])
            h_t.append(self.recurrent(e_t, h_t[-1]))
        z_m_t = [h_k.view(-1, 1, self.motion_dim) for h_k in h_t]
        z_motion = torch.cat(z_m_t[1:], dim=1).view(-1, self.motion_dim)
        return z_motion

    def motion_content_rnn(self, motion_input, content_input):
        video_len = 1 if len(motion_input.shape) == 2 else self.video_len
        if len(motion_input.shape) == 2:
            motion_input = motion_input.unsqueeze(1)
            filler_input = torch.rand(
                (motion_input.shape[0], self.cfg.MART.max_t_len - video_len, motion_input.shape[-1]))
            if self.cfg.CUDA:
                filler_input = filler_input.cuda()
            motion_input = torch.cat((motion_input, filler_input), dim=1)
            mask = torch.cat((torch.ones((motion_input.shape[0], video_len)),
                              torch.zeros((motion_input.shape[0], self.cfg.MART.max_t_len - video_len))), dim=-1)
        else:
            mask = torch.ones((motion_input.shape[0], video_len))

        if self.cfg.CUDA:
            mask = mask.cuda()

        if self.cfg.USE_TRANSFORMER:
            mocornn_co = self.mocornn(self.moco_fc(motion_input), mask).view(-1, self.content_dim)
        else:
            h_t = [self.c_net(content_input)]
            for frame_num in range(video_len):
                h_t.append(self.mocornn(motion_input[:, frame_num, :], h_t[-1]))
            c_m_t = [h_k.view(-1, 1, self.content_dim) for h_k in h_t]
            mocornn_co = torch.cat(c_m_t[1:], dim=1).view(-1, self.content_dim)

        return mocornn_co

    def sample_videos(self, motion_input, content_input):
        content_input = content_input.view(-1, cfg.VIDEO_LEN * content_input.shape[2])
        r_code, r_mu, r_logvar = self.ca_net(torch.squeeze(content_input))
        c_mu = r_mu.repeat(self.video_len, 1).view(-1, r_mu.shape[1])

        crnn_code = self.motion_content_rnn(motion_input, r_code)

        temp = motion_input.view(-1, motion_input.shape[2])
        m_code, m_mu, m_logvar = temp, temp, temp  # self.ca_net(temp)
        m_code = m_code.view(motion_input.shape[0], self.video_len, self.motion_dim)
        zm_code = self.sample_z_motion(m_code)
        # one
        zmc_code = torch.cat((zm_code, c_mu), dim=1)
        # two
        m_image = self.image_net(m_code.view(-1, m_code.shape[2]))
        m_image = m_image.view(-1, self.filter_num, self.image_size)
        c_filter = self.filter_net(crnn_code)
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)
        mc_image = self.dfn_layer([m_image, c_filter])
        zmc_all = torch.cat((zmc_code, mc_image.squeeze(1)), dim=1)
        # combine
        zmc_all = self.fc(zmc_all)
        zmc_all = zmc_all.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(zmc_all)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        # state size 3 x 64 x 64
        h = self.img(h_code)
        fake_video = h.view(int(h.size(0) / self.video_len), self.video_len, self.n_channels, h.size(3), h.size(3))
        fake_video = fake_video.permute(0, 2, 1, 3, 4)
        return None, fake_video, m_mu, m_logvar, r_mu, r_logvar

    def sample_images(self, motion_input, content_input):
        m_code, m_mu, m_logvar = motion_input, motion_input, motion_input  # self.ca_net(motion_input)
        # print(content_input.shape, cfg.VIDEO_LEN)
        # content_input = content_input.view(-1, cfg.VIDEO_LEN * content_input.shape[2])
        content_input = torch.reshape(content_input, (-1, cfg.VIDEO_LEN * content_input.shape[2]))
        c_code, c_mu, c_logvar = self.ca_net(content_input)
        crnn_code = self.motion_content_rnn(motion_input, c_mu)
        zm_code = self.sample_z_motion(m_code)
        # one
        zmc_code = torch.cat((zm_code, c_mu), dim=1)
        # two
        m_image = self.image_net(m_code)
        m_image = m_image.view(-1, self.filter_num, self.image_size)
        c_filter = self.filter_net(crnn_code)
        c_filter = c_filter.view(-1, self.out_num, self.filter_num, self.filter_size)
        mc_image = self.dfn_layer([m_image, c_filter])
        zmc_all = torch.cat((zmc_code, mc_image.squeeze(1)), dim=1)
        # combine
        zmc_all = self.fc(zmc_all)
        zmc_all = zmc_all.view(-1, self.gf_dim, 4, 4)
        h_code = self.upsample1(zmc_all)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)
        # state size 3 x 64 x 64
        fake_img = self.img(h_code)
        return None, fake_img, m_mu, m_logvar, c_mu, c_logvar