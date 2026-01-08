import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from phase1_model import HGCF as HGCF_Phase1
from manifolds.lorentz import Lorentz
from geoopt import ManifoldParameter
import os.path
import re
from util import recall_func, ndcg_func
import torch.optim as optim
import argparse


class Encoder_Visual(nn.Module):
    """视觉特征编码器"""

    def __init__(self, opt):
        super(Encoder_Visual, self).__init__()
        self.opt = opt

        # 修复：输入维度为 resSize (2048)
        self.fc1 = nn.Linear(opt.resSize, opt.ngh * 2)
        self.fc2 = nn.Linear(opt.ngh * 2, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.ngh // 2)

        # 输出头
        self.fc_disc = nn.Linear(opt.ngh // 2, opt.num_clusters)
        self.fc_mu = nn.Linear(opt.ngh // 2, opt.z_dim)
        self.fc_logvar = nn.Linear(opt.ngh // 2, opt.z_dim)

        # 添加dropout和批归一化
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(opt.ngh * 2)
        self.bn2 = nn.BatchNorm1d(opt.ngh)
        self.bn3 = nn.BatchNorm1d(opt.ngh // 2)

    def forward(self, visual_features):
        x = visual_features

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))

        disc_logits = self.fc_disc(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return disc_logits, mu, logvar


class Decoder_Visual(nn.Module):
    """视觉特征解码器"""

    def __init__(self, opt):
        super(Decoder_Visual, self).__init__()
        self.opt = opt

        # 输入：离散变量的 one-hot 嵌入 + 连续变量
        self.fc1 = nn.Linear(opt.num_clusters + opt.z_dim, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.resSize)  # 输出视觉特征 [2048]

        self.bn1 = nn.BatchNorm1d(opt.ngh)
        self.bn2 = nn.BatchNorm1d(opt.ngh)

    def forward(self, disc_embed, z_cont):
        x = torch.cat([disc_embed, z_cont], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        out = self.fc3(x)
        return out  # 输出维度: [batch_size, resSize=2048]

class CrossModalAlignmentLoss(nn.Module):
    """交叉模态对齐损失：L_CM = ||I- Decoder1(Z2)||^2 + ||A-Decoder2(Z1)||^2"""

    def __init__(self, lambda_cm=1.0):
        super(CrossModalAlignmentLoss, self).__init__()
        self.lambda_cm = lambda_cm
        self.mse_loss = nn.MSELoss()

    def forward(self, visual_features, attributes,
                visual_decoder, attribute_decoder,
                disc_embed_visual, z_visual, disc_embed_attr, z_attr):
        """
        Args:
            visual_features: 原始视觉特征 I
            attributes: 原始属性特征 A
            visual_decoder: 视觉解码器 Decoder1
            attribute_decoder: 属性解码器 Decoder2
            disc_embed_visual: 视觉模态的离散嵌入
            z_visual: 视觉模态的连续隐变量 Z1
            disc_embed_attr: 属性模态的离散嵌入
            z_attr: 属性模态的连续隐变量 Z2
        """

        # 用属性隐变量重构视觉特征: I' = Decoder1(disc_embed_attr, z_attr)
        visual_from_attr = visual_decoder(disc_embed_attr, z_attr)
        loss_visual_cm = self.mse_loss(visual_from_attr, visual_features)

        # 用视觉隐变量重构属性特征: A' = Decoder2(disc_embed_visual, z_visual)
        attr_from_visual = attribute_decoder(disc_embed_visual, z_visual)
        loss_attr_cm = self.mse_loss(attr_from_visual, attributes)

        total_loss = self.lambda_cm * (loss_visual_cm + loss_attr_cm)

        return total_loss, loss_visual_cm, loss_attr_cm


class DistributionAlignmentLoss(nn.Module):
    """分布对齐损失：L_D = KL((N(μ_I, Σ_I)|| N(μ_A, Σ_A))"""

    def __init__(self, lambda_d=1.0):
        super(DistributionAlignmentLoss, self).__init__()
        self.lambda_d = lambda_d

    def forward(self, mu_visual, logvar_visual, mu_attribute, logvar_attribute):
        """
        Args:
            mu_visual, logvar_visual: 视觉模态的均值和方差
            mu_attribute, logvar_attribute: 属性模态的均值和方差
        """

        # 计算两个高斯分布之间的KL散度
        # KL(q_visual || q_attribute)
        var_visual = torch.exp(logvar_visual)
        var_attribute = torch.exp(logvar_attribute)

        # KL divergence formula for two Gaussians
        kl_loss = 0.5 * torch.sum(
            logvar_attribute - logvar_visual +
            (var_visual + (mu_visual - mu_attribute).pow(2)) / var_attribute - 1
        )

        # 平均KL损失
        kl_loss = kl_loss / mu_visual.size(0)

        return self.lambda_d * kl_loss


class MultiModalGMVAE(nn.Module):
    """多模态GMVAE，集成交叉模态对齐和分布对齐"""

    def __init__(self, opt):
        super(MultiModalGMVAE, self).__init__()
        self.opt = opt

        # 视觉编码器
        self.visual_encoder = Encoder_Visual(opt)
        # 属性编码器
        self.attr_encoder = Encoder_GMVAE(opt)

        # 视觉解码器
        self.visual_decoder = Decoder_Visual(opt)
        # 属性解码器 - 修复：使用修改后的Decoder_GMVAE
        self.attr_decoder = Decoder_GMVAE(opt)

        # 损失函数
        self.cross_modal_loss = CrossModalAlignmentLoss(lambda_cm=opt.lambda_cm)
        self.distribution_loss = DistributionAlignmentLoss(lambda_d=opt.lambda_d)

        # 先验参数
        self.prior_logits = nn.Parameter(torch.ones(opt.num_clusters))

    def gumbel_softmax(self, logits, temperature=0.5):
        eps = 1e-20
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + eps) + eps)
        y = logits + gumbel
        return F.softmax(y / temperature, dim=1)

    def reparameterize_cont(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_visual(self, visual_features):
        """编码视觉特征"""
        disc_logits, mu_cont, logvar_cont = self.visual_encoder(visual_features)
        disc_embed = self.gumbel_softmax(disc_logits)
        z_cont = self.reparameterize_cont(mu_cont, logvar_cont)
        return disc_logits, mu_cont, logvar_cont, disc_embed, z_cont

    def encode_attr(self, attributes, noise):
        """编码属性特征 - 修复：添加noise参数"""
        # 将属性和噪声连接作为输入
        combined_input = torch.cat([attributes, noise], dim=1)
        disc_logits, mu_cont, logvar_cont = self.attr_encoder(combined_input)
        disc_embed = self.gumbel_softmax(disc_logits)
        z_cont = self.reparameterize_cont(mu_cont, logvar_cont)
        return disc_logits, mu_cont, logvar_cont, disc_embed, z_cont

    def forward(self, visual_features, attributes, noise=None):
        """
        修复：添加noise参数
        Args:
            visual_features: 视觉特征 [batch_size, resSize=2048]
            attributes: 属性特征 [batch_size, attSize=85]
            noise: 噪声输入 [batch_size, nz=85]（可选）
        """

        # 如果没有提供噪声，生成随机噪声
        if noise is None:
            batch_size = visual_features.size(0)
            noise = torch.randn(batch_size, self.opt.nz).to(visual_features.device)

        # 编码视觉模态
        (disc_logits_visual, mu_visual, logvar_visual,
         disc_embed_visual, z_visual) = self.encode_visual(visual_features)

        # 编码属性模态 - 修复：传入noise参数
        (disc_logits_attr, mu_attr, logvar_attr,
         disc_embed_attr, z_attr) = self.encode_attr(attributes, noise)

        # 模态内重构
        recon_visual = self.visual_decoder(disc_embed_visual, z_visual)  # 输出 [batch_size, resSize=2048]
        recon_attr = self.attr_decoder(disc_embed_attr, z_attr)  # 输出 [batch_size, attSize=85]

        # 计算各种损失 - 修复：传递离散嵌入参数
        losses = self.compute_losses(
            visual_features, attributes,
            recon_visual, recon_attr,
            disc_logits_visual, disc_logits_attr,
            mu_visual, logvar_visual,
            mu_attr, logvar_attr,
            z_visual, z_attr,
            disc_embed_visual, disc_embed_attr  # 添加离散嵌入
        )

        return losses

    def compute_losses(self, visual_features, attributes,
                       recon_visual, recon_attr,
                       disc_logits_visual, disc_logits_attr,
                       mu_visual, logvar_visual, mu_attr, logvar_attr,
                       z_visual, z_attr, disc_embed_visual, disc_embed_attr):  # 添加离散嵌入参数
        # 1. 重构损失 L_VAE
        recon_loss_visual = F.mse_loss(recon_visual, visual_features) * 100
        recon_loss_attr = F.mse_loss(recon_attr, attributes) * 100
        recon_loss = recon_loss_visual + recon_loss_attr

        # 2. 模态内KL损失
        # 视觉模态KL
        disc_q_visual = F.log_softmax(disc_logits_visual, dim=1)
        disc_prior_visual = F.softmax(self.prior_logits.unsqueeze(0).repeat(disc_logits_visual.size(0), 1), dim=1)
        kl_disc_visual = F.kl_div(disc_q_visual, disc_prior_visual, reduction='batchmean')
        kl_cont_visual = -0.5 * torch.sum(
            1 + logvar_visual - mu_visual.pow(2) - logvar_visual.exp()) / disc_logits_visual.size(0)

        # 属性模态KL
        disc_q_attr = F.log_softmax(disc_logits_attr, dim=1)
        disc_prior_attr = F.softmax(self.prior_logits.unsqueeze(0).repeat(disc_logits_attr.size(0), 1), dim=1)
        kl_disc_attr = F.kl_div(disc_q_attr, disc_prior_attr, reduction='batchmean')
        kl_cont_attr = -0.5 * torch.sum(1 + logvar_attr - mu_attr.pow(2) - logvar_attr.exp()) / disc_logits_attr.size(0)

        kl_loss = 0.1 * (kl_disc_visual + kl_disc_attr) + 0.1 * (kl_cont_visual + kl_cont_attr)

        # 3. 交叉模态对齐损失 L_CM - 修复：传递所有需要的参数
        cross_modal_loss, loss_visual_cm, loss_attr_cm = self.cross_modal_loss(
            visual_features, attributes,
            self.visual_decoder, self.attr_decoder,
            disc_embed_visual, z_visual, disc_embed_attr, z_attr  # 传递离散嵌入和连续变量
        )

        # 4. 分布对齐损失 L_D
        distribution_loss = self.distribution_loss(
            mu_visual, logvar_visual, mu_attr, logvar_attr
        )

        # 总损失
        total_loss = recon_loss + kl_loss + cross_modal_loss + distribution_loss

        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'cross_modal_loss': cross_modal_loss,
            'distribution_loss': distribution_loss,
            'recon_loss_visual': recon_loss_visual,
            'recon_loss_attr': recon_loss_attr,
            'loss_visual_cm': loss_visual_cm,
            'loss_attr_cm': loss_attr_cm
        }

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class CNZSLModel(nn.Module):
    def __init__(self, attr_dim: int, hid_dim: int, proto_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(attr_dim, hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, hid_dim),
            ClassStandardization(hid_dim),
            nn.ReLU(),

            ClassStandardization(hid_dim),
            nn.Linear(hid_dim, proto_dim),
            nn.ReLU(),
        )

        weight_var = 1 / (hid_dim * proto_dim)
        b = np.sqrt(3 * weight_var)
        self.model[-2].weight.data.uniform_(-b, b)

    def forward(self, x, attrs):
        protos = self.model(attrs)
        x_ns = 5 * x / x.norm(dim=1, keepdim=True)  # [batch_size, x_dim]
        protos_ns = 5 * protos / protos.norm(dim=1, keepdim=True)  # [num_classes, x_dim]
        logits = x_ns @ protos_ns.t()  # [batch_size, num_classes]
        return logits

class ClassStandardization(nn.Module):
    def __init__(self, feat_dim: int):
        super(ClassStandardization, self).__init__()
        self.running_mean = nn.Parameter(torch.zeros(feat_dim), requires_grad = False)
        self.running_var = nn.Parameter(torch.ones(feat_dim), requires_grad = False)

    def forward(self, class_feats):
        """
        Input: class_feats of shape [num_classes, feat_dim]
        Output: class_feats (standardized) of shape [num_classes, feat_dim]
        """
        if self.training:
            batch_mean = class_feats.mean(dim = 0)
            batch_var = class_feats.var(dim = 0)

            # Normalizing the batch
            result = (class_feats - batch_mean.unsqueeze(0)) / (batch_var.unsqueeze(0) + 1e-8)

            # Updating the running mean/std
            self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * batch_mean.detach()
            self.running_var.data = 0.9 * self.running_var.data + 0.1 * batch_var.detach()
        else:
            # Using accumulated statistics
            # Attention! For the test inference, we cant use batch-wise statistics,
            # only the accumulated ones. Otherwise, it will be quite transductive
            result = (class_feats - self.running_mean.unsqueeze(0)) / (self.running_var.unsqueeze(0) + 1e-8)
        return result

class ConvertNet(nn.Module):
    def __init__(self, opt):
        super(ConvertNet, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)

    def forward(self, ori_visual_embeddings, netG):
        reg_attributes = self.relu(self.fc1(ori_visual_embeddings))

        zero_noise = torch.zeros((reg_attributes.shape[0], reg_attributes.shape[1])).cuda()
        with torch.no_grad():
            reg_visual_embeddings = netG(Variable(reg_attributes), Variable(zero_noise))
        return reg_attributes,reg_visual_embeddings

class Encoder(nn.Module):
    '''Encoder of AutoEncoder network'''
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(opt.attSize + opt.nz, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, att, noise):
        h = torch.cat((att, noise), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class Decoder(nn.Module):
    '''Decoder of AutoEncoder network'''
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.attSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, x):
        h = self.lrelu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return h

class AutoEncoder(nn.Module):
    def __init__(self, opt):
        super(AutoEncoder, self).__init__()
        self.__dict__.update(locals())
        self.encoder = Encoder(opt)
        self.decoder = Decoder(opt)

    def forward(self, att, noise):
        res_att = self.decoder(self.encoder(att, noise))
        return res_att


class Encoder_GMVAE(nn.Module):
    def __init__(self, opt):
        super(Encoder_GMVAE, self).__init__()
        self.opt = opt

        # 修复：输入维度为 attSize + nz
        input_dim = opt.attSize + opt.nz

        # 增强编码器结构
        self.fc1 = nn.Linear(input_dim, opt.ngh * 2)
        self.fc2 = nn.Linear(opt.ngh * 2, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.ngh // 2)

        # 输出头
        self.fc_disc = nn.Linear(opt.ngh // 2, opt.num_clusters)
        self.fc_mu = nn.Linear(opt.ngh // 2, opt.z_dim)
        self.fc_logvar = nn.Linear(opt.ngh // 2, opt.z_dim)

        # 添加dropout和批归一化
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(opt.ngh * 2)
        self.bn2 = nn.BatchNorm1d(opt.ngh)
        self.bn3 = nn.BatchNorm1d(opt.ngh // 2)

    def forward(self, combined_input):
        """
        修复：接收连接后的输入（属性 + 噪声）
        Args:
            combined_input: 连接后的输入 [attributes, noise]
        """
        x = combined_input

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))

        disc_logits = self.fc_disc(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return disc_logits, mu, logvar

class Decoder_GMVAE(nn.Module):
    def __init__(self, opt):
        super(Decoder_GMVAE, self).__init__()
        self.opt = opt
        # 输入：离散变量的 one-hot 嵌入 + 连续变量
        self.fc1 = nn.Linear(opt.num_clusters + opt.z_dim, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        # 修复：输出维度改为 attSize (85) 而不是 resSize (2048)
        self.fc3 = nn.Linear(opt.ngh, opt.attSize)  # 输出属性特征

    def forward(self, disc_embed, z_cont):
        x = torch.cat([disc_embed, z_cont], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out


class GMVAE(nn.Module):
    def __init__(self, opt):
        super(GMVAE, self).__init__()
        self.opt = opt
        self.encoder = Encoder_GMVAE(opt)
        self.decoder = Decoder_GMVAE(opt)

        # 改进先验分布 - 修复：正确初始化Parameter
        self.prior_logits = nn.Parameter(torch.ones(opt.num_clusters))
        # 删除这行：if opt.cuda: self.prior_logits = self.prior_logits.cuda()

        # 添加批量归一化
        self.bn1 = nn.BatchNorm1d(opt.ngh)
        self.bn2 = nn.BatchNorm1d(opt.ngh)

    def gumbel_softmax(self, logits, temperature=0.5):  # 提高温度
        """Gumbel-softmax 重参数化"""
        eps = 1e-20
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + eps) + eps)
        y = logits + gumbel
        return F.softmax(y / temperature, dim=1)

    def reparameterize_cont(self, mu, logvar):
        """连续变量重参数化"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, att, noise):
        # 编码器前向传播
        disc_logits, mu_cont, logvar_cont = self.encoder(att, noise)

        # 离散变量采样
        disc_embed = self.gumbel_softmax(disc_logits)

        # 连续变量采样
        z_cont = self.reparameterize_cont(mu_cont, logvar_cont)

        # 解码器重构
        recon_visual = self.decoder(disc_embed, z_cont)

        return recon_visual, disc_logits, mu_cont, logvar_cont, disc_embed

class Encoder_noise(nn.Module):
    '''This module is for generating noise from visual features'''
    def __init__(self, opt):
        super(Encoder_noise, self).__init__()
        self.__dict__.update(locals())
        self.linear = nn.Linear(opt.resSize + opt.attSize, opt.ngh)
        # self.linear = nn.Linear(opt.resSize, opt.ngh)
        self.mu = nn.Linear(opt.ngh, opt.nz)
        self.var = nn.Linear(opt.ngh, opt.nz)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, visual_feats, attrs):
        concat_feats = torch.cat((visual_feats,attrs), dim=1)
        # concat_feats = visual_feats
        hidden = torch.tanh(self.linear(concat_feats))
        mu, var = torch.tanh(self.mu(hidden)), torch.tanh(self.var(hidden))
        return mu, var

class Decoder_noise(nn.Module):
    '''This module is for decoding mu and var to reconstructed visual features'''
    def __init__(self, opt):
        super(Decoder_noise, self).__init__()
        self.__dict__.update(locals())
        self.linear = nn.Linear(opt.nz, opt.ndh)
        self.recon = nn.Linear(opt.ndh, opt.resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.relu = nn.ReLU(True)
        self.apply(weights_init)

    def forward(self, z):
        h = self.lrelu(self.linear(z))
        h = self.relu(self.recon(h))
        return h

class VAE_noise(nn.Module):
    def __init__(self, opt):
        super(VAE_noise, self).__init__()
        self.__dict__.update(locals())
        self.encoder = Encoder_noise(opt)
        self.decoder = Decoder_noise(opt)

    def reparameterize(self, mu, log_var):
        epsilon = torch.randn(mu.shape).cuda()
        z = mu + epsilon * torch.exp(log_var / 2)
        return z

    def forward(self, visual_feats, attrs):
        mu, log_var = self.encoder(visual_feats, attrs)
        z = self.reparameterize(mu, log_var)
        recon_visual_feats = self.decoder(z)
        return recon_visual_feats, mu, log_var

class TripCenterLoss_min_margin(nn.Module):
    def __init__(self, num_classes=40, feat_dim=85, use_gpu=True):
        super(TripCenterLoss_min_margin, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, margin):
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()

        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        dist = distmat[mask]

        other=torch.FloatTensor(batch_size,self.num_classes-1).cuda()
        for i in range(batch_size):
            other[i]=(distmat[i,mask[i,:]==0])
        dist_min,_=other.min(dim=1)
        loss = torch.max(margin+dist-dist_min,torch.tensor(0.0).cuda()).sum() / batch_size
        return loss

class Mapping(nn.Module):
    def __init__(self, opt):
        super(Mapping, self).__init__()
        self.latensize=opt.latenSize
        self.encoder_linear = nn.Linear(opt.resSize, opt.latenSize*2)
        self.discriminator = nn.Linear(opt.latenSize, 1)
        self.classifier = nn.Linear(opt.latenSize, opt.nclass_seen)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.sigmoid = nn.Sigmoid()
        self.logic = nn.LogSoftmax(dim=1)
        self.apply(weights_init)

    def forward(self, x, train_G=False):
        laten=self.lrelu(self.encoder_linear(x))
        mus,stds = laten[:,:self.latensize],laten[:,self.latensize:]
        stds=self.sigmoid(stds)
        encoder_out = reparameter(mus, stds)
        if not train_G:
            dis_out = self.discriminator(encoder_out)
        else:
            dis_out = self.discriminator(mus)
        pred=self.logic(self.classifier(mus))
        return mus,stds,dis_out,pred,encoder_out

def reparameter(mu,sigma):
    return (torch.randn_like(mu) *sigma) + mu

class ML(nn.Module):
    def __init__(self, opt):
        super(ML, self).__init__()
        self.fc1 = nn.Linear(opt.resSize, opt.ndh)
        self.fc2 = nn.Linear(opt.ndh, opt.out_dim)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, x):
        hidden = self.lrelu(self.fc1(x))
        pred1 = self.fc2(hidden)
        return pred1


class HGCF_ZSL(nn.Module):
    """为ZSL任务调整的HGCF模型"""

    def __init__(self, args):
        super(HGCF_ZSL, self).__init__()
        self.device = args.device
        self.manifold = Lorentz(max_norm=args.max_norm)

        # ZSL特定的参数
        self.num_classes = args.nclass_all
        self.attSize = args.attSize
        self.resSize = args.resSize

        # 初始化类别嵌入 - 使用类别属性作为基础
        self.emb_classes = nn.Embedding(self.num_classes, self.attSize)
        self.emb_classes.weight.data.uniform_(-0.1, 0.1)

        # 将类别嵌入投影到双曲空间
        self.emb_classes.weight = nn.Parameter(
            self.manifold.expmap0(self.emb_classes.weight, project=True))
        self.emb_classes.weight = ManifoldParameter(
            self.emb_classes.weight, self.manifold, True)

        # 视觉特征到双曲空间的投影
        self.visual_proj = nn.Linear(self.resSize, self.attSize)

        # 添加逆投影层：从双曲空间回到视觉特征空间
        self.visual_proj_inv = nn.Linear(self.attSize, self.resSize)

        # 属性到双曲空间的投影
        self.att_proj = nn.Linear(self.attSize, self.attSize)

        # 图卷积层参数
        self.num_layers = args.num_layers
        self.margin = args.margin

        # 对比学习参数
        self.t = args.t if hasattr(args, 't') else 0.5

        # ========== 新增：集成GMVAE组件 ==========
        self.use_gmvae = True  # 控制是否使用GMVAE
        if self.use_gmvae:
            # 创建GMVAE所需的参数对象
            class GMVAE_Opt:
                def __init__(self, args):
                    self.attSize = args.attSize
                    self.nz = args.nz
                    self.ngh = getattr(args, 'ngh', 4096)
                    self.num_clusters = getattr(args, 'num_clusters', 10)
                    self.z_dim = getattr(args, 'z_dim', 85)
                    self.resSize = args.resSize
                    self.cuda = args.cuda

            gmvae_opt = GMVAE_Opt(args)

            # 初始化GMVAE编码器和解码器
            self.gmvae_encoder = Encoder_GMVAE(gmvae_opt)
            self.gmvae_decoder = Decoder_GMVAE(gmvae_opt)

            # GMVAE先验参数
            self.prior_logits = nn.Parameter(torch.ones(gmvae_opt.num_clusters))

            print(f"GMVAE integrated into HGCF: {gmvae_opt.num_clusters} clusters, z_dim: {gmvae_opt.z_dim}")

            # ========== 关键修复：确保优化器被正确创建 ==========
            # 收集所有GMVAE参数
            gmvae_params = list(self.gmvae_encoder.parameters()) + list(self.gmvae_decoder.parameters()) + [
                self.prior_logits]

            # 创建优化器
            self.gmvae_optimizer = optim.Adam(
                gmvae_params,
                lr=getattr(args, 'gmvae_lr', 0.0001),
                betas=(0.5, 0.999)
            )

            print(f"GMVAE optimizer created with {len(gmvae_params)} parameter groups")
        # ========== GMVAE集成结束 ==========

        print(f"HGCF_ZSL initialized for ZSL: {self.num_classes} classes, "
              f"attSize: {self.attSize}, resSize: {self.resSize}")

    # ========== 新增：GMVAE相关方法 ==========
    def gumbel_softmax(self, logits, temperature=0.5):
        """Gumbel-softmax 重参数化"""
        eps = 1e-20
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + eps) + eps)
        y = logits + gumbel
        return F.softmax(y / temperature, dim=1)

    def reparameterize_cont(self, mu, logvar):
        """连续变量重参数化"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def gmvae_forward(self, att, noise, visual_features=None):
        """GMVAE前向传播 - 修复参数传递问题"""
        if not self.use_gmvae:
            return None, None, None, None, None

        try:
            # 如果提供了视觉特征，使用完整的前向传播
            if visual_features is not None:
                # 修复：将属性和噪声合并作为输入
                combined_input = torch.cat([att, noise], dim=1)
                # 这里需要根据实际的GMVAE结构调整
                # 假设GMVAE的forward方法接受combined_input
                recon_visual, disc_logits, mu_cont, logvar_cont, disc_embed = self.gmvae_encoder(combined_input)
            else:
                # 生成模式下，只使用属性和噪声
                combined_input = torch.cat([att, noise], dim=1)
                # 这里需要调用GMVAE的生成方法或解码器
                disc_logits, mu_cont, logvar_cont = self.gmvae_encoder(combined_input)
                disc_embed = self.gumbel_softmax(disc_logits)
                z_cont = self.reparameterize_cont(mu_cont, logvar_cont)
                recon_visual = self.gmvae_decoder(disc_embed, z_cont)

            return recon_visual, disc_logits, mu_cont, logvar_cont, disc_embed
        except Exception as e:
            print(f"GMVAE前向传播失败: {e}")
            return None, None, None, None, None

    def compute_gmvae_loss(self, recon_visual, real_visual, disc_logits, mu_cont, logvar_cont):
        """计算GMVAE损失"""
        # 重构损失
        recon_loss = F.mse_loss(recon_visual, real_visual) * 100

        # 离散变量KL散度
        disc_q = F.log_softmax(disc_logits, dim=1)

        # 确保prior_logits在正确的设备上
        if self.prior_logits.device != disc_logits.device:
            self.prior_logits.data = self.prior_logits.data.to(disc_logits.device)

        disc_prior = F.softmax(self.prior_logits.unsqueeze(0).repeat(disc_logits.size(0), 1), dim=1)
        kl_disc = F.kl_div(disc_q, disc_prior, reduction='batchmean')

        # 连续变量KL散度
        kl_cont = -0.5 * torch.sum(1 + logvar_cont - mu_cont.pow(2) - logvar_cont.exp()) / disc_logits.size(0)

        # 总损失
        total_loss = recon_loss + 0.1 * kl_disc + 0.1 * kl_cont

        return total_loss, recon_loss, kl_disc, kl_cont

    def train_gmvae_step(self, att, visual_features):
        """训练GMVAE一步"""
        if not self.use_gmvae:
            return 0, 0, 0, 0

        # ========== 关键修复：确保优化器存在 ==========
        if not hasattr(self, 'gmvae_optimizer'):
            print("Warning: gmvae_optimizer not found, skipping GMVAE training step")
            return 0, 0, 0, 0

        self.gmvae_optimizer.zero_grad()

        # 生成噪声
        batch_size = att.size(0)
        noise = torch.randn(batch_size, self.attSize).to(self.device)

        # GMVAE前向传播
        recon_visual, disc_logits, mu_cont, logvar_cont, _ = self.gmvae_forward(att, noise)

        # 计算损失
        total_loss, recon_loss, kl_disc, kl_cont = self.compute_gmvae_loss(
            recon_visual, visual_features, disc_logits, mu_cont, logvar_cont
        )

        # 反向传播
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.gmvae_encoder.parameters()) +
            list(self.gmvae_decoder.parameters()) +
            [self.prior_logits],
            max_norm=1.0
        )
        self.gmvae_optimizer.step()

        return total_loss.item(), recon_loss.item(), kl_disc.item(), kl_cont.item()

    def forward(self, visual_features, attributes, labels=None, train_gmvae=False):
        """
        ZSL任务的前向传播
        """
        # ========== 新增：训练GMVAE ==========
        gmvae_losses = (0, 0, 0, 0)
        if self.use_gmvae and train_gmvae and self.training:
            gmvae_losses = self.train_gmvae_step(attributes, visual_features)

        # 将视觉特征投影到双曲空间
        visual_emb = self.visual_proj(visual_features)
        visual_emb = self.manifold.expmap0(visual_emb, project=True)

        # 将属性特征投影到双曲空间
        att_emb = self.att_proj(attributes)
        att_emb = self.manifold.expmap0(att_emb, project=True)

        # 获取所有类别的双曲嵌入
        class_emb = self.emb_classes.weight  # [num_classes, attSize]

        # 计算视觉特征与所有类别的双曲距离
        distances = self.manifold.sqdist_multi(visual_emb, class_emb)  # [batch_size, num_classes]

        # 使用负距离并缩放，确保数值稳定性
        scaled_distances = -distances / 10.0  # 缩放因子，避免梯度爆炸

        # ========== 新增：返回GMVAE损失用于记录 ==========
        if self.use_gmvae and train_gmvae:
            return scaled_distances, gmvae_losses
        else:
            return scaled_distances

    def generate_features(self, class_ids, num_samples=100, use_gmvae=True):
        """为指定类别生成特征 - 修复版本"""
        with torch.no_grad():
            # 确保输入是整数类型
            if class_ids.dtype != torch.long:
                class_ids = class_ids.long()

            batch_size = class_ids.size(0)
            total_samples = batch_size * num_samples

            if self.use_gmvae and use_gmvae:
                try:
                    # ========== 使用GMVAE生成特征 ==========
                    # 获取类别属性
                    class_attrs = self.emb_classes(class_ids)  # [batch_size, attSize]
                    class_attrs_expanded = class_attrs.repeat_interleave(num_samples, dim=0)

                    # 生成噪声
                    noise = torch.randn(total_samples, self.attSize).to(self.device)

                    # 修复：使用正确的参数调用gmvae_forward
                    recon_visual, _, _, _, _ = self.gmvae_forward(
                        class_attrs_expanded,
                        noise,
                        visual_features=None  # 生成模式下不提供视觉特征
                    )

                    if recon_visual is not None:
                        # 如果生成的维度不匹配，使用投影层
                        if recon_visual.size(1) != self.resSize:
                            # print(f"GMVAE生成特征维度 {recon_visual.size(1)} 不匹配期望的 {self.resSize}，使用投影层")
                            if not hasattr(self, 'feature_adapter'):
                                self.feature_adapter = nn.Linear(recon_visual.size(1), self.resSize)
                                if self.device.type == 'cuda':
                                    self.feature_adapter = self.feature_adapter.cuda()
                            generated_features = self.feature_adapter(recon_visual)
                        else:
                            generated_features = recon_visual

                        return generated_features
                    else:
                        print("GMVAE生成返回None，回退到标准方法")
                        use_gmvae = False

                except Exception as e:
                    print(f"GMVAE生成失败: {e}，回退到标准方法")
                    use_gmvae = False

            # ========== 原有生成方法 ==========
            # 获取指定类别的嵌入
            class_emb = self.emb_classes(class_ids)  # [batch_size, emb_dim]

            # 生成噪声
            noise = torch.randn(total_samples, self.attSize).to(self.device)

            # 扩展类别嵌入
            class_emb_expanded = class_emb.repeat_interleave(num_samples, dim=0)

            # 在双曲空间中生成特征
            generated_emb = self.manifold.expmap0(class_emb_expanded + 0.1 * noise, project=True)

            # 使用逆投影将双曲特征转换回视觉特征空间
            if hasattr(self, 'visual_proj_inv'):
                generated_features = self.visual_proj_inv(generated_emb)
            else:
                # 如果没有逆投影，创建一个
                self.visual_proj_inv = nn.Linear(self.attSize, self.resSize)
                if self.device.type == 'cuda':
                    self.visual_proj_inv = self.visual_proj_inv.cuda()
                generated_features = self.visual_proj_inv(generated_emb)

            return generated_features

    def margin_loss(self, visual_emb, class_emb, labels, margin=0.1):
        """计算边界损失 - 用于训练"""
        batch_size = visual_emb.size(0)

        # 计算所有样本与所有类别的距离
        distances = self.manifold.sqdist_multi(visual_emb, class_emb)  # [batch_size, num_classes]

        # 获取正样本距离
        positive_dist = distances[torch.arange(batch_size), labels]

        # 创建掩码排除正样本
        mask = torch.ones_like(distances, dtype=torch.bool)
        mask[torch.arange(batch_size), labels] = False

        # 获取负样本最小距离
        negative_dists = distances[mask].view(batch_size, -1)
        negative_min_dist, _ = torch.min(negative_dists, dim=1)

        # 计算边界损失
        loss = torch.clamp(positive_dist - negative_min_dist + margin, min=0)
        loss = torch.mean(loss)

        return loss

    def contrastive_loss(self, visual_emb, class_emb, labels, temperature=0.5):
        """对比学习损失"""
        # 计算相似度矩阵
        similarities = -self.manifold.sqdist_multi(visual_emb, class_emb) / temperature

        # 对比损失
        loss = F.cross_entropy(similarities, labels)
        return loss