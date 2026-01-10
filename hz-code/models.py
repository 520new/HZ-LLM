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
    def __init__(self, opt):
        super(Encoder_Visual, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(opt.resSize, opt.ngh * 2)
        self.fc2 = nn.Linear(opt.ngh * 2, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.ngh // 2)
        self.fc_disc = nn.Linear(opt.ngh // 2, opt.num_clusters)
        self.fc_mu = nn.Linear(opt.ngh // 2, opt.z_dim)
        self.fc_logvar = nn.Linear(opt.ngh // 2, opt.z_dim)
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
    def __init__(self, opt):
        super(Decoder_Visual, self).__init__()
        self.opt = opt
        self.fc1 = nn.Linear(opt.num_clusters + opt.z_dim, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.resSize)

        self.bn1 = nn.BatchNorm1d(opt.ngh)
        self.bn2 = nn.BatchNorm1d(opt.ngh)

    def forward(self, disc_embed, z_cont):
        x = torch.cat([disc_embed, z_cont], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        out = self.fc3(x)
        return out
class CrossModalAlignmentLoss(nn.Module):
    def __init__(self, lambda_cm=1.0):
        super(CrossModalAlignmentLoss, self).__init__()
        self.lambda_cm = lambda_cm
        self.mse_loss = nn.MSELoss()

    def forward(self, visual_features, attributes,
                visual_decoder, attribute_decoder,
                disc_embed_visual, z_visual, disc_embed_attr, z_attr):
        visual_from_attr = visual_decoder(disc_embed_attr, z_attr)
        loss_visual_cm = self.mse_loss(visual_from_attr, visual_features)
        attr_from_visual = attribute_decoder(disc_embed_visual, z_visual)
        loss_attr_cm = self.mse_loss(attr_from_visual, attributes)

        total_loss = self.lambda_cm * (loss_visual_cm + loss_attr_cm)

        return total_loss, loss_visual_cm, loss_attr_cm


class DistributionAlignmentLoss(nn.Module):
    def __init__(self, lambda_d=1.0):
        super(DistributionAlignmentLoss, self).__init__()
        self.lambda_d = lambda_d

    def forward(self, mu_visual, logvar_visual, mu_attribute, logvar_attribute):
        var_visual = torch.exp(logvar_visual)
        var_attribute = torch.exp(logvar_attribute)
        kl_loss = 0.5 * torch.sum(
            logvar_attribute - logvar_visual +
            (var_visual + (mu_visual - mu_attribute).pow(2)) / var_attribute - 1
        )
        kl_loss = kl_loss / mu_visual.size(0)

        return self.lambda_d * kl_loss


class MultiModalGMVAE(nn.Module):
    def __init__(self, opt):
        super(MultiModalGMVAE, self).__init__()
        self.opt = opt

        self.visual_encoder = Encoder_Visual(opt)

        self.attr_encoder = Encoder_GMVAE(opt)


        self.visual_decoder = Decoder_Visual(opt)

        self.attr_decoder = Decoder_GMVAE(opt)

        self.cross_modal_loss = CrossModalAlignmentLoss(lambda_cm=opt.lambda_cm)
        self.distribution_loss = DistributionAlignmentLoss(lambda_d=opt.lambda_d)

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
        disc_logits, mu_cont, logvar_cont = self.visual_encoder(visual_features)
        disc_embed = self.gumbel_softmax(disc_logits)
        z_cont = self.reparameterize_cont(mu_cont, logvar_cont)
        return disc_logits, mu_cont, logvar_cont, disc_embed, z_cont

    def encode_attr(self, attributes, noise):
        combined_input = torch.cat([attributes, noise], dim=1)
        disc_logits, mu_cont, logvar_cont = self.attr_encoder(combined_input)
        disc_embed = self.gumbel_softmax(disc_logits)
        z_cont = self.reparameterize_cont(mu_cont, logvar_cont)
        return disc_logits, mu_cont, logvar_cont, disc_embed, z_cont

    def forward(self, visual_features, attributes, noise=None):
        if noise is None:
            batch_size = visual_features.size(0)
            noise = torch.randn(batch_size, self.opt.nz).to(visual_features.device)

        (disc_logits_visual, mu_visual, logvar_visual,
         disc_embed_visual, z_visual) = self.encode_visual(visual_features)

        (disc_logits_attr, mu_attr, logvar_attr,
         disc_embed_attr, z_attr) = self.encode_attr(attributes, noise)

        recon_visual = self.visual_decoder(disc_embed_visual, z_visual)
        recon_attr = self.attr_decoder(disc_embed_attr, z_attr)

        losses = self.compute_losses(
            visual_features, attributes,
            recon_visual, recon_attr,
            disc_logits_visual, disc_logits_attr,
            mu_visual, logvar_visual,
            mu_attr, logvar_attr,
            z_visual, z_attr,
            disc_embed_visual, disc_embed_attr
        )

        return losses

    def compute_losses(self, visual_features, attributes,
                       recon_visual, recon_attr,
                       disc_logits_visual, disc_logits_attr,
                       mu_visual, logvar_visual, mu_attr, logvar_attr,
                       z_visual, z_attr, disc_embed_visual, disc_embed_attr):
        recon_loss_visual = F.mse_loss(recon_visual, visual_features) * 100
        recon_loss_attr = F.mse_loss(recon_attr, attributes) * 100
        recon_loss = recon_loss_visual + recon_loss_attr

        disc_q_visual = F.log_softmax(disc_logits_visual, dim=1)
        disc_prior_visual = F.softmax(self.prior_logits.unsqueeze(0).repeat(disc_logits_visual.size(0), 1), dim=1)
        kl_disc_visual = F.kl_div(disc_q_visual, disc_prior_visual, reduction='batchmean')
        kl_cont_visual = -0.5 * torch.sum(
            1 + logvar_visual - mu_visual.pow(2) - logvar_visual.exp()) / disc_logits_visual.size(0)

        disc_q_attr = F.log_softmax(disc_logits_attr, dim=1)
        disc_prior_attr = F.softmax(self.prior_logits.unsqueeze(0).repeat(disc_logits_attr.size(0), 1), dim=1)
        kl_disc_attr = F.kl_div(disc_q_attr, disc_prior_attr, reduction='batchmean')
        kl_cont_attr = -0.5 * torch.sum(1 + logvar_attr - mu_attr.pow(2) - logvar_attr.exp()) / disc_logits_attr.size(0)

        kl_loss = 0.1 * (kl_disc_visual + kl_disc_attr) + 0.1 * (kl_cont_visual + kl_cont_attr)

        cross_modal_loss, loss_visual_cm, loss_attr_cm = self.cross_modal_loss(
            visual_features, attributes,
            self.visual_decoder, self.attr_decoder,
            disc_embed_visual, z_visual, disc_embed_attr, z_attr
        )
        distribution_loss = self.distribution_loss(
            mu_visual, logvar_visual, mu_attr, logvar_attr
        )

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
        x_ns = 5 * x / x.norm(dim=1, keepdim=True)
        protos_ns = 5 * protos / protos.norm(dim=1, keepdim=True)
        logits = x_ns @ protos_ns.t()
        return logits

class ClassStandardization(nn.Module):
    def __init__(self, feat_dim: int):
        super(ClassStandardization, self).__init__()
        self.running_mean = nn.Parameter(torch.zeros(feat_dim), requires_grad = False)
        self.running_var = nn.Parameter(torch.ones(feat_dim), requires_grad = False)

    def forward(self, class_feats):
        if self.training:
            batch_mean = class_feats.mean(dim = 0)
            batch_var = class_feats.var(dim = 0)
            result = (class_feats - batch_mean.unsqueeze(0)) / (batch_var.unsqueeze(0) + 1e-8)
            self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * batch_mean.detach()
            self.running_var.data = 0.9 * self.running_var.data + 0.1 * batch_var.detach()
        else:
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
        input_dim = opt.attSize + opt.nz
        self.fc1 = nn.Linear(input_dim, opt.ngh * 2)
        self.fc2 = nn.Linear(opt.ngh * 2, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.ngh // 2)
        self.fc_disc = nn.Linear(opt.ngh // 2, opt.num_clusters)
        self.fc_mu = nn.Linear(opt.ngh // 2, opt.z_dim)
        self.fc_logvar = nn.Linear(opt.ngh // 2, opt.z_dim)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(opt.ngh * 2)
        self.bn2 = nn.BatchNorm1d(opt.ngh)
        self.bn3 = nn.BatchNorm1d(opt.ngh // 2)

    def forward(self, combined_input):
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
        self.fc1 = nn.Linear(opt.num_clusters + opt.z_dim, opt.ngh)
        self.fc2 = nn.Linear(opt.ngh, opt.ngh)
        self.fc3 = nn.Linear(opt.ngh, opt.attSize)

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
        self.prior_logits = nn.Parameter(torch.ones(opt.num_clusters))
        self.bn1 = nn.BatchNorm1d(opt.ngh)
        self.bn2 = nn.BatchNorm1d(opt.ngh)

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

    def forward(self, att, noise):
        disc_logits, mu_cont, logvar_cont = self.encoder(att, noise)

        disc_embed = self.gumbel_softmax(disc_logits)

        z_cont = self.reparameterize_cont(mu_cont, logvar_cont)

        recon_visual = self.decoder(disc_embed, z_cont)

        return recon_visual, disc_logits, mu_cont, logvar_cont, disc_embed

class Encoder_noise(nn.Module):
    def __init__(self, opt):
        super(Encoder_noise, self).__init__()
        self.__dict__.update(locals())
        self.linear = nn.Linear(opt.resSize + opt.attSize, opt.ngh)
        self.mu = nn.Linear(opt.ngh, opt.nz)
        self.var = nn.Linear(opt.ngh, opt.nz)
        self.lrelu = nn.LeakyReLU(0.2, True)
        self.apply(weights_init)

    def forward(self, visual_feats, attrs):
        concat_feats = torch.cat((visual_feats,attrs), dim=1)
        hidden = torch.tanh(self.linear(concat_feats))
        mu, var = torch.tanh(self.mu(hidden)), torch.tanh(self.var(hidden))
        return mu, var

class Decoder_noise(nn.Module):
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
    def __init__(self, args):
        super(HGCF_ZSL, self).__init__()
        self.device = args.device
        self.manifold = Lorentz(max_norm=args.max_norm)
        self.num_classes = args.nclass_all
        self.attSize = args.attSize
        self.resSize = args.resSize
        self.emb_classes = nn.Embedding(self.num_classes, self.attSize)
        self.emb_classes.weight.data.uniform_(-0.1, 0.1)
        self.emb_classes.weight = nn.Parameter(
            self.manifold.expmap0(self.emb_classes.weight, project=True))
        self.emb_classes.weight = ManifoldParameter(
            self.emb_classes.weight, self.manifold, True)
        self.visual_proj = nn.Linear(self.resSize, self.attSize)

        self.visual_proj_inv = nn.Linear(self.attSize, self.resSize)

        self.att_proj = nn.Linear(self.attSize, self.attSize)

        self.num_layers = args.num_layers
        self.margin = args.margin

        self.t = args.t if hasattr(args, 't') else 0.5

        self.use_gmvae = True
        if self.use_gmvae:

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

            self.gmvae_encoder = Encoder_GMVAE(gmvae_opt)
            self.gmvae_decoder = Decoder_GMVAE(gmvae_opt)

            self.prior_logits = nn.Parameter(torch.ones(gmvae_opt.num_clusters))

            print(f"GMVAE integrated into HGCF: {gmvae_opt.num_clusters} clusters, z_dim: {gmvae_opt.z_dim}")

            gmvae_params = list(self.gmvae_encoder.parameters()) + list(self.gmvae_decoder.parameters()) + [
                self.prior_logits]

            self.gmvae_optimizer = optim.Adam(
                gmvae_params,
                lr=getattr(args, 'gmvae_lr', 0.0001),
                betas=(0.5, 0.999)
            )

            print(f"GMVAE optimizer created with {len(gmvae_params)} parameter groups")

        print(f"HGCF_ZSL initialized for ZSL: {self.num_classes} classes, "
              f"attSize: {self.attSize}, resSize: {self.resSize}")

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

    def gmvae_forward(self, att, noise, visual_features=None):
        if not self.use_gmvae:
            return None, None, None, None, None

        try:
            if visual_features is not None:
                combined_input = torch.cat([att, noise], dim=1)
                recon_visual, disc_logits, mu_cont, logvar_cont, disc_embed = self.gmvae_encoder(combined_input)
            else:
                combined_input = torch.cat([att, noise], dim=1)
                disc_logits, mu_cont, logvar_cont = self.gmvae_encoder(combined_input)
                disc_embed = self.gumbel_softmax(disc_logits)
                z_cont = self.reparameterize_cont(mu_cont, logvar_cont)
                recon_visual = self.gmvae_decoder(disc_embed, z_cont)

            return recon_visual, disc_logits, mu_cont, logvar_cont, disc_embed
        except Exception as e:
            return None, None, None, None, None

    def compute_gmvae_loss(self, recon_visual, real_visual, disc_logits, mu_cont, logvar_cont):
        recon_loss = F.mse_loss(recon_visual, real_visual) * 100
        disc_q = F.log_softmax(disc_logits, dim=1)
        if self.prior_logits.device != disc_logits.device:
            self.prior_logits.data = self.prior_logits.data.to(disc_logits.device)

        disc_prior = F.softmax(self.prior_logits.unsqueeze(0).repeat(disc_logits.size(0), 1), dim=1)
        kl_disc = F.kl_div(disc_q, disc_prior, reduction='batchmean')

        kl_cont = -0.5 * torch.sum(1 + logvar_cont - mu_cont.pow(2) - logvar_cont.exp()) / disc_logits.size(0)

        total_loss = recon_loss + 0.1 * kl_disc + 0.1 * kl_cont

        return total_loss, recon_loss, kl_disc, kl_cont

    def train_gmvae_step(self, att, visual_features):
        if not self.use_gmvae:
            return 0, 0, 0, 0

        if not hasattr(self, 'gmvae_optimizer'):
            print("Warning: gmvae_optimizer not found, skipping GMVAE training step")
            return 0, 0, 0, 0

        self.gmvae_optimizer.zero_grad()

        batch_size = att.size(0)
        noise = torch.randn(batch_size, self.attSize).to(self.device)

        recon_visual, disc_logits, mu_cont, logvar_cont, _ = self.gmvae_forward(att, noise)

        total_loss, recon_loss, kl_disc, kl_cont = self.compute_gmvae_loss(
            recon_visual, visual_features, disc_logits, mu_cont, logvar_cont
        )

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
        gmvae_losses = (0, 0, 0, 0)
        if self.use_gmvae and train_gmvae and self.training:
            gmvae_losses = self.train_gmvae_step(attributes, visual_features)

        visual_emb = self.visual_proj(visual_features)
        visual_emb = self.manifold.expmap0(visual_emb, project=True)

        att_emb = self.att_proj(attributes)
        att_emb = self.manifold.expmap0(att_emb, project=True)

        class_emb = self.emb_classes.weight

        distances = self.manifold.sqdist_multi(visual_emb, class_emb)

        scaled_distances = -distances / 10.0

        if self.use_gmvae and train_gmvae:
            return scaled_distances, gmvae_losses
        else:
            return scaled_distances

    def generate_features(self, class_ids, num_samples=100, use_gmvae=True):

        with torch.no_grad():

            if class_ids.dtype != torch.long:
                class_ids = class_ids.long()

            batch_size = class_ids.size(0)
            total_samples = batch_size * num_samples

            if self.use_gmvae and use_gmvae:
                try:
                    class_attrs = self.emb_classes(class_ids)  # [batch_size, attSize]
                    class_attrs_expanded = class_attrs.repeat_interleave(num_samples, dim=0)

                    noise = torch.randn(total_samples, self.attSize).to(self.device)

                    recon_visual, _, _, _, _ = self.gmvae_forward(
                        class_attrs_expanded,
                        noise,
                        visual_features=None
                    )

                    if recon_visual is not None:
                        if recon_visual.size(1) != self.resSize:
                            if not hasattr(self, 'feature_adapter'):
                                self.feature_adapter = nn.Linear(recon_visual.size(1), self.resSize)
                                if self.device.type == 'cuda':
                                    self.feature_adapter = self.feature_adapter.cuda()
                            generated_features = self.feature_adapter(recon_visual)
                        else:
                            generated_features = recon_visual

                        return generated_features
                    else:
                        use_gmvae = False

                except Exception as e:
                    use_gmvae = False
            class_emb = self.emb_classes(class_ids)

            noise = torch.randn(total_samples, self.attSize).to(self.device)

            class_emb_expanded = class_emb.repeat_interleave(num_samples, dim=0)

            generated_emb = self.manifold.expmap0(class_emb_expanded + 0.1 * noise, project=True)

            if hasattr(self, 'visual_proj_inv'):
                generated_features = self.visual_proj_inv(generated_emb)
            else:
                self.visual_proj_inv = nn.Linear(self.attSize, self.resSize)
                if self.device.type == 'cuda':
                    self.visual_proj_inv = self.visual_proj_inv.cuda()
                generated_features = self.visual_proj_inv(generated_emb)

            return generated_features

    def margin_loss(self, visual_emb, class_emb, labels, margin=0.1):
        batch_size = visual_emb.size(0)
        distances = self.manifold.sqdist_multi(visual_emb, class_emb)
        positive_dist = distances[torch.arange(batch_size), labels]
        mask = torch.ones_like(distances, dtype=torch.bool)
        mask[torch.arange(batch_size), labels] = False
        negative_dists = distances[mask].view(batch_size, -1)
        negative_min_dist, _ = torch.min(negative_dists, dim=1)
        loss = torch.clamp(positive_dist - negative_min_dist + margin, min=0)
        loss = torch.mean(loss)

        return loss

    def contrastive_loss(self, visual_emb, class_emb, labels, temperature=0.5):
        similarities = -self.manifold.sqdist_multi(visual_emb, class_emb) / temperature
        loss = F.cross_entropy(similarities, labels)
        return loss