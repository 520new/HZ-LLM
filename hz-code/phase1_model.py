import os.path
import numpy as np
import torch
import torch.nn as nn
from util import recall_func, ndcg_func
from geoopt import ManifoldParameter
from manifolds.lorentz import Lorentz
from torch.distributions import Normal
import torch.nn.functional as F

class MoE(nn.Module):
    def __init__(self, input_dim, output_dim, num_experts):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.gating_network = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        weights = torch.softmax(self.gating_network(x), dim=-1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        output = torch.sum(expert_outputs * weights.unsqueeze(2), dim=1)
        return output
# 双曲VAE解码器
class HyperbolicDecoder(nn.Module):
    def __init__(self, args):
        super(HyperbolicDecoder, self).__init__()
        self.manifold = Lorentz(max_norm=args.max_norm)
        self.fc1 = nn.Linear(args.emb_dim, args.emb_dim * 2)
        self.fc2 = nn.Linear(args.emb_dim * 2, args.emb_dim)
        self.moe = MoE(input_dim=args.emb_dim, output_dim=args.emb_dim, num_experts=args.num_experts)

    def forward(self, z):
        # z是双曲空间中的点 (通过指数映射得到)
        # 先映射到切空间
        z_tangent = self.manifold.logmap0(z)
        h = F.relu(self.fc1(z_tangent))
        h = self.fc2(h)
        h = self.moe(h)
        # 映射回双曲空间
        return self.manifold.expmap0(h, project=True)

class HGCF(nn.Module):
    def __init__(self, args):
        super(HGCF, self).__init__()
        self.device = args.device
        self.manifold = Lorentz(max_norm=args.max_norm)
        self.num_users, self.num_items = args.num_users, args.num_items
        self.num_layers, self.margin = args.num_layers, args.margin
        self.vae_beta = args.vae_beta  # 新增VAE损失权重参数
        emb_user_path = os.path.join(os.path.join('data', args.dataset), f'emb_user.pt')
        self.emb_user = nn.Embedding.from_pretrained(torch.load(emb_user_path), freeze=True)
        self.emb_user.weight = nn.Parameter(self.manifold.expmap0(self.emb_user.state_dict()['weight'], project=True))
        self.emb_user.weight = ManifoldParameter(self.emb_user.weight, self.manifold, False)
        emb_item_path = os.path.join(os.path.join('data', args.dataset), f'emb_item.pt')
        self.emb_item = nn.Embedding.from_pretrained(torch.load(emb_item_path), freeze=True)
        self.emb_item.weight = nn.Parameter(self.manifold.expmap0(self.emb_item.state_dict()['weight'], project=True))
        self.emb_item.weight = ManifoldParameter(self.emb_item.weight, self.manifold, False)
        self.transform = MoE(input_dim=self.emb_user.weight.size(1), output_dim=args.emb_dim,
                             num_experts=args.num_experts).to(self.device)
        # 新增双曲VAE组件
        self.vae_mu = nn.Linear(args.emb_dim, args.emb_dim)  # 均值映射 (切空间)
        self.vae_logvar = nn.Linear(args.emb_dim, args.emb_dim)  # 方差映射 (切空间)
        self.decoder = HyperbolicDecoder(args)  # 双曲解码器

        def reparameterize(self, mu, logvar):
            """双曲空间中的重参数化 trick"""
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_tangent = mu + eps * std  # 切空间中的采样
            return self.manifold.expmap0(z_tangent, project=True)  # 映射到双曲空间

    def forward(self, adj, triples):
        x_user, x_item = self.emb_user.weight, self.emb_item.weight
        t_user, t_item = self.manifold.logmap0(x_user), self.manifold.logmap0(x_item)
        t_user_item = torch.cat([t_user, t_item], dim=0)
        m_user_item = self.transform(t_user_item)

        # VAE部分
        mu = self.vae_mu(m_user_item)
        logvar = self.vae_logvar(m_user_item)
        z = self.reparameterize(mu, logvar)  # 双曲空间中的潜在变量
        recon = self.decoder(z)  # 重构结果 (双曲空间)

        # 计算重构损失 (双曲空间距离)
        recon_loss = torch.mean(self.manifold.sqdist(recon, self.manifold.expmap0(m_user_item)))

        # 计算KL散度 (切空间中的正态分布KL)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss_user_item, out_user_item = self.margin_loss(m_user_item, adj, self.margin, triples)
        # 总损失
        total_loss = loss_user_item + self.vae_beta * (recon_loss + kl_loss)
        return total_loss

    def margin_loss(self, h, adj, margin, triples):
        result = [h]
        for i in range(self.num_layers):
            result.append(torch.spmm(adj, result[i]))
        out = sum(result[1:])
        out = self.manifold.expmap0(out, project=True)
        anchor_embs, pos_embs, neg_embs = out[triples[:, 0], :], out[triples[:, 1], :], out[triples[:, 2], :]
        pos_scores = self.manifold.sqdist(anchor_embs, pos_embs)
        neg_scores = self.manifold.sqdist(anchor_embs, neg_embs)
        loss = pos_scores - neg_scores + margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss, out

    def encode(self, adj):
        x_user, x_item = self.emb_user.weight, self.emb_item.weight
        t_user, t_item = self.manifold.logmap0(x_user), self.manifold.logmap0(x_item)
        t_user_item = torch.cat([t_user, t_item], dim=0)
        m_user_item = self.transform(t_user_item)
        result = [m_user_item]
        for i in range(self.num_layers):
            result.append(torch.spmm(adj, result[i]))
        out = sum(result[1:])
        return self.manifold.expmap0(out, project=True)

    def get_embs(self):
        x_user, x_item = self.emb_user.weight, self.emb_item.weight
        t_user, t_item = self.manifold.logmap0(x_user), self.manifold.logmap0(x_item)
        return self.transform(t_user), self.transform(t_item)

    def predict(self, h, train_csr, test_dict, eval_batch_num):
        arr = [10, 20]
        recall, ndcg = {}, {}
        item = h[np.arange(self.num_users, self.num_users + self.num_items), :]
        batch_size = (self.num_users // eval_batch_num) + 1
        all_probs = []
        for start in range(0, self.num_users, batch_size):
            end = min(start + batch_size, self.num_users)
            user_batch = h[np.arange(start, end), :]
            probs_batch = -1 * self.manifold.sqdist_multi(user_batch, item).detach().cpu().numpy()
            all_probs.append(probs_batch)
        probs = np.concatenate(all_probs, axis=0)
        probs[train_csr.nonzero()] = np.NINF
        ind = np.argpartition(probs, -20)[:, -20:]
        arr_ind = probs[np.arange(len(probs))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(probs)), ::-1]
        pred_list = ind[np.arange(len(probs))[:, None], arr_ind_argsort]
        all_ndcg = ndcg_func([*test_dict.values()], pred_list)
        for k in arr:
            recall[k] = recall_func(test_dict, pred_list, k)
            ndcg[k] = all_ndcg[k - 1]
        return recall, ndcg
