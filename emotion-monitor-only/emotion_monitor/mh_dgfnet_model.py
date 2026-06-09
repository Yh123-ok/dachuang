import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGNN(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4, dropout=0.2, ffn_ratio=2):
        super().__init__()

        assert out_dim % heads == 0
        assert in_dim == out_dim

        self.heads = heads
        self.head_dim = out_dim // heads

        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        self.proj = nn.Linear(out_dim, out_dim)

        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(out_dim, ffn_ratio * out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_ratio * out_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, N, C = x.shape

        residual = x
        x_norm = self.norm1(x)

        q = self.q(x_norm).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(x_norm).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        v = self.v(x_norm).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            attn = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.dropout(out)

        x = residual + out
        x = x + self.ffn(self.norm2(x))

        return x


class DynamicGraphFusion(nn.Module):
    def __init__(self, dim, k_neighbors=10, dropout=0.3):
        super().__init__()

        self.k_neighbors = k_neighbors

        self.theta = nn.Linear(dim, dim)
        self.phi = nn.Linear(dim, dim)
        self.gcn_weight = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        k = min(self.k_neighbors, N)

        theta_x = self.theta(x)
        phi_x = self.phi(x)

        sim_matrix = torch.matmul(theta_x, phi_x.transpose(1, 2)) / (C ** 0.5)

        if k < N:
            mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
            topk_idx = torch.topk(sim_matrix, k=k, dim=-1).indices
            mask.scatter_(2, topk_idx, True)
            sim_matrix = sim_matrix.masked_fill(~mask, -1e4)

        adj = F.softmax(sim_matrix, dim=-1)
        adj = self.dropout(adj)

        gcn_x = self.gcn_weight(x)
        out = torch.matmul(adj, gcn_x)

        return self.norm(x + self.act(out))


class MH_DGFNet_Full(nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.3):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.regions = {
            'Frontal': [0, 1, 2, 3, 4, 11, 12, 13, 14, 15],
            'Parietal': [6, 7, 8, 20, 21, 22],
            'Temporal': [9, 10, 24, 25],
            'Occipital': [28, 29, 30],
            'Central': [5, 16, 17, 18, 19, 23, 26, 27, 31]
        }

        self.eeg_proj = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout)
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, 32, hidden_dim))

        self.eeg_gnn = nn.Sequential(
            AdaptiveGNN(hidden_dim, hidden_dim, heads=4, dropout=dropout),
            AdaptiveGNN(hidden_dim, hidden_dim, heads=4, dropout=dropout)
        )

        self.region_ops = nn.ModuleDict({
            r: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout)
            )
            for r in self.regions.keys()
        })

        self.ms_enc = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),

            nn.Linear(32 * 16, 8 * hidden_dim),
            nn.LayerNorm(8 * hidden_dim),
            nn.Dropout(dropout)
        )

        self.peri_enc = nn.Sequential(
            nn.Linear(55, 128),
            nn.LayerNorm(128),
            nn.ELU(),
            nn.Dropout(dropout),

            nn.Linear(128, 8 * hidden_dim),
            nn.LayerNorm(8 * hidden_dim),
            nn.Dropout(dropout)
        )

        self.vis_enc = nn.Sequential(
            nn.LayerNorm(512),
            nn.Dropout(dropout),

            nn.Linear(512, 4 * hidden_dim),
            nn.LayerNorm(4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.modality_embed = nn.Parameter(torch.zeros(4, hidden_dim))
        self.super_node = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.modality_embed, std=0.02)
        nn.init.trunc_normal_(self.super_node, std=0.02)

        self.dgf_layers = nn.Sequential(
            DynamicGraphFusion(hidden_dim, k_neighbors=12, dropout=dropout),
            DynamicGraphFusion(hidden_dim, k_neighbors=12, dropout=dropout)
        )

        self.v_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

        self.a_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, maps, stats, peri, visual):
        h_eeg = self.eeg_proj(stats) + self.pos_embed
        h_eeg = self.eeg_gnn(h_eeg)

        h_neuro = torch.stack([
            self.region_ops[r](h_eeg[:, idx, :].mean(dim=1))
            for r, idx in self.regions.items()
        ], dim=1)

        h_ms = self.ms_enc(maps).view(-1, 8, self.hidden_dim)
        h_peri = self.peri_enc(peri).view(-1, 8, self.hidden_dim)
        h_vis = self.vis_enc(visual).view(-1, 4, self.hidden_dim)

        h_neuro = h_neuro + self.modality_embed[0]
        h_ms = h_ms + self.modality_embed[1]
        h_peri = h_peri + self.modality_embed[2]
        h_vis = h_vis + self.modality_embed[3]

        combined_nodes = torch.cat([h_neuro, h_ms, h_peri, h_vis], dim=1)

        B = combined_nodes.size(0)
        super_n = self.super_node.expand(B, -1, -1)

        graph_nodes = torch.cat([super_n, combined_nodes], dim=1)

        fused_nodes = self.dgf_layers(graph_nodes)

        graph_repr = fused_nodes[:, 0] + fused_nodes[:, 1:].mean(dim=1)

        v_logits = self.v_head(graph_repr)
        a_logits = self.a_head(graph_repr)

        return v_logits, a_logits
