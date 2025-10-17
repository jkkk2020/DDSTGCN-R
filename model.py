import torch
import torch.nn as nn
import torch.nn.functional as F


class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # x: [B, C, V, T], A: [W, V] or [B, W, V]
        if A.dim() == 2:
            out = torch.einsum('ncvl,wv->ncwl', (x, A))
        else:
            # batched A: [B, W, V]
            out = torch.einsum('ncvl,nwv->ncwl', (x, A))
        return out.contiguous()


class d_nconv(nn.Module):
    def __init__(self):
        super(d_nconv, self).__init__()

    def forward(self, x, A):
        # x: [B, C, W, T], A: [B, V, W] or [V, W]
        if A.dim() == 2:
            out = torch.einsum('ncwl,vw->ncvl', (x, A))
        else:
            out = torch.einsum('ncwl,nvw->ncvl', (x, A))
        return out.contiguous()


class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)


class linear_(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear_, self).__init__()
        # kernel (1,2) dilation=2 to mimic original
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 2), dilation=2, padding=(0, 0), stride=(1, 1),
                                   bias=True)

    def forward(self, x):
        return self.mlp(x)


class dgcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=2):
        super(dgcn, self).__init__()
        self.d_nconv = d_nconv()
        self.order = order
        c_in = (order * 3 + 1) * c_in
        self.mlp = linear_(c_in, c_out)
        self.dropout = dropout

    def forward(self, x, support):
        # x: [B, C, V, T]
        out = [x]
        for a in support:
            x1 = self.d_nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.d_nconv(x1, a)
                out.append(x2)
                x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class hgcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=2):
        super(hgcn, self).__init__()
        self.nconv = nconv()
        self.order = order
        c_in = (order + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout

    def forward(self, x, G):
        out = [x]
        x1 = self.nconv(x, G)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, G)
            out.append(x2)
            x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class hgcn_edge_At(nn.Module):
    def __init__(self, c_in, c_out, dropout, order=1):
        super(hgcn_edge_At, self).__init__()
        self.nconv = nconv()
        self.order = order
        c_in = (order + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout

    def forward(self, x, G):
        # x expected shape consistent with caller
        out = [x]
        x1 = self.nconv(x, G)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, G)
            out.append(x2)
            x1 = x2
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class spatial_attention(nn.Module):
    def __init__(self, in_channels, num_of_timesteps, num_of_edge, num_of_vertices):
        super(spatial_attention, self).__init__()
        # 使用较小 hidden 避免维度问题
        hidden = max(4, in_channels // 2)
        # 不强制 W1/W2 长度匹配固定 T，使用可学习缩放 + 动态时间窗
        self.W1_scale = nn.Parameter(torch.randn(1), requires_grad=True)
        self.W2_scale = nn.Parameter(torch.randn(1), requires_grad=True)
        self.W3 = nn.Parameter(torch.randn(in_channels, hidden), requires_grad=True)
        self.W4 = nn.Parameter(torch.randn(in_channels, hidden), requires_grad=True)
        # out_conv 映射 hidden*2 -> in_channels
        self.out_conv = nn.Conv2d(hidden * 2, in_channels, kernel_size=(1, 1))

        # 初始化
        nn.init.xavier_uniform_(self.W3)
        nn.init.xavier_uniform_(self.W4)
        nn.init.kaiming_uniform_(self.out_conv.weight, a=0.2)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, idx, idy):
        """
        x: [B, C, N, T]
        idx, idy: 1D LongTensor of length E specifying edges
        returns: S shape [B, E, C]
        """
        # ensure idx/idy on same device
        idx = idx.to(x.device)
        idy = idy.to(x.device)

        B, C, N, T = x.shape
        # reorder to [B, N, T, C]
        x_ntc = x.permute(0, 2, 3, 1)  # [B, N, T, C]

        # 动态时间权重：两种线性模式，乘以可学习缩放因子
        # 这里使用简单可导的权重：linearly spaced [0..1] 和 [1..0]，再乘可学习系数
        w1 = (torch.linspace(0.0, 1.0, T, device=x.device).view(1, 1, T, 1) * self.W1_scale)
        w2 = (torch.linspace(1.0, 0.0, T, device=x.device).view(1, 1, T, 1) * self.W2_scale)

        # 加权求和时间维 -> (B, N, C)
        # sum over T: result [B, N, C]
        lhs_time = torch.sum(x_ntc * w1, dim=2)  # [B, N, C]
        rhs_time = torch.sum(x_ntc * w2, dim=2)  # [B, N, C]

        # 通道映射 -> [B, N, hidden]
        lhs = torch.matmul(lhs_time, self.W3)  # [B, N, hidden]
        rhs = torch.matmul(rhs_time, self.W4)  # [B, N, hidden]

        # 选出边端点特征 -> [B, E, hidden]
        lhs_edge = lhs[:, idx, :]
        rhs_edge = rhs[:, idy, :]

        # concat -> [B, E, hidden*2]
        summed = torch.cat([lhs_edge, rhs_edge], dim=-1)

        # conv expects [B, C_in, E, 1]
        conv_in = summed.permute(0, 2, 1).unsqueeze(-1)  # [B, hidden*2, E, 1]
        out = self.out_conv(conv_in)  # [B, C, E, 1]
        out = out.squeeze(-1).permute(0, 2, 1)  # [B, E, C]
        return out


class ddstgcn(nn.Module):
    def __init__(self, batch_size, H_a, H_b, G0, G1, indices, G0_all, G1_all, H_T_new, lwjl, num_nodes,
                 dropout=0.3, supports=None, in_dim=1, out_dim=12, residual_channels=40, dilation_channels=40,
                 skip_channels=320, end_channels=640, kernel_size=2, blocks=3, layers=1):
        super(ddstgcn, self).__init__()

        self.batch_size = batch_size
        self.H_a = H_a
        self.H_b = H_b
        self.G0 = G0
        self.G1 = G1
        self.indices = indices  # expected shape [2, E] or two 1D tensors
        self.G0_all = G0_all
        self.G1_all = G1_all
        self.lwjl = lwjl

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        # learnable vectors
        self.edge_node_vec1 = nn.Parameter(torch.rand(self.H_a.size(1), 10), requires_grad=True)
        self.edge_node_vec2 = nn.Parameter(torch.rand(10, self.H_a.size(0)), requires_grad=True)
        self.node_edge_vec1 = nn.Parameter(torch.rand(self.H_a.size(0), 10), requires_grad=True)
        self.node_edge_vec2 = nn.Parameter(torch.rand(10, self.H_a.size(1)), requires_grad=True)
        self.hgcn_w_vec_edge_At_forward = nn.Parameter(torch.rand(self.H_a.size(1)), requires_grad=True)
        self.hgcn_w_vec_edge_At_backward = nn.Parameter(torch.rand(self.H_a.size(1)), requires_grad=True)

        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        # layers lists
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.dgconv = nn.ModuleList()
        self.SAt_forward = nn.ModuleList()
        self.SAt_backward = nn.ModuleList()
        self.hgconv_edge_At_forward = nn.ModuleList()
        self.hgconv_edge_At_backward = nn.ModuleList()
        self.bn_g = nn.ModuleList()

        # start conv
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.supports = supports or []
        self.num_nodes = num_nodes

        receptive_field = 1
        self.supports_len = 0
        self.supports_len += len(self.supports)
        self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes), requires_grad=True)
        self.supports_len += 1

        # build blocks
        for b in range(blocks):
            additional_scope = kernel_size
            new_dilation = 2
            for i in range(layers):
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation))
                self.gate_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))

                # num_of_timesteps argument not fixed; SpatialAttention handles dynamic T
                self.SAt_forward.append(spatial_attention(residual_channels, 0, self.indices.size(1), num_nodes))
                self.SAt_backward.append(spatial_attention(residual_channels, 0, self.indices.size(1), num_nodes))
                receptive_field += (additional_scope * 2)
                self.hgconv_edge_At_forward.append(hgcn_edge_At(residual_channels, 1, dropout))
                self.hgconv_edge_At_backward.append(hgcn_edge_At(residual_channels, 1, dropout))
                self.bn_g.append(nn.BatchNorm2d(residual_channels))

                # fix: append dgconv now (was missing)
                self.dgconv.append(dgcn(dilation_channels, residual_channels, dropout, order=2))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
        self.bn_start = nn.BatchNorm2d(in_dim, affine=False)

        # NOTE: do NOT create tensors here that you will mutate in forward() — keep them local in forward
        # self.new_supports_w would have been problematic if stored and mutated as module attribute

    def forward(self, input):
        """
        input: [B, C, N, T]
        returns: [B, out_dim, N, 1]
        """
        B, C, N, T = input.size()
        if T < self.receptive_field:
            x = nn.functional.pad(input, (self.receptive_field - T, 0, 0, 0))
        else:
            x = input

        x = self.bn_start(x)
        x = self.start_conv(x)
        skip = 0

        # adaptive adjacency (static across layers in this forward pass)
        adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # [N, N]
        adp_new = adp.unsqueeze(0).repeat(B, 1, 1)  # [B, N, N]
        new_supports = self.supports + [adp]

        # compute G0G1 edge matrices (these may be used by hgconv edge At)
        # keep as local; these are not mutated
        G0G1_edge_At_forward = self.G0_all @ (torch.diag_embed(self.hgcn_w_vec_edge_At_forward)) @ self.G1_all
        G0G1_edge_At_backward = self.G0_all @ (torch.diag_embed(self.hgcn_w_vec_edge_At_backward)) @ self.G1_all

        # IMPORTANT: create fresh local adjacency containers for each iteration to avoid in-place modification
        device = x.device
        E = self.indices.size(1)
        # ensure idx/idy are 1D long tensors on the correct device
        idx = self.indices[0].to(device)
        idy = self.indices[1].to(device)

        # iterate layers
        total_layers = self.blocks * self.layers
        for i in range(total_layers):
            # --- Forward edge extraction ---
            # SpatialAttention expects x in shape [B, C, N, T] (we supply x)
            batch_edge_forward = self.SAt_forward[i](x, idx, idy)  # [B, E, C]
            # convert to shape expected by hgconv_edge_At: unsqueeze and transpose as original code
            batch_edge_forward_in = batch_edge_forward.unsqueeze(3).transpose(1, 2)  # [B, C, E, 1]
            batch_edge_forward_out = self.hgconv_edge_At_forward[i](batch_edge_forward_in, G0G1_edge_At_forward)
            # collapse to [B, E] or [B, E, 1] depending; keep [B, E]
            batch_edge_forward_vals = torch.sigmoid(torch.squeeze(batch_edge_forward_out))  # [B, E] or [B, E, ?] -> squeeze

            # create a fresh forward adjacency batch for this iteration (avoid reusing same tensor)
            forward_medium = torch.eye(self.num_nodes, device=device).unsqueeze(0).repeat(B, 1, 1)  # [B, N, N]
            # assign edge values into the fresh tensor (this mutates forward_medium but it's local/fresh)
            # ensure batch_edge_forward_vals shape is [B, E]
            if batch_edge_forward_vals.dim() == 1:
                # single batch edge values (rare) -> make [B, E]
                batch_edge_forward_vals = batch_edge_forward_vals.unsqueeze(0).repeat(B, 1)
            forward_medium[:, idx, idy] = batch_edge_forward_vals

            # --- Backward edge extraction ---
            batch_edge_backward = self.SAt_backward[i](x, idx, idy)
            batch_edge_backward_in = batch_edge_backward.unsqueeze(3).transpose(1, 2)
            batch_edge_backward_out = self.hgconv_edge_At_backward[i](batch_edge_backward_in, G0G1_edge_At_backward)
            batch_edge_backward_vals = torch.sigmoid(torch.squeeze(batch_edge_backward_out))  # [B, E]

            backward_medium = torch.eye(self.num_nodes, device=device).unsqueeze(0).repeat(B, 1, 1)  # fresh
            if batch_edge_backward_vals.dim() == 1:
                batch_edge_backward_vals = batch_edge_backward_vals.unsqueeze(0).repeat(B, 1)
            backward_medium[:, idx, idy] = batch_edge_backward_vals

            # assemble new_supports_w as local list for this iteration (do not store on the module)
            # new_supports_w[0] = forward_medium * (static new_supports[0] if exists)
            # new_supports_w[1] = backward_medium^T * (static new_supports[1] if exists)
            # new_supports_w[2] = adp_new
            new_supports_w = [None, None, None]
            # optionally multiply by provided static supports to mask (if available)
            if len(new_supports) > 0:
                try:
                    # If new_supports[0] exists and broadcastable, multiply, else just assign
                    new_supports_w[0] = forward_medium * new_supports[0].unsqueeze(0) if getattr(new_supports[0], "dim", None) and new_supports[0].dim() == 2 else forward_medium * new_supports[0]
                except Exception:
                    new_supports_w[0] = forward_medium
                try:
                    # new_supports[1] may need transpose
                    candidate = backward_medium.transpose(1, 2)
                    new_supports_w[1] = candidate * (new_supports[1].unsqueeze(0) if getattr(new_supports[1], "dim", None) and new_supports[1].dim() == 2 else new_supports[1])
                except Exception:
                    new_supports_w[1] = backward_medium.transpose(1, 2)
            else:
                new_supports_w[0] = forward_medium
                new_supports_w[1] = backward_medium.transpose(1, 2)
            new_supports_w[2] = adp_new  # adaptive adjacency kept (no mutation)

            # ---------- TCN Gate ----------
            residual = x
            filter_out = self.filter_convs[i](residual)
            filter_out = torch.tanh(filter_out)
            gate_out = self.gate_convs[i](residual)
            gate_out = torch.sigmoid(gate_out)
            x = filter_out * gate_out

            # ---------- DGCN (uses new_supports_w) ----------
            x = self.dgconv[i](x, new_supports_w)
            x = self.bn_g[i](x)

            # residual connection (align time dim)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

            # skip connection
            s = x
            s = self.skip_convs[i](s)
            # avoid in-place ops on skip: always create new object
            try:
                # align time dims: take last s.time length from skip if skip exists
                skip = skip[:, :, :, -s.size(3):]
            except Exception:
                skip = 0
            skip = s + skip

        # final transforms
        x = F.leaky_relu(skip)
        x = F.leaky_relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        # ensure we return time dim = 1 at the end (last time step)
        return x[:, :, :, -1:].contiguous()
