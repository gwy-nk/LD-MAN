import torch.nn as nn
import torch.nn.functional as F
import torch, math


class MHAtt(nn.Module):
    def __init__(self, hidden, n_head, hidden_size_head, dropout_r=0.1):
        super(MHAtt, self).__init__()

        self.linear_v = nn.Linear(hidden, hidden)
        self.linear_k = nn.Linear(hidden, hidden)
        self.linear_q = nn.Linear(hidden, hidden)
        self.linear_merge = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(dropout_r)
        self.n_head = n_head
        self.hidden_size_head = hidden_size_head
        self.hidden = hidden

    def forward(self, v, k, q, mask):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.n_head,
            self.hidden_size_head
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.n_head,
            self.hidden_size_head
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.n_head,
            self.hidden_size_head
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.hidden
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1), -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)


class MHAtt_mine(nn.Module):
    def __init__(self, hidden, n_head, hidden_size_head, dropout_r=0.1):
        super(MHAtt, self).__init__()

        self.linear_v = nn.Linear(hidden, hidden)
        self.linear_k = nn.Linear(hidden, hidden)
        self.linear_q = nn.Linear(hidden, hidden)
        self.linear_merge = nn.Linear(hidden, hidden)

        self.dropout = nn.Dropout(dropout_r)
        self.n_head = n_head
        self.hidden_size_head = hidden_size_head
        self.hidden = hidden

        self.mlp = MLP(
            in_size=hidden_size_head,
            mid_size=hidden,
            out_size=1,
            dropout_r=0.1,
            use_relu=True
        )

        # self.linear_merge = nn.Linear(
        #     hidden,
        #     hidden
    #

    def forward(self, v, mask):

        n_batches = v.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.n_head,
            self.hidden_size_head
        )

        atted, att = self.att(v, mask)
        # atted, att = atted.transpose(1, 2).contiguous().view(
        #     n_batches,
        #     -1,
        #     self.hidden
        # )

        # atted = self.linear_merge(atted)

        return atted, att

    def att(self, value, mask):

        n_batches = value.size(0)

        att = self.mlp(value)
        att = att.masked_fill(
            mask.squeeze(1).squeeze(1).unsqueeze(2).unsqueeze(3),
            -1e9
        ).transpose(1, 2)
        att = F.softmax(att, dim=2)

        x_atted = torch.sum(att * value.transpose(1, 2), dim=2)

        x_atted = x_atted.view(
            n_batches,
            -1
        )

        # x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        # # return x_atted, att
        #
        # if mask is not None:
        #     scores = scores.masked_fill(mask, -1e9)
        #
        # att_map = F.softmax(scores, dim=-1)
        # att_map = self.dropout(att_map)

        return x_atted, att


class SenAttFlat(nn.Module):
    def __init__(self, hidden_size, flatten_mlp_size, flatten_out):
        super(SenAttFlat, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=flatten_mlp_size,
            out_size=1,
            dropout_r=0.1,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            hidden_size,
            flatten_out
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        x_atted = torch.sum(att * x, dim=1)

        # x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted, att


class AttFlat(nn.Module):
    def __init__(self, hidden_size, flatten_mlp_size, flatten_out):
        super(AttFlat, self).__init__()

        self.mlp = MLP(
            in_size=hidden_size,
            mid_size=flatten_mlp_size,
            out_size=1,
            dropout_r=0.1,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            hidden_size,
            flatten_out
        )

    def forward(self, x, x_mask):
        att = self.mlp(x)
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)

        x_atted = torch.sum(att * x, dim=1)

        # x_atted = torch.cat(att_list, dim=1)
        x_atted = self.linear_merge(x_atted)

        return x_atted, att


class MLP(nn.Module):
    def __init__(self, in_size, mid_size, out_size, dropout_r=0.1, use_relu=True):
        super(MLP, self).__init__()

        self.fc = FC(in_size, mid_size, dropout_r=dropout_r, use_relu=use_relu)
        self.linear = nn.Linear(mid_size, out_size)

    def forward(self, x):
        return self.linear(self.fc(x))


class FC(nn.Module):
    def __init__(self, in_size, out_size, dropout_r=0., use_relu=True):
        super(FC, self).__init__()
        self.dropout_r = dropout_r
        self.use_relu = use_relu

        self.linear = nn.Linear(in_size, out_size)

        if use_relu:
            self.relu = nn.ReLU(inplace=True)

        if dropout_r > 0:
            self.dropout = nn.Dropout(dropout_r)

    def forward(self, x):
        x = self.linear(x)

        if self.use_relu:
            x = self.relu(x)

        if self.dropout_r > 0:
            x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a_2 = nn.Parameter(torch.ones(size))
        self.b_2 = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class FFN(nn.Module):
    def __init__(self, input_size, mid_size, out_size, dropout_r=0.1):
        super(FFN, self).__init__()

        self.mlp = MLP(
            in_size=input_size,
            mid_size=mid_size,
            out_size=out_size,
            dropout_r=dropout_r,
            use_relu=True
        )

    def forward(self, x):
        return self.mlp(x)
