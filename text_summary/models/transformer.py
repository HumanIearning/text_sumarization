import torch
import torch.nn as nn


class MultiHead(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.Q_Linear = nn.Linear(d_model, d_model, bias=False)
        self.K_Linear = nn.Linear(d_model, d_model, bias=False)
        self.V_Linear = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model, bias=False)

        self.attn = Attention()

    def forward(self, Q, K, V, mask=None):
        QWs = self.Q_Linear(Q).split(self.d_model // self.num_heads, dim=-1)
        KWs = self.Q_Linear(K).split(self.d_model // self.num_heads, dim=-1)
        VWs = self.Q_Linear(V).split(self.d_model // self.num_heads, dim=-1)

        QWs = torch.cat(QWs, dim=0)
        KWs = torch.cat(KWs, dim=0)
        VWs = torch.cat(VWs, dim=0)

        if mask is not None:
            mask = torch.cat([mask for _ in range(self.num_heads)], dim=0)

        c = self.attn(QWs, KWs, VWs, mask, self.d_model // self.num_heads)

        c = c.split(Q.size(0), dim=0)
        c = self.linear(torch.cat(c, dim=-1))

        return c


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None, dk=64):
        w = torch.bmm(Q, K.transpose(1, 2))

        if mask is not None:
            assert mask.size() == w.size()
            w.masked_fill_(mask, -float('inf'))

        w = self.softmax(w / dk ** .5)
        c = torch.bmm(w, V)

        return c


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout_p=.1, use_leaky_relu=False):
        super().__init__()

        self.attn = MultiHead(d_model, num_heads)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout_p)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.fc_norm = nn.LayerNorm(d_model)
        self.fc_dropout = nn.Dropout(dropout_p)


    def forward(self, x, mask):

        z = self.attn_norm(x)
        z = x + self.attn_dropout(self.attn(z, z, z, mask))

        z = z + self.fc_dropout(self.fc(self.fc_norm(z)))

        return z, mask


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout_p=.1, use_leaky_relu=False):
        super().__init__()

        self.masked_attn = MultiHead(d_model, num_heads)

        self.attn = MultiHead(d_model, num_heads)

        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.LayerNorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, K_V, pad_mask, prev, look_ahead_mask):
        if prev is None:
            z = self.LayerNorm(x)
            z = x + self.dropout(self.masked_attn(z, z, z, mask=look_ahead_mask))
        else:
            normed_prev = self.LayerNorm(prev)
            z = self.LayerNorm(x)
            z = x + self.dropout(self.masked_attn(z, normed_prev, normed_prev, mask=None))


        normed_K_V = self.LayerNorm(K_V)
        z = z + self.dropout(self.attn(self.LayerNorm(z), normed_K_V, normed_K_V, mask=pad_mask))


        z = z + self.dropout(self.fc(self.LayerNorm(z)))

        return z, K_V, pad_mask, prev, look_ahead_mask

class MySequential(nn.Sequential):

    def forward(self, *x):
        # nn.Sequential class does not provide multiple input arguments and returns.
        # Thus, we need to define new class to solve this issue.
        # Note that each block has same function interface.

        for module in self._modules.values():
            x = module(*x)

        return x

class Transformer(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 d_model,
                 num_heads=8,
                 num_enc_blocks=6,
                 num_dec_blocks=6,
                 dropout_p=.1,
                 use_leaky_relu=False,
                 max_length=2048):

        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        self.num_heads = num_heads
        self.num_enc_blocks = num_enc_blocks
        self.num_dec_blocks = num_dec_blocks
        self.dropout_p = dropout_p
        self.max_length = max_length

        super().__init__()

        self.emb_enc = nn.Embedding(input_size, d_model)
        self.emb_dec = nn.Embedding(output_size, d_model)
        self.emb_dropout = nn.Dropout(dropout_p)

        self.pos_enc = self.generate_pos_enc(d_model, max_length)

        self.encoder = MySequential(
            *[EncoderBlock(d_model, num_heads, dropout_p, use_leaky_relu)
                                       for _ in range(num_enc_blocks)]
        )
        self.decoder = MySequential(
            *[DecoderBlock(d_model, num_heads, dropout_p, use_leaky_relu)
                                       for _ in range(num_dec_blocks)]
        )
        self.generator = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, output_size),
            nn.LogSoftmax(dim=-1)
        )

    def positional_encoding(self, x, init_pos=0):
        assert x.size(-1) == self.pos_enc.size(-1)
        assert x.size(1) + init_pos <= self.max_length

        pos_enc = self.pos_enc[init_pos: x.size(1) + init_pos].unsqueeze(0)
        x = x + pos_enc.to(x.device)

        return x

    @torch.no_grad()
    def generate_pos_enc(self, d_model, max_length):
        pos_enc = torch.FloatTensor(max_length, d_model).zero_()

        pos = torch.arange(0, max_length).unsqueeze(-1).float()
        dim = torch.arange(0, d_model // 2).float()

        pos_enc[:, 0::2] = torch.sin(pos / 1e+4 ** (dim / float(d_model)))
        pos_enc[:, 1::2] = torch.cos(pos / 1e+4 ** (dim / float(d_model)))

        return pos_enc

    @torch.no_grad()
    def generate_pad_mask(self, x, length):
        mask = []

        max_length = max(length)

        for l in length:
            if l is not max_length:
                mask += [torch.cat([x.new_ones(1, l).zero_(),
                                    x.new_ones(1, max_length - l)], dim=-1)]
            else:
                mask += [x.new_ones(1, l).zero_()]

        mask = torch.cat(mask, dim=0).bool()

        return mask

    def forward(self, x, y):
        # |x[0]| = (bs, len)  one hot vector
        # |x[1]| = (bs, 1)  length
        # |y| = (bs, len)

        with torch.no_grad():
            mask = self.generate_pad_mask(x[0], x[1])
            # |mask| = (bs, max_length) 이때 max_length는 각 배치에서의 max_length임
            x = x[0]

            mask_enc = mask.unsqueeze(1).expand(*x.size(), mask.size(-1))
            # |mask_enc| = (bs, len, max_length)
            mask_dec = mask.unsqueeze(1).expand(*y.size(), mask.size(-1))
            # |mask_dec| = (bs, len, max_length)

        z = self.emb_dropout(self.positional_encoding(self.emb_enc(x)))
        # |z| = (bs, len, d_model)

        z, _ = self.encoder(z, mask_enc)

        with torch.no_grad():
            look_ahead_mask = torch.triu(x.new_ones((y.size(1), y.size(1))), diagonal=1).bool()

            look_ahead_mask = look_ahead_mask.expand(y.size(0), *look_ahead_mask.size())

        h = self.emb_dropout(self.positional_encoding(self.emb_dec(y)))
        h, _, _, _, _ = self.decoder(h, z, mask_dec, None, look_ahead_mask)

        y_hat = self.generator(h)
        # |y_hat| = (bs, len, output_size)

        return y_hat
















# class Attention(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, Q, K, V, mask=None, dk=64):
#         # |Q| = (batch_size, m, hidden_size)
#         # |K| = |V| = (batch_size, n, hidden_size)
#         # |mask| = (batch_size, m, n)
#
#         w = torch.bmm(Q, K.transpose(1, 2))
#         # |w| = (batch_size, m, n)
#         if mask is not None:
#             assert w.size() == mask.size()
#             w.masked_fill_(mask, -float('inf'))
#
#         w = self.softmax(w / (dk**.5))
#         c = torch.bmm(w, V)
#         # |c| = (batch_size, m, hidden_size)
#
#         return c
#
#
# class MultiHead(nn.Module):
#
#     def __init__(self, hidden_size, n_splits):
#         super().__init__()
#
#         self.hidden_size = hidden_size
#         self.n_splits = n_splits
#
#         # Note that we don't have to declare each linear layer, separately.
#         self.Q_linear = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.K_linear = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.V_linear = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
#
#         self.attn = Attention()
#
#     def forward(self, Q, K, V, mask=None):
#         # |Q|    = (batch_size, m, hidden_size)
#         # |K|    = (batch_size, n, hidden_size)
#         # |V|    = |K|
#         # |mask| = (batch_size, m, n)
#
#         QWs = self.Q_linear(Q).split(self.hidden_size // self.n_splits, dim=-1)
#         KWs = self.K_linear(K).split(self.hidden_size // self.n_splits, dim=-1)
#         VWs = self.V_linear(V).split(self.hidden_size // self.n_splits, dim=-1)
#         # |QW_i| = (batch_size, m, hidden_size / n_splits)
#         # |KW_i| = |VW_i| = (batch_size, n, hidden_size / n_splits)
#
#         # By concatenating splited linear transformed results,
#         # we can remove sequential operations,
#         # like mini-batch parallel operations.
#         QWs = torch.cat(QWs, dim=0)
#         KWs = torch.cat(KWs, dim=0)
#         VWs = torch.cat(VWs, dim=0)
#         # |QWs| = (batch_size * n_splits, m, hidden_size / n_splits)
#         # |KWs| = |VWs| = (batch_size * n_splits, n, hidden_size / n_splits)
#
#         if mask is not None:
#             mask = torch.cat([mask for _ in range(self.n_splits)], dim=0)
#             # |mask| = (batch_size * n_splits, m, n)
#
#         c = self.attn(
#             QWs, KWs, VWs,
#             mask=mask,
#             dk=self.hidden_size // self.n_splits,
#         )
#         # |c| = (batch_size * n_splits, m, hidden_size / n_splits)
#
#         # We need to restore temporal mini-batchfied multi-head attention results.
#         c = c.split(Q.size(0), dim=0)
#         # |c_i| = (batch_size, m, hidden_size / n_splits)
#         c = self.linear(torch.cat(c, dim=-1))
#         # |c| = (batch_size, m, hidden_size)
#
#         return c
#
#
# class EncoderBlock(nn.Module):
#
#     def __init__(
#         self,
#         hidden_size,
#         n_splits,
#         dropout_p=.1,
#         use_leaky_relu=False,
#     ):
#         super().__init__()
#
#         self.attn = MultiHead(hidden_size, n_splits)
#         self.attn_norm = nn.LayerNorm(hidden_size)
#         self.attn_dropout = nn.Dropout(dropout_p)
#
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 4),
#             nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
#             nn.Linear(hidden_size * 4, hidden_size),
#         )
#         self.fc_norm = nn.LayerNorm(hidden_size)
#         self.fc_dropout = nn.Dropout(dropout_p)
#
#     def forward(self, x, mask):
#         # |x|    = (batch_size, n, hidden_size)
#         # |mask| = (batch_size, n, n)
#
#         # Post-LN:
#         # z = self.attn_norm(x + self.attn_dropout(self.attn(Q=x,
#         #                                                    K=x,
#         #                                                    V=x,
#         #                                                    mask=mask)))
#         # z = self.fc_norm(z + self.fc_dropout(self.fc(z)))
#
#         # Pre-LN:
#         z = self.attn_norm(x)
#         z = x + self.attn_dropout(self.attn(Q=z,
#                                             K=z,
#                                             V=z,
#                                             mask=mask))
#         z = z + self.fc_dropout(self.fc(self.fc_norm(z)))
#         # |z| = (batch_size, n, hidden_size)
#
#         return z, mask
#
#
# class DecoderBlock(nn.Module):
#
#     def __init__(
#         self,
#         hidden_size,
#         n_splits,
#         dropout_p=.1,
#         use_leaky_relu=False,
#     ):
#         super().__init__()
#
#         self.masked_attn = MultiHead(hidden_size, n_splits)
#         self.masked_attn_norm = nn.LayerNorm(hidden_size)
#         self.masked_attn_dropout = nn.Dropout(dropout_p)
#
#         self.attn = MultiHead(hidden_size, n_splits)
#         self.attn_norm = nn.LayerNorm(hidden_size)
#         self.attn_dropout = nn.Dropout(dropout_p)
#
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size * 4),
#             nn.LeakyReLU() if use_leaky_relu else nn.ReLU(),
#             nn.Linear(hidden_size * 4, hidden_size),
#         )
#         self.fc_norm = nn.LayerNorm(hidden_size)
#         self.fc_dropout = nn.Dropout(dropout_p)
#
#     def forward(self, x, key_and_value, mask, prev, future_mask):
#         # |key_and_value| = (batch_size, n, hidden_size)
#         # |mask|          = (batch_size, m, n)
#
#         # In case of inference, we don't have to repeat same feed-forward operations.
#         # Thus, we save previous feed-forward results.
#         if prev is None: # Training mode
#             # |x|           = (batch_size, m, hidden_size)
#             # |prev|        = None
#             # |future_mask| = (batch_size, m, m)
#             # |z|           = (batch_size, m, hidden_size)
#
#             # Post-LN:
#             # z = self.masked_attn_norm(x + self.masked_attn_dropout(
#             #     self.masked_attn(x, x, x, mask=future_mask)
#             # ))
#
#             # Pre-LN:
#             z = self.masked_attn_norm(x)
#             z = x + self.masked_attn_dropout(
#                 self.masked_attn(z, z, z, mask=future_mask)
#             )
#         else: # Inference mode
#             # |x|           = (batch_size, 1, hidden_size)
#             # |prev|        = (batch_size, t - 1, hidden_size)
#             # |future_mask| = None
#             # |z|           = (batch_size, 1, hidden_size)
#
#             # Post-LN:
#             # z = self.masked_attn_norm(x + self.masked_attn_dropout(
#             #     self.masked_attn(x, prev, prev, mask=None)
#             # ))
#
#             # Pre-LN:
#             normed_prev = self.masked_attn_norm(prev)
#             z = self.masked_attn_norm(x)
#             z = x + self.masked_attn_dropout(
#                 self.masked_attn(z, normed_prev, normed_prev, mask=None)
#             )
#
#         # Post-LN:
#         # z = self.attn_norm(z + self.attn_dropout(self.attn(Q=z,
#         #                                                    K=key_and_value,
#         #                                                    V=key_and_value,
#         #                                                    mask=mask)))
#
#         # Pre-LN:
#         normed_key_and_value = self.attn_norm(key_and_value)
#         z = z + self.attn_dropout(self.attn(Q=self.attn_norm(z),
#                                             K=normed_key_and_value,
#                                             V=normed_key_and_value,
#                                             mask=mask))
#         # |z| = (batch_size, m, hidden_size)
#
#         # Post-LN:
#         # z = self.fc_norm(z + self.fc_dropout(self.fc(z)))
#
#         # Pre-LN:
#         z = z + self.fc_dropout(self.fc(self.fc_norm(z)))
#         # |z| = (batch_size, m, hidden_size)
#
#         return z, key_and_value, mask, prev, future_mask
#
#
# class MySequential(nn.Sequential):
#
#     def forward(self, *x):
#         # nn.Sequential class does not provide multiple input arguments and returns.
#         # Thus, we need to define new class to solve this issue.
#         # Note that each block has same function interface.
#
#         for module in self._modules.values():
#             x = module(*x)
#
#         return x
#
#
# class Transformer(nn.Module):
#
#     def __init__(
#         self,
#         input_size,
#         output_size,
#         d_model,
#         num_heads=8,
#         num_enc_blocks=6,
#         num_dec_blocks=6,
#         dropout_p=.1,
#         use_leaky_relu=False,
#         max_length=2048
#     ):
#         self.input_size = input_size
#         self.hidden_size = d_model
#         self.output_size = output_size
#         self.n_splits = num_heads
#         self.n_enc_blocks = num_enc_blocks
#         self.n_dec_blocks = num_dec_blocks
#         self.dropout_p = dropout_p
#         self.max_length = max_length
#
#         super().__init__()
#
#         self.emb_enc = nn.Embedding(input_size, d_model)
#         self.emb_dec = nn.Embedding(output_size, d_model)
#         self.emb_dropout = nn.Dropout(dropout_p)
#
#         self.pos_enc = self._generate_pos_enc(d_model, max_length)
#
#         self.encoder = MySequential(
#             *[EncoderBlock(
#                 d_model,
#                 self.n_splits,
#                 dropout_p,
#                 use_leaky_relu,
#               ) for _ in range(self.n_enc_blocks)],
#         )
#         self.decoder = MySequential(
#             *[DecoderBlock(
#                 d_model,
#                 self.n_splits,
#                 dropout_p,
#                 use_leaky_relu,
#               ) for _ in range(self.n_dec_blocks)],
#         )
#         self.generator = nn.Sequential(
#             nn.LayerNorm(d_model), # Only for Pre-LN Transformer.
#             nn.Linear(d_model, output_size),
#             nn.LogSoftmax(dim=-1),
#         )
#
#     @torch.no_grad()
#     def _generate_pos_enc(self, hidden_size, max_length):
#         enc = torch.FloatTensor(max_length, hidden_size).zero_()
#         # |enc| = (max_length, hidden_size)
#
#         pos = torch.arange(0, max_length).unsqueeze(-1).float()
#         dim = torch.arange(0, hidden_size // 2).unsqueeze(0).float()
#         # |pos| = (max_length, 1)
#         # |dim| = (1, hidden_size // 2)
#
#         enc[:, 0::2] = torch.sin(pos / 1e+4**dim.div(float(hidden_size)))
#         enc[:, 1::2] = torch.cos(pos / 1e+4**dim.div(float(hidden_size)))
#
#         return enc
#
#     def _position_encoding(self, x, init_pos=0):
#         # |x| = (batch_size, n, hidden_size)
#         # |self.pos_enc| = (max_length, hidden_size)
#         assert x.size(-1) == self.pos_enc.size(-1)
#         assert x.size(1) + init_pos <= self.max_length
#
#         pos_enc = self.pos_enc[init_pos:init_pos + x.size(1)].unsqueeze(0)
#         # |pos_enc| = (1, n, hidden_size)
#         x = x + pos_enc.to(x.device)
#
#         return x
#
#     @torch.no_grad()
#     def _generate_mask(self, x, length):
#         mask = []
#
#         max_length = max(length)
#         for l in length:
#             if max_length - l > 0:
#                 # If the length is shorter than maximum length among samples,
#                 # set last few values to be 1s to remove attention weight.
#                 mask += [torch.cat([x.new_ones(1, l).zero_(),
#                                     x.new_ones(1, (max_length - l))
#                                     ], dim=-1)]
#             else:
#                 # If length of sample equals to maximum length among samples,
#                 # set every value in mask to be 0.
#                 mask += [x.new_ones(1, l).zero_()]
#
#         mask = torch.cat(mask, dim=0).bool()
#         # |mask| = (batch_size, max_length)
#
#         return mask
#
#     def forward(self, x, y):
#         # |x[0]| = (batch_size, n)
#         # |y|    = (batch_size, m)
#
#         # Mask to prevent having attention weight on padding position.
#         with torch.no_grad():
#             mask = self._generate_mask(x[0], x[1])
#             # |mask| = (batch_size, n)
#             x = x[0]
#
#             mask_enc = mask.unsqueeze(1).expand(*x.size(), mask.size(-1))
#             mask_dec = mask.unsqueeze(1).expand(*y.size(), mask.size(-1))
#             # |mask_enc| = (batch_size, n, n)
#             # |mask_dec| = (batch_size, m, n)
#
#         z = self.emb_dropout(self._position_encoding(self.emb_enc(x)))
#         z, _ = self.encoder(z, mask_enc)
#         # |z| = (batch_size, n, hidden_size)
#
#         # Generate future mask
#         with torch.no_grad():
#             future_mask = torch.triu(x.new_ones((y.size(1), y.size(1))), diagonal=1).bool()
#             # |future_mask| = (m, m)
#             future_mask = future_mask.unsqueeze(0).expand(y.size(0), *future_mask.size())
#             # |fwd_mask| = (batch_size, m, m)
#
#         h = self.emb_dropout(self._position_encoding(self.emb_dec(y)))
#         h, _, _, _, _ = self.decoder(h, z, mask_dec, None, future_mask)
#         # |h| = (batch_size, m, hidden_size)
#
#         y_hat = self.generator(h)
#         # |y_hat| = (batch_size, m, output_size)
#
#         return y_hat
