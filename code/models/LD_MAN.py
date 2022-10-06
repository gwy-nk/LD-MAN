import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from models.module import MHAtt, AttFlat, FFN, LayerNorm


class HiMAN(nn.Module):
    def __init__(self, n_classes, vocab_size, emb_size, word_rnn_size, sentence_rnn_size, word_rnn_layers,
                 sentence_rnn_layers, word_att_size, sentence_att_size, options, dropout=0.5,
                 pretrained_mat=None, img_global_size=None):
        super(HiMAN, self).__init__()

        self.sentence_attention = AttentionSentRNN(
            batch_size=options['batch_sz'],
            sent_gru_hidden=sentence_rnn_size,
            word_gru_hidden=word_rnn_size,
            # img_number=options['img_num'],
            # s_number=options['sen_limit'],
            n_classes=n_classes,
            bidirectional=True,
            option=options,
            img_global_size=img_global_size,
            sen_limit=options['sen_limit']
        )

        self.word_attention = AttentionWordRNN(
            batch_size=options['batch_sz'],
            pred_embedding_mat=pretrained_mat,
            word_gru_hidden=word_rnn_size,
            bidirectional=True,
            option=options
        )

        self.sent_limit = options['sen_limit']
        self.word_limit = options['word_limit']
        self.img_num = options['img_num']

    def forward(self, x_t, sen_num, sen_len,
                x_v_g, x_v_s, x_v_o, x_v_p, x_img_dis):
        s = None

        for i in range(self.sent_limit):
            _s, state_word, _ = self.word_attention(x_t[:, i, :], sen_len[:, i])

            if s is None:
                s = _s.unsqueeze(1)
            else:
                s = torch.cat((s, _s.unsqueeze(1)), 1)
        predict = self.sentence_attention(s, sen_num, x_v_g, x_v_s,
                                          x_v_o, x_img_dis)
        return predict


class AttentionWordRNN(nn.Module):
    def __init__(self, batch_size, pred_embedding_mat, word_gru_hidden, bidirectional=True, option=None):

        super(AttentionWordRNN, self).__init__()

        self.batch_size = batch_size
        self.num_tokens = pred_embedding_mat.shape[0]
        self.embed_size = pred_embedding_mat.shape[1]
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        dropout_r = option['dropout']

        self.word_attn = MHAtt(option['hidden'], option['n_head'], option['hidden_size_head'])

        self.lookup = nn.Embedding(self.num_tokens, self.embed_size)
        self.lookup.weight.data.copy_(torch.from_numpy(pred_embedding_mat))

        if bidirectional:
            self.word_gru = nn.GRU(self.embed_size, word_gru_hidden, bidirectional=True, batch_first=True)
            self.h_h_linear = nn.Linear(2 * word_gru_hidden, 2 * word_gru_hidden)
            self.h_1_linear = nn.Linear(2 * word_gru_hidden, 1, bias=False)

            self.mhatt = MHAtt(2 * option['hidden'], option['n_head'], 2 * option['hidden_size_head'], dropout_r=dropout_r)
            self.ffn = FFN(2 * option['hidden'], 2 * option['mid_hidden'], 2 * option['hidden'], dropout_r=dropout_r)
            self.dropout1 = nn.Dropout(dropout_r)
            self.norm1 = LayerNorm(2 * option['hidden'])
            self.dropout2 = nn.Dropout(dropout_r)
            self.norm2 = LayerNorm(2 * option['hidden'])

            self.attflat_lang = AttFlat(2 * option['hidden'], 2 * option['hidden'], 2 * option['hidden'])

        else:
            self.word_gru = nn.GRU(self.embed_size, word_gru_hidden, bidirectional=False, batch_first=True)
            self.h_h_linear = nn.Linear(word_gru_hidden, word_gru_hidden)
            self.h_1_linear = nn.Linear(word_gru_hidden, 1, bias=False)

            self.mhatt = MHAtt(option['hidden'], option['n_head'], option['hidden_size_head'], dropout_r=dropout_r)
            self.ffn = FFN(option['hidden'], option['mid_hidden'], option['hidden'], dropout_r=dropout_r)
            self.dropout1 = nn.Dropout(dropout_r)
            self.norm1 = LayerNorm(option['hidden'])
            self.dropout2 = nn.Dropout(dropout_r)
            self.norm2 = LayerNorm(option['hidden'])

            self.attflat_lang = AttFlat(option['hidden'], option['hidden'], option['hidden'])

    def forward(self, words, sen_len):

        mask = self.make_mask(words.unsqueeze(2))

        # embeddings
        embedded = self.lookup(words)
        # word level gru
        output_word, state_word = self.word_gru(embedded)

        word_attn_vectors, word_attn_norm = self.attflat_lang(
            output_word,
            mask
        )

        return word_attn_vectors, state_word, word_attn_norm

    def init_hidden(self):
        if self.bidirectional:
            return Variable(torch.zeros(2, self.batch_size, self.word_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.word_gru_hidden))

    def make_mask(self, words):
        return (torch.sum(
            torch.abs(words),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)

        # return sen_len == 0


class AttentionSentRNN(nn.Module):
    def __init__(self, batch_size, sent_gru_hidden, word_gru_hidden, n_classes, bidirectional=True,
                 option=None,
                 img_global_size=None, sen_limit=None):

        super(AttentionSentRNN, self).__init__()

        self.sen_limit = sen_limit

        self.batch_size = batch_size
        self.sent_gru_hidden = sent_gru_hidden
        self.n_classes = n_classes
        self.word_gru_hidden = word_gru_hidden
        self.bidirectional = bidirectional
        dropout_r = option['dropout']

        if bidirectional:
            self.sent_gru = nn.GRU(2 * word_gru_hidden, sent_gru_hidden, bidirectional=True)
            self.h_h_linear = nn.Linear(2 * sent_gru_hidden, 2 * sent_gru_hidden)
            self.h_1_linear = nn.Linear(2 * sent_gru_hidden, 1, bias=False)
            self.final_linear = nn.Linear(2 * sent_gru_hidden, n_classes)

            self.mhatt1 = MHAtt(2 * option['hidden'], option['n_head'], 2 * option['hidden_size_head'], dropout_r=dropout_r)
            self.mhatt2 = MHAtt(2 * option['hidden'], option['n_head'], 2 * option['hidden_size_head'], dropout_r=dropout_r)
            self.ffn = FFN(2 * option['hidden'], 2 * option['mid_hidden'], 2 * option['hidden'], dropout_r=dropout_r)
            self.dropout1 = nn.Dropout(dropout_r)
            self.norm1 = LayerNorm(2 * option['hidden'])
            self.dropout2 = nn.Dropout(dropout_r)
            self.norm2 = LayerNorm(2 * option['hidden'])

            self.attflat_lang = AttFlat(2 * option['hidden'], 2 * option['hidden'], 2 * option['hidden'])

            self.linear2hid_img = nn.Linear(
                2 * img_global_size,
                2 * option['hidden']
            )

            self.convert_c = nn.Linear(
                4 * option['hidden'],
                2 * option['hidden']
            )

            self.hidden = 2 * option['hidden']

        else:
            self.sent_gru = nn.GRU(word_gru_hidden, sent_gru_hidden, bidirectional=False)
            self.h_h_linear = nn.Linear(sent_gru_hidden, sent_gru_hidden)
            self.h_1_linear = nn.Linear(sent_gru_hidden, 1, bias=False)
            self.final_linear = nn.Linear(sent_gru_hidden, n_classes)

            self.mhatt1 = MHAtt(option['hidden'], option['n_head'], option['hidden_size_head'], dropout_r=dropout_r)
            self.mhatt2 = MHAtt(option['hidden'], option['n_head'], option['hidden_size_head'], dropout_r=dropout_r)
            self.ffn = FFN(option['hidden'], option['mid_hidden'], option['hidden'], dropout_r=dropout_r)
            self.dropout1 = nn.Dropout(dropout_r)
            self.norm1 = LayerNorm(option['hidden'])
            self.dropout2 = nn.Dropout(dropout_r)
            self.norm2 = LayerNorm(option['hidden'])

            self.attflat_lang = AttFlat(option['hidden'], option['hidden'], option['hidden'])

            self.linear2hid_img = nn.Linear(
                2*img_global_size,
                option['hidden']
            )

            self.convert_c = nn.Linear(
                2 * option['hidden'],
                option['hidden']
            )

            self.hidden = option['hidden']

        self.img_lamda = option['img_lamda']
        self.pos_lamda = option['pos_lamda']
        self.mul_lamda = option['mul_lamda']

    def forward(self, word_attention_vectors,
                sen_num,
                x_v_g, x_v_s, x_v_o, x_img_dis):

        output_sent, state_sent = self.sent_gru(word_attention_vectors)
        img_mask = self.make_mask_img(x_img_dis)
        lang_mask = self.make_mask_lang(sen_num)

        # TODO: To revise this image features
        img_feat = self.linear2hid_img(torch.cat((x_v_g, x_v_s), -1))

        """
        FLATTEN SENTENCE
        """
        # Sentence Self Attention
        _, sen_attn_norm = self.attflat_lang(
            output_sent,
            lang_mask
        )
        sen_self_weight = sen_attn_norm.squeeze(-1).unsqueeze(1).\
            expand(sen_attn_norm.size(0), img_feat.size(1), sen_attn_norm.size(1))

        sent_feat = output_sent.unsqueeze(1).expand(output_sent.size(0), img_feat.size(1),
                                                    output_sent.size(1), output_sent.size(2))
        img_feat_expand = img_feat.unsqueeze(2).expand(img_feat.size(0), img_feat.size(1),
                                                       output_sent.size(1), img_feat.size(2))

        # GMU combine
        c_vector = F.sigmoid(
            self.convert_c(
                torch.cat(
                    (sent_feat, img_feat_expand), -1
                )
            )
        )
        mul_feat = torch.mul(c_vector, sent_feat) + \
                   torch.mul((1 - c_vector), img_feat_expand)

        # Image Content Attention
        sematic_weight = F.softmax(
            self.h_1_linear(mul_feat).squeeze().masked_fill(lang_mask, -1e9), -1
        )

        position_weight = F.softmax(
            (40-x_img_dis).masked_fill(img_mask, -1e9), -1
        )

        total_feat = self.pos_lamda * position_weight + self.mul_lamda * sematic_weight + sen_self_weight
        # total_feat = 10 * position_weight + 8 * sematic_weight + sen_self_weight
        # img_lamda,pos_lamda,mul_lamda:1 8 7
        x_atted = torch.sum(
            torch.mul(
                total_feat.unsqueeze(3).expand(total_feat.size(0), total_feat.size(1), total_feat.size(2), self.hidden),
                sent_feat
            ),
            -2
        )

        x_atted = x_atted + self.img_lamda * img_feat
        # x_atted = x_atted + 10 * img_feat

        final_mask = self.make_mask(x_img_dis)

        sent_attn_vectors, sen_attn_norm = self.attflat_lang(
            x_atted,
            final_mask
        )

        final_map = self.final_linear(sent_attn_vectors.squeeze(0))
        return final_map

    def init_hidden(self):
        if self.bidirectional:
            return Variable(torch.zeros(2, self.batch_size, self.sent_gru_hidden))
        else:
            return Variable(torch.zeros(1, self.batch_size, self.sent_gru_hidden))

    def make_mask_lang(self, sen_num):

        sen_num_matrix = torch.zeros((sen_num.size(0), self.sen_limit)).cuda()

        for index in range(sen_num.size(0)):
            sen_num_matrix[index, 0:int(((sen_num[index])).item())] = 1

        return (sen_num_matrix == 0).unsqueeze(1)

    def make_mask_img(self, x_img_dis):
        return torch.abs(x_img_dis) == 0

    def make_mask(self, x_img_dis):
        return (torch.sum(
            torch.abs(x_img_dis),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)