import torch
import torch.nn as nn

from onmt.models.stacked_rnn import StackedLSTM, StackedGRU
from onmt.modules.multi_attention import MultiAttention
from onmt.modules.global_attention import GlobalAttention
from onmt.utils.rnn_factory import rnn_factory

from onmt.utils.misc import aeq


class DecoderBase(nn.Module):
    """Abstract class for decoders.

    Args:
        attentional (bool): The decoder returns non-empty attention.
    """

    def __init__(self, attentional=True):
        super(DecoderBase, self).__init__()
        self.attentional = attentional

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor.

        Subclasses should override this method.
        """

        raise NotImplementedError


class RNNDecoderBase(DecoderBase):
    """Base recurrent attention-based decoder class.


    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general", attn_func="softmax",
                 coverage_attn=False, context_gate=None,
                 copy_attn=False, dropout=0.0, embeddings=None,
                 reuse_copy_attn=False, copy_attn_type="general",add=None):
        super(RNNDecoderBase, self).__init__(
            attentional=attn_type != "none" and attn_type is not None)

        self.add = add

        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)

        # Set up the context gate.
        self.context_gate = None
        if context_gate is not None:
            self.context_gate = context_gate_factory(
                context_gate, self._input_size,
                hidden_size, hidden_size, hidden_size
            )

        # Set up the standard attention.
        self._coverage = coverage_attn
        if not self.attentional:
            if self._coverage:
                raise ValueError("Cannot use coverage term with no attention.")
            self.attn = None
        else:
            self.linear_out = nn.Linear(hidden_size * 2, hidden_size, bias=False)

            outsize = 512
            self.attn = MultiAttention(
                outsize,
                hidden_size, coverage=coverage_attn,
                attn_type=attn_type, attn_func=attn_func
            )

            if self.add:
                outsize = 512

                # self.gobleatten = GlobalAttention(hidden_size, coverage=coverage_attn,
                #                                   attn_type=attn_type, attn_func=attn_func)

            else:
                outsize = 512

            self.attn2 = MultiAttention(
                outsize,
                hidden_size, coverage=coverage_attn,
                attn_type=attn_type, attn_func=attn_func
            )


        if copy_attn and not reuse_copy_attn:
            if copy_attn_type == "none" or copy_attn_type is None:
                raise ValueError(
                    "Cannot use copy_attn with copy_attn_type none")
            self.copy_attn = MultiAttention(
                hidden_size, attn_type=copy_attn_type, attn_func=attn_func

            )
        else:
            self.copy_attn = None

        self._reuse_copy_attn = reuse_copy_attn and copy_attn
        if self._reuse_copy_attn and not self.attentional:
            raise ValueError("Cannot reuse copy attention with no attention.")

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.coverage_attn,
            opt.context_gate,
            opt.copy_attn,
            opt.dropout[0] if type(opt.dropout) is list
            else opt.dropout,
            embeddings,
            opt.reuse_copy_attn,
            opt.copy_attn_type,
            opt.add_m
           )

    def init_state(self, src, memory_bank, encoder_final):
        """Initialize decoder state with last state of the encoder."""
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).


            if self.bidirectional_encoder:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden
        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple(_fix_enc_hidden(enc_hid)
                                         for enc_hid in encoder_final)
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final), )

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)
        self.state["coverage"] = None

    def map_state(self, fn):
        self.state["hidden"] = tuple(fn(h, 1) for h in self.state["hidden"])
        self.state["input_feed"] = fn(self.state["input_feed"], 1)
        if self._coverage and self.state["coverage"] is not None:
            self.state["coverage"] = fn(self.state["coverage"], 1)

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self, tgt, memory_bank, memory_lengths=None, step=None,
                **kwargs):
        """
        Args:
            tgt (LongTensor): sequences of padded tokens
                 ``(tgt_len, batch, nfeats)``.
            memory_bank (FloatTensor): vectors from the encoder
                 ``(src_len, batch, hidden)``.
            memory_lengths (LongTensor): the padded source lengths
                ``(batch,)``.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(tgt_len, batch, hidden)``.
            * attns: distribution over src at each tgt
              ``(tgt_len, batch, src_len)``.
        """

        # c 的size是 bz, 512,attn是将整个w x h 内的值加权和为一个值，因为有512个通道，所以最后为512
        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, memory_bank, memory_lengths=memory_lengths)

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
        self.state["coverage"] = None
        if "coverage" in attns:
            self.state["coverage"] = attns["coverage"][-1].unsqueeze(0)

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            # for k in attns:
            #     if type(attns[k]) == list:
            #         attns[k] = torch.stack(attns[k])
        return dec_outs, attns

    # p_attn = (p_attn1, p_attn2)
    # attns["std"].append(p_attn)


    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.embeddings.update_dropout(dropout)


class MultiRNNDecoder(RNNDecoderBase):

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):
        """
        -multi_scale
        """
        # Additional args check.

        input_feed = self.state["input_feed"].squeeze(0)                    # ot-1
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.
        dec_outs = []
        attns = {}
        if self.attn is not None:
            attns["std"] = []
        if self.copy_attn is not None or self._reuse_copy_attn:
            attns["copy"] = []
        if self._coverage:
            attns["coverage"] = []

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]
        coverage = self.state["coverage"].squeeze(0) \
            if self.state["coverage"] is not None else None

        # Input feed concatenates hidden state with
        # input at every time step.
        #
        for emb_t in emb.split(1):
            decoder_input = torch.cat([emb_t.squeeze(0), input_feed], 1)   # decoder_input 对应论文[yt-1, ot-1] emb_t.squeeze(0) 为 yt-1,input_feed为 ot-1

            rnn_output, dec_state = self.rnn(decoder_input, dec_state)    # dec_state 对应论文中ht-1

            if self.attentional:
                c1, p_attn1 = self.attn(    # ot
                    rnn_output,
                    memory_bank[0].transpose(0, 1),
                    memory_lengths=memory_lengths)

                c2, p_attn2 = self.attn2(  # ot
                    rnn_output,
                    memory_bank[1].transpose(0, 1),
                    memory_lengths=memory_lengths)

                if self.add:
                    c = c1+c2

                else:
                    c = torch.cat([c1, c2], 2)
                # hidden使用行的hidden
                # c = c1+c2

                if rnn_output.dim() == 2:
                    one_step = True
                    rnn_output = rnn_output.unsqueeze(1)
                else:
                    one_step = False
                batch_, target_l, dim_ = rnn_output.size()
                # print('c', c.size())                    # c  torch.Size([20, 1, 768])
                # print('rnn_output', rnn_output.size())  # rnn_output torch.Size([20, 1, 768])
                # print('batch', batch_)                    # c  torch.Size([20, 1, 768])
                # print('target_l', target_l)                    # c  torch.Size([20, 1, 768])
                # print('dim_', dim_)                    # c  torch.Size([20, 1, 768])
                concat_c = torch.cat([c, rnn_output], 2).view(batch_ * target_l, dim_ * 2)  # ot = tanh(Wc[ht; ct])
                attn_h = self.linear_out(concat_c).view(batch_, target_l, dim_)
                attn_h = torch.tanh(attn_h)

                if one_step:
                    p_attn1 = p_attn1.squeeze(1)
                    p_attn2 = p_attn2.squeeze(1)
                    attn_h = attn_h.squeeze(1)

                else:
                    attn_h = attn_h.transpose(0, 1).contiguous()
                    p_attn1 = p_attn1.transpose(0, 1).contiguous()
                    p_attn2 = p_attn2.transpose(0, 1).contiguous()

                p_attn = (p_attn1, p_attn2)
                attns["std"].append(p_attn)
                decoder_output = attn_h
            else:
                decoder_output = rnn_output
            if self.context_gate is not None:
                # TODO: context gate should be employed
                # instead of second RNN transform.
                decoder_output = self.context_gate(
                    decoder_input, rnn_output, decoder_output
                )
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            dec_outs += [decoder_output]            # ot

            # Update the coverage attention.
            if self._coverage:
                coverage = p_attn if coverage is None else p_attn + coverage
                attns["coverage"] += [coverage]

            if self.copy_attn is not None:
                _, copy_attn = self.copy_attn(
                    decoder_output, memory_bank.transpose(0, 1))
                attns["copy"] += [copy_attn]
            elif self._reuse_copy_attn:
                attns["copy"] = attns["std"]

        return dec_state, dec_outs, attns           # dec_out ot

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        assert rnn_type != "SRU", "SRU doesn't support input feed! " \
            "Please set -input_feed 0!"
        stacked_cell = StackedLSTM if rnn_type == "LSTM" else StackedGRU
        return stacked_cell(num_layers, input_size, hidden_size, dropout)

    @property
    def _input_size(self):
        """Using input feed by concatenating input with attention vectors."""
        return self.embeddings.embedding_size + self.hidden_size

    def update_dropout(self, dropout):
        self.dropout.p = dropout
        self.rnn.dropout.p = dropout
        self.embeddings.update_dropout(dropout)

