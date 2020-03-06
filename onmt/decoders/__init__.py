"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder
from onmt.decoders.multi_decoder import MultiRNNDecoder
from onmt.decoders.rowcol_decoder import ROWCOLRNNDecoder

str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder, "multi_decoder": MultiRNNDecoder, "rowcol": ROWCOLRNNDecoder}

__all__ = ["DecoderBase", "StdRNNDecoder", "InputFeedRNNDecoder", "str2dec"]
