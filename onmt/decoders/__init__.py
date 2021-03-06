"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder, SimpleRNNDecoder
from onmt.decoders.multi_decoder import MultiRNNDecoder
from onmt.decoders.rowcol_decoder import ROWCOLRNNDecoder
from onmt.decoders.sam_decoder import SAM_Decoder

str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder, "multi_decoder": MultiRNNDecoder, "rowcol": ROWCOLRNNDecoder,
           "simple": SimpleRNNDecoder, "sam": SAM_Decoder}

__all__ = ["DecoderBase", "StdRNNDecoder", "InputFeedRNNDecoder", "str2dec"]
