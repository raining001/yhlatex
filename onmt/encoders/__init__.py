"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.image_encoder import ImageEncoder
from onmt.encoders.DenseNetEncoder import DenseNet
from onmt.encoders.rowcol_encoder import RCEncoder
from onmt.encoders.trainsformer_encoder import TransformerEncoder
from onmt.encoders.gru_encoder import GRUEncoder

str2enc = {"img": ImageEncoder, "imgdense": DenseNet, "rc": RCEncoder, "transformer": TransformerEncoder, 'gru': GRUEncoder}

__all__ = ["EncoderBase" ,"str2enc"]
