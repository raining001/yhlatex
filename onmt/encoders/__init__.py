"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.image_encoder import ImageEncoder
from onmt.encoders.DenseNetEncoder import DenseNet

str2enc = {"img": ImageEncoder, "imgdense": DenseNet}

__all__ = ["EncoderBase" ,"str2enc"]
