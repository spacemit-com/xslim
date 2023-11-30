class XQUANT_GLOBAL_CONFIGURATION:
    def __init__(self) -> None:
        import torch

        self.cuda_support = torch.cuda.is_available()

        self.default_block_size = 8

        self.fine_tune_epoch = 2

        self.loss_threshold = 0.05


XQUANT_CONFIG = XQUANT_GLOBAL_CONFIGURATION()
from .xquantizer import XQuantizer, AutoFinetuneLevel
