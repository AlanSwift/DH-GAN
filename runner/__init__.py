# final runner
from runner.trainer_seqgan_bleu import TrainerSeqGANVHSample


str2trainer = {"trainer_seqgan_vh_sample": TrainerSeqGANVHSample,
               }

__all__ = ["str2trainer", "TrainerSeqGANVHSample"]
