from module.encoder.cnn_encoder import CNNEncoder
from module.encoder.text_encoder import TextEncoder, EncoderRNN


from module.encoder.graph_encoder_final_vh import GraphEncoderVisualHint


str2encoder = {"resnet_encoder": CNNEncoder, "text_encoder": TextEncoder, "rnn_encoder": EncoderRNN,
               "graph_encoder_final_vh": GraphEncoderVisualHint}

__all__ = ["GraphEncoder", "CNNEncoder", "str2encoder", "TextEncoder", "GraphEncoderCrossModal",
           "GraphEncoderClassifier", "GraphEncoderClassifierAlign", "GraphEncoderCrossModalLatent", "GraphEncoderCrossModalLatentControlVAE",
           "GraphEncoderCrossModalLatentControl", "GraphEncoderNoPrj", "GraphEncoderClassifierKD",
           "GraphEncoderClassifierVAE", "GraphEncoderGGNN", "GraphEncoderGGNNShare", "GraphEncoderGraphsage",
           "GraphEncoderGCNSpatial", "GraphEncoderGCNSpatialStd", "GraphEncoderGCNSpatialMultiTask",
           "GraphEncoderGCNSpatialMultiTaskAttnpos", "GraphEncoderAlign", "GraphEncoderAlign_1", "EncoderRNN",
           "GraphEncoderAlign_ans", "GraphEncoderAlign_ans_1", "GraphEncoderAlign_ans_2", "GraphEncoderContrastive",
           "GraphEncoderFinal", "GraphEncoderCombine", "GraphEncoderGolden", "GraphEncoderFix", "GraphEncoderNoise",
           "GraphEncoderClassifierCls", "GraphEncoderWithNoise", "GraphEncoderVisualHint", "GraphEncoderNoVH",
           "GraphEncoderNoAlign", "GraphEncoderNoAlignAnswer", "GraphEncoderVisualHint_1", "GraphEncoderRebuttalBase",
           "GraphEncoderRebuttalImplicitGraph", "GraphEncoderRebuttalImplicitGraphNoGNN"]
