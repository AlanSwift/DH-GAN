from module.decoder.seq_decoder import RNNDecoder
from module.decoder.seq_decoder_1 import BGFGRNNDecoder
from module.decoder.seq_decoder_2 import BGFGRNNDecoderAns
from module.decoder.seq_decoder_rl import RNNDecoderRL
from module.decoder.seq_decoder_seperate import RNNDecoderSep
from module.decoder.seq_decoder_image import RNNDecoderImage
from module.decoder.seq_decoder_graph import RNNDecoderGraph


str2decoder = {"hie_rnn_decoder_base": RNNDecoder, "hie_rnn_decoder_rl": RNNDecoderRL,
               "bgfg_rnn_decoder": BGFGRNNDecoder, "bgfg_rnn_decoder_ans": BGFGRNNDecoderAns,
               "hie_rnn_decoder_sep": RNNDecoderSep, "hie_rnn_decoder_image": RNNDecoderImage,
               "hie_rnn_decoder_graph": RNNDecoderGraph}
