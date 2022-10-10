# Learning to Generate Visual Questions with Noisy Supervision

============


Dependencies
------------
- pytorch v1.2
- sklearn
- torchtext
- Accelerator

Data preprocess
------------
Please refer to ``preprocess/`` folder. I will provide shell script soon.


Evaluation
------------
Please download the evaluation tool from [evaluation tool](https://github.com/graph4ai/graph4nlp/tree/master/graph4nlp/pytorch/modules/evaluation).

How to run
----------

Run with following:

#### Train
```bash
python -m main.py --config config/vqa2_max4/seqgan_vh_sample_1_rlratial0.9.yaml
```

TODO
-----------
Provide detailed data preprocess procedure (soon).

Citation
-----------
If you found this code useful, please consider citing the following papers.
```
@article{kai2021learning,
  title={Learning to Generate Visual Questions with Noisy Supervision},
  author={Kai, Shen and Wu, Lingfei and Tang, Siliang and Zhuang, Yueting and Ding, Zhuoye and Xiao, Yun and Long, Bo and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={11604--11617},
  year={2021}
}

@misc{
kai2021ask,
title={Ask Question with Double Hints:  Visual Question Generation with Answer-awareness and Region-reference},
author={Shen Kai and Lingfei Wu and Siliang Tang and Fangli Xu and Zhu Zhang and Yu Qiang and Yueting Zhuang},
year={2021},
url={https://openreview.net/forum?id=-WwaX9vKKt}
}
```