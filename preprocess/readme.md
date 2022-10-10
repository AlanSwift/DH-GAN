### notes
* raw resnet features: /home/shiina/data/iclr_further/img_resnet_feats


### VQA2.0
#### 1. process raw annotations: add ppl and resnet feature path
``` python
python -m preprocess.vqa2.process_raw
```
#### 2. annotate instances: make double hints
annotator.ipynb

#### 3. filter duplicate items and split the val and test
``` python
python -m preprocess.vqa2.make_split
```

### cocoqa
``` python
python -m preprocess.cocoqa.process_raw
```

