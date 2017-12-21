# SAN implementation on VQA V2.0 dataset
Torch implementation of an attention-based visual question answering model ([Stacked Attention Networks for Image Question Answering, Yang et al., CVPR16][1]).

![Imgur](http://i.imgur.com/VbqIRZz.png)

1. [Train your own network]
    1. [Extract image features]
    2. [Preprocess VQA dataset]
    3. [Training]
2. [Running evaluation]
3. [Results]

The model looks at an image, reads a question, and comes up with an answer to the question and a heatmap of where it looked in the image to answer it.

The model/code also supports referring back to the image multiple times ([Stacked Attention][1]) before producing the answer. This is supported via a `num_attention_layers` parameter in the code (default = 1).

## Train your own network

### Preprocess VQA dataset

Pass `split` as `1` to train on `train` and evaluate on `val`, and `2` to train on `train`+`val` and evaluate on `test`.

```
cd data/
python vqa_preprocessing.py --download True --split 1
cd ..
```
```
python prepro.py --input_train_json data/vqa_raw_train.json --input_test_json data/vqa_raw_test.json --num_ans 1000
```

### Extract image features

Since we don't finetune the CNN, training is significantly faster if image features are pre-extracted. We use image features from VGG-19. The model can be downloaded and features extracted using:

```
sh scripts/download_vgg19.sh
th prepro_img.lua -image_root /path/to/coco/images/ -gpuid 0
```

### Training

```
th train.lua
```

### Running evaluation

```
model_path=checkpoints/model.t7 qa_h5=data/qa.h5 params_json=data/params.json img_test_h5=data/img_test.h5 th eval.lua
```

This will generate a JSON file containing question ids and predicted answers. To compute accuracy on `val`, use [VQA Evaluation Tools][13]. For `test`, submit to [VQA evaluation server on EvalAI][14].

## Results

**Format**: sets of 3 columns, col 1 shows original image, 2 shows 'attention' heatmap of where the model looks, 3 shows image overlaid with attention. Input question and answer predicted by model are shown below examples.
![](http://i.imgur.com/Q0byOyp.jpg)

### Quantitative Results

Trained on `train` for `val` accuracies, and trained on `train`+`val` for `test` accuracies.

#### VQA v2.0

| Method                | val     | test    |
| ------                | ---     | ----    |
| SAN-1                 | 53.15   | 55.28   |
| SAN-2                 | 52.82   | -       |
| [d-LSTM + n-I][4]     | 51.62   | 54.22   |
| [HieCoAtt][9]         | 54.57   | -       |
| [MCB][7]              | 59.14   | -       |

## References

- [Stacked Attention Networks for Image Question Answering][1], Yang et al., CVPR16
- [Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering][11], Goyal and Khot et al., CVPR17

## Acknowledgements

- Data preprocessing script borrowed from [VT-vision-lab/VQA_LSTM_CNN](https://github.com/GT-Vision-Lab/VQA_LSTM_CNN)
- Torch implementation of neural-vqa-attention from [abhshkdz](https://github.com/abhshkdz/neural-vqa-attention)



[1]: https://arxiv.org/abs/1511.02274
[2]: https://abhshkdz.mit-license.org/
[3]: https://computing.ece.vt.edu/~abhshkdz/neural-vqa-attention/figures/
[4]: https://github.com/VT-vision-lab/VQA_LSTM_CNN
[5]: http://visualqa.org/download.html
[6]: http://arxiv.org/abs/1505.00468
[7]: https://github.com/akirafukui/vqa-mcb
[8]: https://github.com/jnhwkim/MulLowBiVQA
[9]: https://github.com/jiasenlu/HieCoAttenVQA
[10]: https://computing.ece.vt.edu/~abhshkdz/neural-vqa-attention/pretrained/
[11]: https://arxiv.org/abs/1612.00837
[12]: https://computing.ece.vt.edu/~abhshkdz/vqa-hat/
[13]: https://github.com/VT-vision-lab/VQA
[14]: https://evalai.cloudcv.org/featured-challenges/1/overview
