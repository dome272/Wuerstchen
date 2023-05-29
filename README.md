[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://colab.research.google.com/drive/1geY_Z8m8dyjrky6uwiMepwySTWkVYl1j?usp=sharing](https://colab.research.google.com/drive/1JMLGW1hjsFDDNFYf9e9ABEO5VEDts3K4?usp=sharing))
# Würstchen
 ![main-figure-github](https://github.com/dome272/wuerstchen/assets/61938694/5a047f1a-01ab-4747-b7ed-15c2d9969339)

## What is this?
Würstchen is a new framework for training text-conditional models by moving the computationally expensive text-conditional stage into a highly compressed latent space. Common approaches make use of a single stage compression, while Würstchen introduces another Stage that introduces even more compression. In total we have Stage A & B that are responsible for compressing images and Stage C that learns the text-conditional part in the low dimensional latent space. With that Würstchen achieves a 42x compression factor, while still reconstructing images faithfully. This enables training of Stage C to be fast and computationally cheap. We refer to [the paper](https://arxiv.org/abs/2211.07292) for details.

## Use Würstchen
You can use the model simply through the notebooks here. The [Stage B](https://github.com/dome272/wuerstchen/blob/main/w%C3%BCrstchen-stage-B.ipynb) notebook only for reconstruction and the [Stage C](https://github.com/dome272/wuerstchen/blob/main/w%C3%BCrstchen-stage-C.ipynb) notebook is for the text-conditional generation.

## Train your own Würstchen
We provide training scripts for both [Stage B](https://github.com/dome272/wuerstchen/blob/main/train_stage_B.py) and [Stage C](https://github.com/dome272/wuerstchen/blob/main/train_stage_C.py).

## Download Models
| Model           | Download                                             | Parameters      | Conditioning                       |
|-----------------|------------------------------------------------------|-----------------|------------------------------------|
| Würstchen v1    | [Huggingface](https://huggingface.co/dome272/wuerstchen) | 1B (Stage C) + 600M (Stage B) + 19M (Stage A)  | CLIP-H-Text                     |

## Acknowledgment
Special thanks to [Stability AI](https://stability.ai/) for providing compute for our research.
