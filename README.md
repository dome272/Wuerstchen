[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KeN407dItcjLcWdMLrByZ8mPa1MT2_DJ?usp=sharing)
# Würstchen (ICLR 2024, oral)
![huggingface-blog-post-thumbnail](https://github.com/dome272/Wuerstchen/assets/61938694/b4253c80-2a88-41a5-80e5-cadfe17a05e0)


## What is this?
Würstchen is a new framework for training text-conditional models by moving the computationally expensive text-conditional stage into a highly compressed latent space. Common approaches make use of a single stage compression, while Würstchen introduces another Stage that introduces even more compression. In total we have Stage A & B that are responsible for compressing images and Stage C that learns the text-conditional part in the low dimensional latent space. With that Würstchen achieves a 42x compression factor, while still reconstructing images faithfully. This enables training of Stage C to be fast and computationally cheap. We refer to [the paper](https://arxiv.org/abs/2306.00637) for details.

## Use Würstchen
You can use the model simply through the notebooks here. The [Stage B](https://github.com/dome272/wuerstchen/blob/main/w%C3%BCrstchen-stage-B.ipynb) notebook only for reconstruction and the [Stage C](https://github.com/dome272/wuerstchen/blob/main/w%C3%BCrstchen-stage-C.ipynb) notebook is for the text-conditional generation. You can also try the text-to-image generation on [Google Colab](https://colab.research.google.com/drive/1KeN407dItcjLcWdMLrByZ8mPa1MT2_DJ?usp=sharing).

### Using in 🧨 diffusers

Würstchen is fully integrated into the [`diffusers` library](https://huggingface.co/docs/diffusers). Here's how to use it: 

```python
# pip install -U transformers accelerate diffusers

import torch
from diffusers import AutoPipelineForText2Image
from diffusers.pipelines.wuerstchen import DEFAULT_STAGE_C_TIMESTEPS

pipe = AutoPipelineForText2Image.from_pretrained("warp-ai/wuerstchen", torch_dtype=torch.float16).to("cuda")

caption = "Anthropomorphic cat dressed as a fire fighter"
images = pipe(
    caption, 
    width=1024,
    height=1536,
    prior_timesteps=DEFAULT_STAGE_C_TIMESTEPS,
    prior_guidance_scale=4.0,
    num_images_per_prompt=2,
).images
```

Refer to the [official documentation](https://huggingface.co/docs/diffusers/main/en/api/pipelines/wuerstchen) to learn more. 

## Train your own Würstchen
Training Würstchen is considerably faster and cheaper than other text-to-image as it trains in a much smaller latent space of 12x12.
We provide training scripts for both [Stage B](https://github.com/dome272/wuerstchen/blob/main/train_stage_B.py) and [Stage C](https://github.com/dome272/wuerstchen/blob/main/train_stage_C.py). 

## Download Models
| Model           | Download                                             | Parameters      | Conditioning                       | Training Steps | Resolution |
|-----------------|------------------------------------------------------|-----------------|------------------------------------|--------------------|------|
| Würstchen v1    | [Hugging Face](https://huggingface.co/dome272/wuerstchen) | 1B (Stage C) + 600M (Stage B) + 19M (Stage A)  | CLIP-H-Text | 800.000| 512x512 |
| Würstchen v2    | [Hugging Face](https://huggingface.co/dome272/wuerstchen) | 1B (Stage C) + 600M (Stage B) + 19M (Stage A)  | CLIP-bigG-Text | 918.000| 1024x1024 |

## Acknowledgment
Special thanks to [Stability AI](https://stability.ai/) for providing compute for our research.

## Citation
If you use our approach in your research or were inspired by it, we would be thrilled if you cite our paper:
```
@inproceedings{
            pernias2024wrstchen,
            title={W\"urstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models},
            author={Pablo Pernias and Dominic Rampas and Mats Leon Richter and Christopher Pal and Marc Aubreville},
            booktitle={The Twelfth International Conference on Learning Representations},
            year={2024},
            url={https://openreview.net/forum?id=gU58d5QeGv}
      }
```
