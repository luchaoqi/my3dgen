<!-- # MultiModal-DeepFake
[TPAMI 2024 & CVPR 2023] PyTorch code for DGM4: Detecting and Grounding Multi-Modal Media Manipulation and Beyond -->

<div align="center">

<h1>My3DGen: A Scalable Personalized 3D Generative Model <br>
🌟 WACV 2025 Oral Presentation 🌟</h1>

<div>
    <a href="http://luchaoqi.github.io/" target="_blank">Luchao Qi<sup>1</sup></a>
    <a href="https://scholar.google.com/citations?user=g2CuYi4AAAAJ&hl=en" target="_blank">Jiaye Wu<sup>2</sup></a>
    <a href="https://www.linkedin.com/in/annie-w-928955101/" target="_blank">Annie N. Wang<sup>1</sup></a>
    <a href="https://mcmvmc.github.io/" target="_blank">Shengze Wang<sup>1</sup></a>
    <a href="https://www.cs.unc.edu/~ronisen//" target="_blank">Roni Sengupta<sup>2</sup></a>
</div>
<div>
    <sup>1</sup>University of North Carolina at Chapel Hill
    <br>
    <sup>2</sup>University of Maryland
</div>

<h4 align="center">
  <a href="https://my3dgen.github.io/" target='_blank'>[Project Page]</a> |
  <a href="https://www.youtube.com/watch?v=wTvWGFRh_Lg" target='_blank'>[Video]</a> |
  <a href="https://arxiv.org/abs/2307.05468" target='_blank'>[Arxiv]</a>
  <!-- <a href="https://arxiv.org/pdf/2309.14203.pdf" target='_blank'>[TPAMI Paper]</a> |
  <a href="https://huggingface.co/datasets/rshaojimmy/DGM4" target='_blank'>[Dataset]</a> -->
</h4>
<br>

<img src='./docs/teaser.jpg' width='100%'>

</div>

<!-- ## Updates
- [02/2024] Extension paper has been accepted by TPAMI.
- [01/2024] Dataset link has been updated with hugginface.
- [09/2023] Arxiv extension paper released.
- [04/2023] Trained checkpoint is updated.
- [04/2023] Arxiv paper released.
- [04/2023] Project page and video are released.
- [04/2023] Code and dataset are released. -->

## Introduction
This is the official implementation of *My3DGen: A Scalable Personalized 3D Generative Model*. We personalize a pretrained GAN-based model (EG3D) using a few (~50) selfies of an individual without full finetuning, enabling scalable personalization in a real-world scenario.

The framework of the proposed My3DGen model:

<div align="center">
<img src='./docs/architecture.jpg' width='90%'>
</div>




## Installation

<!-- ### Download
```
mkdir code
cd code
git clone https://github.com/rshaojimmy/MultiModal-DeepFake.git
cd MultiModal-DeepFake
``` -->


### Environment
We build our code on top of the [EG3D](https://github.com/NVlabs/eg3d) repository. Please refer to EG3D to setup the environment & checkpoints and run the following commands after setup the EG3D environment.
```
conda activate eg3d
pip install lpips
```


## Dataset Preparation

### Images
We use the dataset of celebrities from [images-of-celebs](https://github.com/images-of-celebs/images-of-celebs).

We follow [EG3D's processing pipeline](https://github.com/NVlabs/eg3d/blob/main/dataset_preprocessing/ffhq/preprocess_in_the_wild.py) to preprocess the images-of-celebs dataset.


### Latent codes
We originally follow the [oneThousand1000/EG3D-projector: An unofficial inversion code of eg3d.](https://github.com/oneThousand1000/EG3D-projector) to generate the latent codes of the images-of-celebs dataset. You may use more recent SOTA inversion methods to generate the latent codes e.g.
> [[2303.13497] TriPlaneNet: An Encoder for EG3D Inversion](https://arxiv.org/abs/2303.13497)  
> [[2303.12326] Make Encoder Great Again in 3D GAN Inversion through Geometry and Occlusion-Aware Encoding](https://arxiv.org/abs/2303.12326)

```
└── datasets
    ├── Barack
    │   ├── image_0.png
    │   ├── image_0_latent.npy
    │   ├── ...
    │   ├── image_n.png
    │   ├── image_n_latent.npy
    │   └── dataset.json
    ├── Morgan
    │   ├── image_0.png
    │   ├── image_0_latent.npy
    │   ├── ...
    │   ├── image_n.png
    │   ├── image_n_latent.npy
    │   └── dataset.json
    ├── ...
    └── Celebrity_N
        ├── image_0.png
        ├── image_0_latent.npy
        ├── ...
        ├── image_n.png
        ├── image_n_latent.npy
        └── dataset.json
```

## Training

Go to the `./eg3d` directory and modify the following command to train the model:
```
python train.py --outdir=<output_dir> \
--data=<dataset_dir e.g. ./datasets/Barack> \
--resume=<pretrained_model_path e.g. ./networks/ffhqrebalanced512-128.pkl> \
--cfg=<cfg_file e.g. ffhq> \
--batch=<batch_size e.g. 4> \
--gpus=<num_gpus e.g. 4> \
--snap=<num_snapshots e.g. 5> \
--kimg=<num_kimg e.g. 500> \
--lora=<lora_rank e.g. 1> \
--lora_alpha=<lora_alpha e.g. 1> \
--adalora=False \
--freeze_render=False
```


## Testing
After training, you can run various downstream tasks.

For inversion, you can refer to [danielroich/PTI: Official Implementation for "Pivotal Tuning for Latent-based editing of Real Images" (ACM TOG 2022) https://arxiv.org/abs/2106.05744](https://github.com/danielroich/PTI).

For editing, you can refer to [google/mystyle](https://github.com/google/mystyle).

## Citation
If you find this work useful for your research, please kindly cite our paper:
```
@misc{qi2024my3dgenscalablepersonalized3d,
      title={My3DGen: A Scalable Personalized 3D Generative Model}, 
      author={Luchao Qi and Jiaye Wu and Annie N. Wang and Shengze Wang and Roni Sengupta},
      year={2024},
      eprint={2307.05468},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2307.05468}, 
}
```
