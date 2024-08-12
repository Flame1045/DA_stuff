<!-- <p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" alt="project-logo">
</p> -->
<p align="center">
    <h1 align="center">Officail Code for "Efficient Transfer Learning with Spatial Attention and Channel Communication for Unsupervised Domain Adaptation in Object Detection"</h1>
</p>
<!-- <p align="center">
    <em><code>► INSERT-TEXT-HERE</code></em>
</p> -->
<!-- <p align="center">
	<img src="https://img.shields.io/github/license/Flame1045/DA_stuff.git?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Flame1045/DA_stuff.git?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Flame1045/DA_stuff.git?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Flame1045/DA_stuff.git?style=default&color=0080ff" alt="repo-language-count">
<p> -->
<!-- <p align="center"> -->
	<!-- default option, no dependency badges. -->
<!-- </p> -->

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [ Overview](#-overview)
<!-- - [ Features](#-features) -->
- [ Repository Structure](#-repository-structure)
- [ Getting Started](#-getting-started)
  - [ Installation](#-installation)
  - [ Dataset preparation](#-installation)
  - [ Usage](#-usage)
- [ Config Details](#-Details)
<!-- - [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license) -->
- [ Future work](#-Future)
- [ Acknowledgments](#-acknowledgments)
</details>
<hr>

## [ Overview](#-overview)


#### Code release for the paper: https://drive.google.com/file/d/19ZY15MAgTQxjjQTUtHEn1vZDbd18qiny/view?usp=sharing

<img src=figures/PrpoArch.jpg>

Overview of our proposed architecture, which is based on DeformableDETR. The FG Adapter, depicted in the orange dotted box, is integrated into both the encoder and decoder of Deformable-DETR, while the CS Adapter, shown in the red dotted box, is only in the decoder. The domain classifier, indicated by the purple dotted box, is placed after the decoder. 

<!-- ##  Features

<code>► INSERT-TEXT-HERE</code> -->

---

##  [ Repository Structure](#-repository-structure)

```sh
└── DA_stuff/
    ├── README.md
    ├── experiment_saved
    │   ├── CITY2FOGGY_baseline_and_pretrained_on_source
    │   ├── CITY2FOGGY_oracle_and_trained_on_source_and_target
    │   ├── CITY2FOGGY_with_Dcls
    │   ├── CITY2FOGGY_with_Dcls_channel_mixing
    │   ├── CITY2FOGGY_with_Dcls_channel_mixing_spatail_attention
    │   ├── CITY2FOGGY_with_Dcls_spatail_attention
    │   ├── SIM2CITY_baseline_and_pretrained_on_source
    │   ├── SIM2CITY_oracle_and_trained_on_source_and_target
    │   ├── SIM2CITY_with_Dcls
    │   ├── SIM2CITY_with_Dcls_channel_mixing
    │   ├── SIM2CITY_with_Dcls_channel_mixing_spatail_attention
    │   └── SIM2CITY_with_Dcls_spatail_attention
    ├── figures
    │   ├── framework.png
    │   └── performance.png
    ├── projects
    │   ├── __init__.py
    │   ├── configs
    │   └── models
            ├── co_detr.py 
            └── _transformer.py
    ├── requirements.txt
    └── tools
        ├── train.py
        ├── experiment_CITY2FOGGY.sh
        └── experiment_SIMCITY.sh


```

##  [ Getting Started](#-getting-started)

**System Requirements:**

* **Python**: `version 3.7.11`
* **GPU**: `NVIDIA GeForce RTX 3090 Ti`

###  [ Installation](#-installation)

1. Clone the DA_stuff repository:
> ```console
> $ git clone https://github.com/Flame1045/DA_stuff.git
> ```

2. Change to the project directory:
> ```console
> $ cd DA_stuff
> ```

3. Install the dependencies:
> ```console
> $ conda create -n your_repo_name python==3.7.11 -y
> $ conda activate your_repo_name
> $ conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
> $ pip install -r requirements.txt
> $ pip install -U openmim
> $ mim install mmengine
> $ pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html 
> $ pip install -v -e .
> $ pip install fairscale
> $ pip install timm
> $ pip3 install natten==0.14.6+torch1110cu113 -f https://shi-labs.com/natten/wheels
> $ pip install tensorboard
> ```

4. Install pretrained weight:
Cilck  <a href="https://drive.google.com/file/d/1ezCc0LeGXj_7uTVBLKknJWufHAQt0TGL/view?usp=sharing" download>
  <button style="background-color: #4CAF50; border: none; color: white; padding: 5px 5px; text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px; cursor: pointer;">
    Here
  </button>
</a>
 to download and extract and merge files in experiment_saved.
It will looks like the structure below 
```sh
└── DA_stuff/
    ├── experiment_saved
        ├── pretrained
            └── XXX.pth
        └── XXX.pth
    ...

```

> ```console
> $ git clone https://github.com/Flame1045/DA_stuff.git
> ```

### [ Dataset preparation](#-installation)
Download SIM10k, CITYSCAPES, FOGGY CITYSCAPES from its offical website, convert it to COCO format. 
Make directory "./data/" and put dataset in below sturcture.
```sh
└── DA_stuff/
    ├── data
        └── coco
            ├──  City2Foggy_source # CITYSCAPES dataset
            ├──  City2Foggy_target # FOGGY CITYSCAPES dataset
            ├──  SIM2Real_source     # SIM10k dataset
            └──  SIM2Real_target     # CITYSCAPES dataset

```




###  [ Usage](#-usage)

<h4>Evaluation on <code>CITY2FOGGY</code></h4>
We provide six different experimental setups for evaluating the model on the CITY2FOGGY task. These setups are included in the script `tools/experiment_CITY2FOGGY.sh`.

The available experiments are:
- **CITY2FOGGY_baseline_and_pretrained_on_source**
- **CITY2FOGGY_oracle_and_trained_on_source_and_target**
- **CITY2FOGGY_with_Dcls_channel_mixing_spatial_attention**
- **CITY2FOGGY_with_Dcls**
- **CITY2FOGGY_with_Dcls_channel_mixing**
- **CITY2FOGGY_with_Dcls_spatial_attention**

To evaluate a specific experiment, uncomment (# python3 tools/test.py ...) in the desired experiment in the script and run the following command:

> ```console
> $  bash tools/experiment_CITY2FOGGY.sh 
> ```

<h4>Evaluation on <code>SIMCITY</code></h4>
We provide six different experimental setups for evaluating the model on the SIMCITY task. These setups are included in the script `tools/experiment_SIMCITY.sh`.

The available experiments are:
- **SIMCITY_baseline_and_pretrained_on_source**
- **SIMCITY_oracle_and_trained_on_source_and_target**
- **SIMCITY_with_Dcls_channel_mixing_spatial_attention**
- **SIMCITY_with_Dcls**
- **SIMCITY_with_Dcls_channel_mixing**
- **SIMCITY_with_Dcls_spatial_attention**

To evaluate a specific experiment, uncomment (# python3 tools/test.py ...) the desired experiment in the script and run the following command:

> ```console
> $  bash tools/experiment_SIMCITY.sh 
> ```

<h4>Training on <code>CITY2FOGGY</code></h4>
We provide six different experimental setups for training the model on the CITY2FOGGY task. These setups are included in the script `tools/experiment_CITY2FOGGY.sh`.

The available experiments are:
- **CITY2FOGGY_baseline_and_pretrained_on_source**
- **CITY2FOGGY_oracle_and_trained_on_source_and_target**
- **CITY2FOGGY_with_Dcls_channel_mixing_spatial_attention**
- **CITY2FOGGY_with_Dcls**
- **CITY2FOGGY_with_Dcls_channel_mixing**
- **CITY2FOGGY_with_Dcls_spatial_attention**

To train a specific experiment, uncomment (# CONFIG= ... to # done) in the desired experiment in the script and run the following command:

> ```console
> $  bash tools/experiment_CITY2FOGGY.sh 
> ```

<h4>Training on <code>SIMCITY</code></h4>
We provide six different experimental setups for training the model on the SIMCITY task. These setups are included in the script `tools/experiment_SIMCITY.sh`.

The available experiments are:
- **SIMCITY_baseline_and_pretrained_on_source**
- **SIMCITY_oracle_and_trained_on_source_and_target**
- **SIMCITY_with_Dcls_channel_mixing_spatial_attention**
- **SIMCITY_with_Dcls**
- **SIMCITY_with_Dcls_channel_mixing**
- **SIMCITY_with_Dcls_spatial_attention**

To train a specific experiment, uncomment (# CONFIG= ... to # done) the desired experiment in the script and run the following command:

> ```console
> $  bash tools/experiment_SIMCITY.sh 
> ```

---

##  [ Config Details](#-Details)

<details closed><summary>experiment_saved.SIM2CITY_baseline_and_pretrained_on_source</summary>

| File                                                                                                                                                                 | Summary                         |
| ---                                                                                                                                                                  | ---                             |
| [custom_sim2city_base.py](https://github.com/Flame1045/DA_stuff/tree/main/experiment_saved/SIM2CITY_baseline_and_pretrained_on_source/custom_sim2city_base.py) | <code>SIM2CITY pretrained on source</code> |

</details>

<details closed><summary>experiment_saved.SIM2CITY_with_Dcls</summary>

| File                                                                                                                                                                                               | Summary                         |
| ---                                                                                                                                                                                                | ---                             |
| [custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff/tree/main/experiment_saved/SIM2CITY_with_Dcls/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py) | <code>SIM2CITY with domain classifier</code> |

</details>

<details closed><summary>experiment_saved.SIM2CITY_with_Dcls_spatail_attention</summary>

| File                                                                                                                                                                                                                 | Summary                         |
| ---                                                                                                                                                                                                                  | ---                             |
| [custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff/tree/main/experiment_saved/SIM2CITY_with_Dcls_spatail_attention/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py) | <code>SIM2CITY with spatail attention</code> |

</details>


<details closed><summary>experiment_saved.SIM2CITY_with_Dcls_channel_mixing</summary>

| File                                                                                                                                                                                                              | Summary                         |
| ---                                                                                                                                                                                                               | ---                             |
| [custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff/tree/main/experiment_saved/SIM2CITY_with_Dcls_channel_mixing/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py) | <code>SIM2CITY with channel communication</code> |

</details>

<details closed><summary>experiment_saved.SIM2CITY_with_Dcls_channel_mixing_spatail_attention</summary>

| File                                                                                                                                                                                                                                | Summary                         |
| ---                                                                                                                                                                                                                                 | ---                             |
| [custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff/tree/main/experiment_saved/SIM2CITY_with_Dcls_channel_mixing_spatail_attention/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py) | <code>SIM2CITY with channel communication and spatail attention</code> |

</details>

<details closed><summary>experiment_saved.SIM2CITY_oracle_and_trained_on_source_and_target</summary>

| File                                                                                                                                                                                                                                             | Summary                         |
| ---                                                                                                                                                                                                                                              | ---                             |
| [custom_sim2city_unsupervised_base_wA_woCTBV2_B4_ORALCLE.py](https://github.com/Flame1045/DA_stuff/tree/main/experiment_saved/SIM2CITY_oracle_and_trained_on_source_and_target/custom_sim2city_unsupervised_base_wA_woCTBV2_B4_ORALCLE.py) | <code>SIM2CITY oralcle</code> |

</details>

<details closed><summary>experiment_saved.CITY2FOGGY_baseline_and_pretrained_on_source</summary>

| File                                                                                                                                                                       | Summary                         |
| ---                                                                                                                                                                        | ---                             |
| [custom_city2foggy_base.py](https://github.com/Flame1045/DA_stuff/tree/main/experiment_saved/CITY2FOGGY_baseline_and_pretrained_on_source/custom_city2foggy_base.py) | <code>CITY2FOGGY pretrained on source</code> |

</details>

<details closed><summary>experiment_saved.CITY2FOGGY_with_Dcls</summary>

| File                                                                                                                                                                                                     | Summary                         |
| ---                                                                                                                                                                                                      | ---                             |
| [custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff/tree/main/experiment_saved/CITY2FOGGY_with_Dcls/custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py) | <code>CITY2FOGGY with domain classifier</code> |

</details>

<details closed><summary>experiment_saved.CITY2FOGGY_with_Dcls_spatail_attention</summary>

| File                                                                                                                                                                                                                       | Summary                         |
| ---                                                                                                                                                                                                                        | ---                             |
| [custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff/tree/main/experiment_saved/CITY2FOGGY_with_Dcls_spatail_attention/custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py) | <code>CITY2FOGGY with spatail attention</code> |

</details>

<details closed><summary>experiment_saved.CITY2FOGGY_with_Dcls_channel_mixing</summary>

| File                                                                                                                                                                                                                    | Summary                         |
| ---                                                                                                                                                                                                                     | ---                             |
| [custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff/tree/main/experiment_saved/CITY2FOGGY_with_Dcls_channel_mixing/custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py) | <code>CITY2FOGGY with channel communication</code> |

</details>

<details closed><summary>experiment_saved.CITY2FOGGY_with_Dcls_channel_mixing_spatail_attention</summary>

| File                                                                                                                                                                                                                                      | Summary                         |
| ---                                                                                                                                                                                                                                       | ---                             |
| [custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff/tree/main/experiment_saved/CITY2FOGGY_with_Dcls_channel_mixing_spatail_attention/custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py) | <code>CITY2FOGGY with channel communication and spatail attention</code> |

</details>

<details closed><summary>experiment_saved.CITY2FOGGY_oracle_and_trained_on_source_and_target</summary>

| File                                                                                                                                                                                                                                                   | Summary                         |
| ---                                                                                                                                                                                                                                                    | ---                             |
| [custom_city2foggy_unsupervised_base_wA_woCTBV2_B4_ORALCLE.py](https://github.com/Flame1045/DA_stuff/tree/main/experiment_saved/CITY2FOGGY_oracle_and_trained_on_source_and_target/custom_city2foggy_unsupervised_base_wA_woCTBV2_B4_ORALCLE.py) | <code>CITY2FOGGY oralcle</code> |

</details>

---


## [ Future work](#-Future)
We add number of percent of target domain label in our proposed method to see if the adapter really need label to fine tune.

### Results of Sim10k (Ds) → Foggy Cityscapes (Dt):

| Percent of target domain label  |   AP         |
|---------------------|------------------------------------|
| 0 %  | 57.7 |
| 1 %  | XX.X |

### Results of Cityscapes (Ds) → Foggy Cityscapes (Dt)
| Percent of target domain label  |  person | rider | car | truck | bus | train | motorcycle | bicycle | mAP |
|--------------|-------|---|---|----|----|-------|---|--------|----|
| 0 %  | 42.7 | 48.5 | 56.8 | 32.7 | 47.0 | 32.5 | 33.0 | 42.6 | 42.0 |
| 1 %  | XX.X  | XX.X  | XX.X  | XX.X  | XX.X  | XX.X  | XX.X  | XX.X  | XX.X  |


---

##  [ Acknowledgments](#-acknowledgments)

- [SAPNetV2](https://github.com/Shuntw6096/SAPNetV2)
- Professor Wen-Hsien Fang and Professor Yie-Tarng Chen

[**Return**](#-overview)

---
