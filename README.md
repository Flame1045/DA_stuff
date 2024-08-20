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
- [ Repository Structure](#-repository-structure)
- [ Getting Started](#-getting-started)
  - [ Installation](#-installation)
  - [ Dataset preparation](#-dataset-preparation)
  - [ Usage](#-usage)
  - [ Config Details](#-config-details)
- [ Future work](#-future-work)
  - [ Pretrained Weight Preparation](#-pretrained-weight-preparation)
  - [ Tiny Percentage of Target Domain label Experiments](#-tiny-percentage-of-target-domain-label-experiments)
  - [ Self-training Strategy Implement Thoughts](#-self-training-strategy-implement-thoughts)
- [ Acknowledgments](#-acknowledgments)
</details>
<hr>

## [ Overview](#-overview)


#### Code release for the paper: [https://drive.google.com/file/d/19ZY15MAgTQxjjQTUtHEn1vZDbd18qiny/view?usp=sharing](https://drive.google.com/file/d/19ZY15MAgTQxjjQTUtHEn1vZDbd18qiny/view?usp=sharing)

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
    ├── experiment_saved_future_work
    │   ├── CITY2FOGGY_with_Dcls_channel_mixing_spatail_attention_TINY_GT_LABEL
    │   └── SIM2CITY_with_Dcls_channel_mixing_spatail_attention_TINY_GT_LABEL
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
        └── experiment_SIM2CITY.sh


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

Cilck here [https://drive.google.com/file/d/1ezCc0LeGXj_7uTVBLKknJWufHAQt0TGL/view?usp=sharing](https://drive.google.com/file/d/1ezCc0LeGXj_7uTVBLKknJWufHAQt0TGL/view?usp=sharing) to download and extract and merge files in experiment_saved.
It will looks like the structure below 
```sh
└── DA_stuff/
    ├── experiment_saved
        ├── [Experiment_name]
            ├── pretrained
                └── XXX.pth # saved pretrained weight
            ├── XXX.log # log
            ├── XXX.py # config
            └── XXX.pth # saved experiment weight
    ...

```

> ```console
> $ git clone https://github.com/Flame1045/DA_stuff.git
> ```

### [ Dataset preparation](#-dataset)
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

###  [ Config Details](#-details)

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


###  [ Usage](#-usage)

<h4>Evaluation & Visualization on <code>CITY2FOGGY</code></h4>
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
To visualize specific experiment, add below in script behind python3 tools/test.py ... --eval bbox
> ```
--show --show-score-thr 0.5 --show-dir your_output_dir
> ```
--------------

<h4>Evaluation on & Visualization<code>SIM2CITY</code></h4>
We provide six different experimental setups for evaluating the model on the SIM2CITY task. These setups are included in the script `tools/experiment_SIM2CITY.sh`.

The available experiments are:
- **SIM2CITY_baseline_and_pretrained_on_source**
- **SIM2CITY_oracle_and_trained_on_source_and_target**
- **SIM2CITY_with_Dcls_channel_mixing_spatial_attention**
- **SIM2CITY_with_Dcls**
- **SIM2CITY_with_Dcls_channel_mixing**
- **SIM2CITY_with_Dcls_spatial_attention**

To evaluate a specific experiment, uncomment (# python3 tools/test.py ...) the desired experiment in the script and run the following command:

> ```console
> $  bash tools/experiment_SIM2CITY.sh 
> ```
To visualize specific experiment, add below in script behind python3 tools/test.py ... --eval bbox
> ```
--show --show-score-thr 0.5 --show-dir your_output_dir
> ```
--------------

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
--------------

<h4>Training on <code>SIM2CITY</code></h4>
We provide six different experimental setups for training the model on the SIM2CITY task. These setups are included in the script `tools/experiment_SIM2CITY.sh`.

The available experiments are:
- **SIM2CITY_baseline_and_pretrained_on_source**
- **SIM2CITY_oracle_and_trained_on_source_and_target**
- **SIM2CITY_with_Dcls_channel_mixing_spatial_attention**
- **SIM2CITY_with_Dcls**
- **SIM2CITY_with_Dcls_channel_mixing**
- **SIM2CITY_with_Dcls_spatial_attention**

To train a specific experiment, uncomment (# CONFIG= ... to # done) the desired experiment in the script and run the following command:

> ```console
> $  bash tools/experiment_SIM2CITY.sh 
> ```

---

## [ Future Work](#-future)
We incorporate two additional experiments: 
1. A tiny percentage of target domain labels is used in our proposed method to evaluate whether the adapter requires labeled data for effective fine-tuning in domain adaptation. 
2. We explore a self-training strategy to assess whether the model can improve unsupervised domain adaptation with an enhanced training approach.

### [ Pretrained Weight Preparation](#-Pretrained)
Cilck here https://drive.google.com/file/d/1Wst2HzzjMkm4ryjZTI-GEl_RUDbo6FC9/view?usp=sharing to download and extract in experiment_saved_future_work.
It will looks like the structure below 
```sh
└── DA_stuff/
    ├── experiment_saved_future_work
        ├── [Experiment_name]
            ├── XXX.log # log
            ├── XXX.py # config
            └── XXX.pth # saved experiment weight
    ...

```

###  [ Tiny Percentage of Target Domain label Experiments](#-tiny-percentage-of-target-domain-label-experiments)

<h4>Evaluation on <code>CITY2FOGGY</code></h4>
We provide one experimental setup for evaluating the model on the CITY2FOGGY task. This is included in the script `tools/experiment_CITY2FOGGY.sh`.

The available experiments are:
- **CITY2FOGGY_with_Dcls_channel_mixing_spatail_attention_TINY_GT_LABEL**

To evaluate a specific experiment, uncomment (# python3 tools/test.py ...) in the desired experiment in the script and run the following command:

> ```console
> $  bash tools/experiment_CITY2FOGGY.sh 
> ```

#### Results of Cityscapes (Ds) → Foggy Cityscapes (Dt)
| Percent of target domain label  |  person | rider | car | truck | bus | train | motorcycle | bicycle | mAP |
|--------------|-------|---|---|----|----|-------|---|--------|----|
| 0 %  | 42.7 | 48.5 | 56.8 | 32.7 | 47.0 | 32.5 | 33.0 | 42.6 | 42.0 |
| 1 %  | 44.2  | 50.4  | 62.7  | 32.6  | 46.8  | 34.9  | 33.3  | 42.6  | 43.4  |

--------------

<h4>Evaluation on <code>SIM2CITY</code></h4>
We provide one experimental setup for evaluating the model on the CITY2FOGGY task. This is included in the script `tools/experiment_SIM2CITY.sh`.

The available experiments are:
- **SIM2CITY_with_Dcls_channel_mixing_spatail_attention_TINY_GT_LABEL**

To evaluate a specific experiment, uncomment (# python3 tools/test.py ...) in the desired experiment in the script and run the following command:

> ```console
> $  bash tools/experiment_SIM2CITY.sh 
> ```

#### Results of Sim10k (Ds) → Foggy Cityscapes (Dt)

| Percent of target domain label  |   AP         |
|---------------------|------------------------------------|
| 0 %  | 57.7 |
| 1 %  | 64.8|

--------------

<h4>Training on <code>CITY2FOGGY</code></h4>
We provide one experimental setup for training the model on the CITY2FOGGY task. This is included in the script `tools/experiment_CITY2FOGGY.sh`.

The available experiments are:
- **CITY2FOGGY_with_Dcls_channel_mixing_spatail_attention_TINY_GT_LABEL**

To train a specific experiment, uncomment (# CONFIG= ... to # done) in the desired experiment in the script and run the following command:

> ```console
> $  bash tools/experiment_CITY2FOGGY.sh 
> ```
--------------

<h4>Training on <code>SIM2CITY</code></h4>
We provide one experimental setup for training the model on the CITY2FOGGY task. This is included in the script `tools/experiment_SIM2CITY.sh`.

The available experiments are:
- **SIM2CITY_with_Dcls_channel_mixing_spatail_attention_TINY_GT_LABEL**

To train a specific experiment, uncomment (# CONFIG= ... to # done) the desired experiment in the script and run the following command:

> ```console
> $  bash tools/experiment_SIM2CITY.sh 
> ```


###  [ Self-training Strategy Implement Thoughts](#-self-training-strategy-implement-thoughts)

#### Guidelines for Converting This Code to Self-Training

##### Step 1: Understand the Current Codebase
- **Identify Key Components**:
  - Locate the discriminator and generator/feature extractor (found in `projects/models/da_head.py` at line 113).
  - Review how adversarial loss is computed and integrated (see `projects/models/da_head.py` at line 70).

- **Analyze Data Flow**:
  - Trace how source and target domain data are handled (refer to the config file for each experiment).

##### Step 2: Remove Adversarial Components
- **Discriminator Removal**:
  - Remove the discriminator network and related loss computations.

- **Feature Extractor Adjustment**:
  - Detach the feature extractor from any adversarial dependencies.

##### Step 3: Implement Pseudo-Labeling
- **Generate Pseudo-Labels**:
  - Mask target data according to the MIC paper (https://arxiv.org/abs/2212.01322).
  - Use the Student networks to predict target domain labels, including confidence scores.

- **Filter Pseudo-Labels**:
  - Apply a confidence threshold to filter pseudo-labels for training, as outlined in the MIC (https://arxiv.org/abs/2212.01322) paper.

- **Assign Pseudo-Labels**:
  - Store the filtered pseudo-labels for training purposes.

##### Step 4: Modify the Training Loop
- **Combine Data**:
  - Mix labeled source data with pseudo-labeled target data in the data loader.
  - Use the Teacher networks to generate predicted target labels.

- **Update Loss Function**:
  - Modify the loss function to calculate the loss between predicted target labels and pseudo-labels. This loss is used to update the Student networks.
  - Update the Teacher networks using EMA, as described in the MIC (https://arxiv.org/abs/2212.01322) paper.


---

##  [ Acknowledgments](#-acknowledgments)

- [SAPNetV2](https://github.com/Shuntw6096/SAPNetV2)
- Professor Wen-Hsien Fang and Professor Yie-Tarng Chen

[**Return**](#-overview)

---
