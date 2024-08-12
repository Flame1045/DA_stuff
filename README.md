<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" alt="project-logo">
</p>
<p align="center">
    <h1 align="center">DA_STUFF</h1>
</p>
<p align="center">
    <em><code>► INSERT-TEXT-HERE</code></em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/Flame1045/DA_stuff.git?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/Flame1045/DA_stuff.git?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Flame1045/DA_stuff.git?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Flame1045/DA_stuff.git?style=default&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [ Overview](#-overview)
- [ Features](#-features)
- [ Repository Structure](#-repository-structure)
- [ Modules](#-modules)
- [ Getting Started](#-getting-started)
  - [ Installation](#-installation)
  - [ Usage](#-usage)
  - [ Tests](#-tests)
- [ Project Roadmap](#-project-roadmap)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)
</details>
<hr>

##  Overview

<code>► INSERT-TEXT-HERE</code>

---

##  Features

<code>► INSERT-TEXT-HERE</code>

---

##  Repository Structure

```sh
└── DA_stuff/
    ├── README.md
    ├── environment.yml
    ├── envsetup.txt
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
    ├── requirements.txt
    └── tools
        ├── analysis_tools
        ├── count_each.py
        ├── dataset_converters
        ├── deployment
        ├── dist_test.sh
        ├── dist_train.sh
        ├── experiment_CITY2FOGGY.sh
        ├── experiment_SIMCITY.sh
        ├── experiment_old.sh
        ├── imgs2video.py
        ├── misc
        ├── model_converters
        ├── multi.sh
        ├── multi_2.sh
        ├── multi_3.sh
        ├── multi_C2F.sh
        ├── multi_C2F_2.sh
        ├── multi_C2F_3.sh
        ├── multi_C2F_4.sh
        ├── resume.sh
        ├── slurm_test.sh
        ├── slurm_train.sh
        ├── test.py
        ├── test.sh
        ├── train.py
        ├── train_Dcls.sh
        └── train_adapter.sh
```

---

##  Modules


<details closed><summary>experiment_saved.SIM2CITY_oracle_and_trained_on_source_and_target</summary>

| File                                                                                                                                                                                                                                             | Summary                         |
| ---                                                                                                                                                                                                                                              | ---                             |
| [custom_sim2city_unsupervised_base_wA_woCTBV2_B4_ORALCLE.py](https://github.com/Flame1045/DA_stuff.git/blob/master/experiment_saved/SIM2CITY_oracle_and_trained_on_source_and_target/custom_sim2city_unsupervised_base_wA_woCTBV2_B4_ORALCLE.py) | <code>► INSERT-TEXT-HERE</code> |

</details>

<details closed><summary>experiment_saved.SIM2CITY_with_Dcls_channel_mixing_spatail_attention</summary>

| File                                                                                                                                                                                                                                | Summary                         |
| ---                                                                                                                                                                                                                                 | ---                             |
| [custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff.git/blob/master/experiment_saved/SIM2CITY_with_Dcls_channel_mixing_spatail_attention/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py) | <code>► INSERT-TEXT-HERE</code> |

</details>

<details closed><summary>experiment_saved.CITY2FOGGY_with_Dcls_channel_mixing_spatail_attention</summary>

| File                                                                                                                                                                                                                                      | Summary                         |
| ---                                                                                                                                                                                                                                       | ---                             |
| [custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff.git/blob/master/experiment_saved/CITY2FOGGY_with_Dcls_channel_mixing_spatail_attention/custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py) | <code>► INSERT-TEXT-HERE</code> |

</details>

<details closed><summary>experiment_saved.CITY2FOGGY_with_Dcls_channel_mixing</summary>

| File                                                                                                                                                                                                                    | Summary                         |
| ---                                                                                                                                                                                                                     | ---                             |
| [custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff.git/blob/master/experiment_saved/CITY2FOGGY_with_Dcls_channel_mixing/custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py) | <code>► INSERT-TEXT-HERE</code> |

</details>

<details closed><summary>experiment_saved.CITY2FOGGY_with_Dcls_spatail_attention</summary>

| File                                                                                                                                                                                                                       | Summary                         |
| ---                                                                                                                                                                                                                        | ---                             |
| [custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff.git/blob/master/experiment_saved/CITY2FOGGY_with_Dcls_spatail_attention/custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py) | <code>► INSERT-TEXT-HERE</code> |

</details>

<details closed><summary>experiment_saved.CITY2FOGGY_oracle_and_trained_on_source_and_target</summary>

| File                                                                                                                                                                                                                                                   | Summary                         |
| ---                                                                                                                                                                                                                                                    | ---                             |
| [custom_city2foggy_unsupervised_base_wA_woCTBV2_B4_ORALCLE.py](https://github.com/Flame1045/DA_stuff.git/blob/master/experiment_saved/CITY2FOGGY_oracle_and_trained_on_source_and_target/custom_city2foggy_unsupervised_base_wA_woCTBV2_B4_ORALCLE.py) | <code>► INSERT-TEXT-HERE</code> |

</details>

<details closed><summary>experiment_saved.SIM2CITY_baseline_and_pretrained_on_source</summary>

| File                                                                                                                                                                 | Summary                         |
| ---                                                                                                                                                                  | ---                             |
| [custom_sim2city_base.py](https://github.com/Flame1045/DA_stuff.git/blob/master/experiment_saved/SIM2CITY_baseline_and_pretrained_on_source/custom_sim2city_base.py) | <code>► INSERT-TEXT-HERE</code> |

</details>

<details closed><summary>experiment_saved.SIM2CITY_with_Dcls_channel_mixing</summary>

| File                                                                                                                                                                                                              | Summary                         |
| ---                                                                                                                                                                                                               | ---                             |
| [custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff.git/blob/master/experiment_saved/SIM2CITY_with_Dcls_channel_mixing/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py) | <code>► INSERT-TEXT-HERE</code> |

</details>

<details closed><summary>experiment_saved.CITY2FOGGY_baseline_and_pretrained_on_source</summary>

| File                                                                                                                                                                       | Summary                         |
| ---                                                                                                                                                                        | ---                             |
| [custom_city2foggy_base.py](https://github.com/Flame1045/DA_stuff.git/blob/master/experiment_saved/CITY2FOGGY_baseline_and_pretrained_on_source/custom_city2foggy_base.py) | <code>► INSERT-TEXT-HERE</code> |

</details>

<details closed><summary>experiment_saved.SIM2CITY_with_Dcls_spatail_attention</summary>

| File                                                                                                                                                                                                                 | Summary                         |
| ---                                                                                                                                                                                                                  | ---                             |
| [custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff.git/blob/master/experiment_saved/SIM2CITY_with_Dcls_spatail_attention/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py) | <code>► INSERT-TEXT-HERE</code> |

</details>

<details closed><summary>experiment_saved.CITY2FOGGY_with_Dcls</summary>

| File                                                                                                                                                                                                     | Summary                         |
| ---                                                                                                                                                                                                      | ---                             |
| [custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff.git/blob/master/experiment_saved/CITY2FOGGY_with_Dcls/custom_city2foggy_unsupervised_base_wA_woCTBV2_B4.py) | <code>► INSERT-TEXT-HERE</code> |

</details>

<details closed><summary>experiment_saved.SIM2CITY_with_Dcls</summary>

| File                                                                                                                                                                                               | Summary                         |
| ---                                                                                                                                                                                                | ---                             |
| [custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py](https://github.com/Flame1045/DA_stuff.git/blob/master/experiment_saved/SIM2CITY_with_Dcls/custom_sim2city_unsupervised_base_wA_woCTBV2_B4.py) | <code>► INSERT-TEXT-HERE</code> |

</details>

<details closed><summary>projects.models</summary>

| File                                                                                                                           | Summary                         |
| ---                                                                                                                            | ---                             |
| [co_deformable_detr_head.py](https://github.com/Flame1045/DA_stuff.git/blob/master/projects/models/co_deformable_detr_head.py) | <code>► INSERT-TEXT-HERE</code> |
| [co_dino_head.py](https://github.com/Flame1045/DA_stuff.git/blob/master/projects/models/co_dino_head.py)                       | <code>► INSERT-TEXT-HERE</code> |
| [co_detr.py](https://github.com/Flame1045/DA_stuff.git/blob/master/projects/models/co_detr.py)                                 | <code>► INSERT-TEXT-HERE</code> |
| [test.py](https://github.com/Flame1045/DA_stuff.git/blob/master/projects/models/test.py)                                       | <code>► INSERT-TEXT-HERE</code> |
| [_transformer.py](https://github.com/Flame1045/DA_stuff.git/blob/master/projects/models/_transformer.py)                       | <code>► INSERT-TEXT-HERE</code> |
| [query_denoising.py](https://github.com/Flame1045/DA_stuff.git/blob/master/projects/models/query_denoising.py)                 | <code>► INSERT-TEXT-HERE</code> |
| [distillation.py](https://github.com/Flame1045/DA_stuff.git/blob/master/projects/models/distillation.py)                       | <code>► INSERT-TEXT-HERE</code> |
| [slide_attention.py](https://github.com/Flame1045/DA_stuff.git/blob/master/projects/models/slide_attention.py)                 | <code>► INSERT-TEXT-HERE</code> |
| [co_atss_head.py](https://github.com/Flame1045/DA_stuff.git/blob/master/projects/models/co_atss_head.py)                       | <code>► INSERT-TEXT-HERE</code> |
| [sapnet.py](https://github.com/Flame1045/DA_stuff.git/blob/master/projects/models/sapnet.py)                                   | <code>► INSERT-TEXT-HERE</code> |
| [da_head.py](https://github.com/Flame1045/DA_stuff.git/blob/master/projects/models/da_head.py)                                 | <code>► INSERT-TEXT-HERE</code> |
| [co_roi_head.py](https://github.com/Flame1045/DA_stuff.git/blob/master/projects/models/co_roi_head.py)                         | <code>► INSERT-TEXT-HERE</code> |
| [transformer.py](https://github.com/Flame1045/DA_stuff.git/blob/master/projects/models/transformer.py)                         | <code>► INSERT-TEXT-HERE</code> |
| [swin_transformer.py](https://github.com/Flame1045/DA_stuff.git/blob/master/projects/models/swin_transformer.py)               | <code>► INSERT-TEXT-HERE</code> |

</details>

##  Getting Started

**System Requirements:**

* **Python**: `version x.y.z`

###  Installation

<h4>From <code>source</code></h4>

> 1. Clone the DA_stuff repository:
>
> ```console
> $ git clone https://github.com/Flame1045/DA_stuff.git
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd DA_stuff

> ```

> 3. Install the dependencies:
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
> ```

###  Usage

<h4>From <code>source</code></h4>

> Run DA_stuff using the command below:
> ```console
> $ python main.py
> ```

###  Tests

> Run the test suite using the command below:
> ```console
> $ pytest
> ```

---

##  Project Roadmap

- [X] `► INSERT-TASK-1`
- [ ] `► INSERT-TASK-2`
- [ ] `► ...`

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://github.com/Flame1045/DA_stuff.git/issues)**: Submit bugs found or log feature requests for the `DA_stuff` project.
- **[Submit Pull Requests](https://github.com/Flame1045/DA_stuff.git/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/Flame1045/DA_stuff.git/discussions)**: Share your insights, provide feedback, or ask questions.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/Flame1045/DA_stuff.git
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to github**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="center">
   <a href="https://github.com{/Flame1045/DA_stuff.git/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=Flame1045/DA_stuff.git">
   </a>
</p>
</details>

---

##  License

This project is protected under the [SELECT-A-LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

##  Acknowledgments

- List any resources, contributors, inspiration, etc. here.

[**Return**](#-overview)

---
