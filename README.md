# Mobile Your Robot Learning Policy
# 机器人移动和导航策略学习


<a href='https://mobipi.github.io'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2505.23692'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

![The Policy Mobilization Problem](assets/github_teaser.gif)

## 1 Introduction

We introduce the "policy mobilization" problem: find a mobile robot base pose in a novel environment that is in distribution with respect to a manipulation policy trained on a limited set of camera viewpoints.

To study policy mobilization, we introduce the Mobi-π framework, which includes: (1) metrics that quantify the difficulty of mobilizing a given policy, (2) a suite of simulated mobile manipulation tasks based on RoboCasa to evaluate policy mobilization, (3) visualization tools for analysis, and (4) several baseline methods. We also propose a novel approach that bridges navigation and manipulation by optimizing the robot's base pose to align with an in-distribution base pose for a learned policy.

This repository includes the following contents:
* Pre-trained manipulation checkpoints for the 5 simulated benchmark tasks.
* Instructions for training new manipulation policies.
* Implementation of our proposed method and two baselines (BC w/ Nav and LeLaN).
* Instructions for training and evaluating our proposed method.
* Instructions for running baselines with provided checkpoints.
* Visualization tools.
* Plotting scripts used in the paper.


我们提出了“策略移动化”问题：在一种新颖的环境中，寻找一个移动机器人基座姿态，使其处于与仅基于有限摄像机视角集训练出的操作策略相匹配的分布中。

为了研究策略移动化问题，我们引入了 Mobi-π 框架，其中包括：（1）用于量化调动特定策略难度的指标；（2）基于 RoboCasa 构建的一系列模拟移动操作任务，用于评估策略移动化能力；（3）分析用可视化工具；以及（4）若干基线方法。我们还提出了一种新方法，通过优化机器人的基座姿态以与已学习策略的分布内基座姿态对齐，从而在导航和操作之间建立桥梁。

本代码仓库包含以下内容：
* 针对5个模拟基准任务预训练的操作模型检查点。
* 训练新操作策略的说明。
* 我们提出的方法及两种基线方法（BC w/ Nav 和 LeLaN）的实现。
* 训练和评估我们提出方法的说明。
* 使用提供检查点运行基线方法的说明。
* 可视化工具。
* 论文中使用的绘图脚本。

### 1.1 Installation

Install our repo with the following script:

```
conda create -c conda-forge -n mobipi python=3.10
conda activate mobipi
chmod +x install.sh
./install.sh
```
### 1.1.1 ./install.sh文件里面的链接需要手动下载

```
DOWNLOAD_ASSET_REGISTRY = dict(
    textures=dict(
        message="Downloading environment textures",
        url="https://utexas.box.com/shared/static/otdsyfjontk17jdp24bkhy2hgalofbh4.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/textures"),
        check_folder_exists=False,
    ),
    fixtures=dict(
        message="Downloading fixtures",
        url="https://utexas.box.com/shared/static/pobhbsjyacahg2mx8x4rm5fkz3wlmyzp.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/fixtures"),
        check_folder_exists=False,
    ),
    objaverse=dict(
        message="Downloading objaverse objects",
        url="https://utexas.box.com/shared/static/ejt1kc2v5vhae1rl4k5697i4xvpbjcox.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/objects/objaverse"),
        check_folder_exists=False,
    ),
    aigen_objs=dict(
        message="Downloading AI-generated objects",
        url="https://utexas.box.com/shared/static/os3hrui06lasnuvwqpmwn0wcrduh6jg3.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/objects/aigen_objs"),
        check_folder_exists=False,
    ),
    generative_textures=dict(
        message="Downloading AI-generated environment textures",
        url="https://utexas.box.com/shared/static/gf9nkadvfrowkb9lmkcx58jwt4d6c1g3.zip",
        folder=os.path.join(robocasa.__path__[0], "models/assets/generative_textures"),
        check_folder_exists=False,
    ),
)




```

If you wish to setup custom directories for your data and checkpoints, we recommend setting up macros. Run the following script: `python -m mobipi.scripts.setup_macros`. This should create a `mobipi/macros_private.py`. In this private macros file, edit the following constants:

```
SCENE_MODEL_ROOT_DIR = [insert your selected directory for 3D Gaussian Splatting models]
POLICY_CKPT_ROOT_DIR = [insert your selected directory for reading policy checkpoints]
LOG_ROOT_DIR = [insert your selected directory for evaluation logging]
DATA_ROOT_DIR = [insert your selected directory for policy training data]
```

<details>
<summary>Troubleshooting Tips</summary>

* If you get any pytorch-related errors, make sure (1) your `torch` version is `2.1.1` and supports your GPU; (2) your `numpy` version is `1.23.5`; (3) your `timm` version is `1.0.12`
* If you get any errors that look like the following: `'NoneType' object has no attribute 'CameraModelType'`, or if your script gets stuck forever at `gsplat: Setting up CUDA with MAX_JOBS=10`, try reinstalling the `gsplat` library following instructions in [this GitHub issue](https://github.com/nerfstudio-project/nerfstudio/issues/2685#issuecomment-1859660671).
</details>

### 1.2 Code Overview

Here is the outline of prominent components of the codebase:

* `external/`: third-party libraries such as `diffusion-policy` and `robocasa`.
* `mobipi/`:
    * `scene_model/`: scripts for collecting images for training a 3D Gaussian Splatting model and interfacing with a trained model.
    * `nav/`: scripts for generating LeLaN fine-tuning episodes and interfacing with a trained LeLaN checkpoint.
    * `eval/`: scripts for evaluating competing methods and retrieving evaluation result statistics.
    * `utils/`: various utility scripts, including code for computing score functions, dealing with I/O, processing media contents, and loading policy checkpoints.
    * `vis/`: stand-alone scripts for visualizing method performances and plotting result figures.
    * `scripts/`: utility script to setup macros.

## 2 Use Cases

### 2.1 Manipulation Policies

To run any method for policy mobilization, we need to prepare the manipulation policies to be mobilized. There are two ways to obtain them: (1) download our pre-trained checkpoint for the 5 benchmark tasks or (2) train a new checkpoint on your own.

**Downloading Existing Policy Checkpoints:** You can download pre-trained manipulation policies using the following script: `python mobipi/scripts/download_pi.py`. The script will walk you through selecting the models you wish to download.

<details>
<summary>Training Policies from Scratch</summary>

To train a policy in RoboCasa from scratch, you will need to first download a dataset generated by MimicGen, filter it so we use only the training scene split, and then train the policy.

To begin, navigate into `external/robocasa/robocasa/macros_private.py` and set the `DATASET_BASE_PATH` to the same directory as your specified dataset root directory in the mobipi macros file. Download the MimicGen dataset using the following script (you can switch the task name in the command line arguments):
```
python robocasa/scripts/download_datasets.py --ds_types mg_im --tasks CloseSingleDoor
```

Then, assuming that the data is downloaded in directory `~/robocasa`, run the following script to filter the dataset. Make sure you are using the robocasa installation in the git submodules (in `external`), not the original robocasa library. We use layouts 0, 2, 3, 5, 6 for training and 1, 4, 7, 8, 9 for testing. This split is selected by looking through all layouts and balancing the appearances of different room shapes between the training and test splits.
```
OMP_NUM_THREADS=8 MPI_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python robocasa/scripts/dataset_states_to_obs.py --dataset ~/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/mg/2024-05-04-22-34-56/demo_gentex_im128_randcams.hdf5 --filter_layouts 0,2,3,5,6 --n 300
```

Finally, set up policy learning with the following steps:
- Navigate to `external/robomimic/scripts`; run `python setup_macros.py`
- Navigate to `external/robomimic`; edit `macros_private.py` to configure wandb and experiment data directories
- Retrieve training commands by running `python robomimic/scripts/config_gen/gen_door.py --name CloseSingleDoor --n_seeds 3`. You can change the script name `gen_door` and command line arguments `--name` to switch from one environment to another.
- Then, simply execute the training commands printed out by the previous script execution to run policy training.
</details>

### 2.2 Scene Models

Our proposed method assumes the availability of 3D Gaussian Splatting models of the scene. There are two ways to obtain 3D Gaussian Splatting models: (1) download a 3D Gaussian Splatting model that we already trained for the benchmark; (2) train one of your own.

**Downloading Existing 3D Gaussian Splatting Checkpoints:** We train a Gaussian Splatting model for each task and scene. To download one or more models, run the following script: `python mobipi/scripts/download_scene_models.py`. The script will walk you through selecting the models you wish to download.

**Train 3D Gaussian Splatting Models from Scratch:** To obtain 3DGS models of scenes in the sim benchmark, navigate to `mobipi/scene_model` and run the following script. You can switch environments and scene IDs via command line arguments. The test-time scenes have IDs 1, 4, 7, 8, 9. This script will generate necessary data for training the 3DGS model and train the model. After the execution completes, find the generated `scene_data` directory and move contents inside it to your specified `SCENE_MODEL_ROOT_DIR`.
```
python collect_images_batch.py --env_names CloseDrawer --style_ids 1,4,7,8,9
```

### 2.3 Running the Given Manipulation Policies

To evaluate the performance of a given manipulation policy, make sure you have completed Section 2.1 of the readme. Then, navigate into `mobipi/eval/` and run the following script:

```
python eval_baseline.py --env_name CloseSingleDoor --layout_id -1 --seed 1 \
    --baseline_name vanilla_policy --randomize_base_init_pose 0.00
```

Here, you can switch out `CloseSingleDoor` to any other environment (aka. task) name. In our evaluation setup, we keep `layout_id` and `style_id` the same (`layout_id = 1, styld_id = 1`, `layout_id = 4, style_id = 4`, etc.). You can also specify `--layout_id -1` and drop the `--style_id` argument to test in all scenes in the evaluation split. If you wish to test a different layout and style combination, you can train scene models for that specific combination before evaluation.

The `--randomize_base_init_pose` option allows you to test policy performance at varying base pose offsets from 

### 2.4 Running Our Method

To run our method, make sure you have completed Sections 2.1 and 2.2 in this readme. Navigate into `mobipi/eval/` in this repository, and then run the following script:

```
python eval_mobipi.py --env_name CloseSingleDoor --scene_ids 1 --seed 1 --vis
```

In this command, `--env_name` sets the environment name. You can select among the following environments: `CloseSingleDoor`, `CloseDrawer`, `TurnOnMicrowave`, `TurnOnSinkFaucet`, `TurnOnStove`. The `--scene_id` argument sets the `(layout_id, style_id)` setup for evaluation (RoboCasa maintains a set of different room layouts and styles). In our experiment setup, we always keep layout and style IDs the same. You can set the scene ID among numbers 0 to 9. To test in all evaluation layouts, set `--scene_ids` to `-1`.

### 2.5 Running LeLaN

To run LeLaN, make sure you have completed Sections 2.1 and 2.2 in this readme. First, download the fine-tuned LeLaN checkpoint [here](https://download.cs.stanford.edu/juno/mobipi/lelan/lelan_finetuned.pth). Then, run the LeLaN baseline by navigating into `mobipi/eval/` and running the following script:

```
python eval_baseline.py --env_name CloseSingleDoor --layout_id 1 --style_id 1 --seed 1 \
    --baseline_name lelan --check_collisions --lelan_ckpt_path [insert checkpoint path]
```

As explained in Section 2.2, you can switch to different environments, layouts, and styles using the command line arguments. The script will run LeLaN for navigation for a maximum of 500 steps and then switch to running the manipulation policy. Results will be saved into the log directory.

### 2.6 Running BC w/ Nav

To run BC w/ Nav, make sure you have completed Section 2.1 of in this readme. First, download the policy checkpoints:

```
python mobipi/scripts/download_pi.py --nav
```

Then, run evaluation for this method by navigating into `mobipi/eval/` and running the following script:

```
python eval_baseline.py --env_name CloseSingleDoor --layout_id 1 --style_id 1 --seed 1 \
    --baseline_name il_nav --horizon 800
```

Similar to running the LeLaN baseline, you can switch to any other environment, layout, and style with the command line arguments. The script will run the BC w/ Nav baseline with an episode horizon of 800. This horizon value is set to be larger than the length of all training demos to ensure there is enough time for task completion.

### 2.7 Collecting Evaluation Results

To collect evaluation statistics, navigate into `mobipi/eval/` and run `python compute_success_rate.py [your log root directory]/[task name]/[method name]/bc_xfmr`. The script will compute and print out mean and stadard deviation of success rates for the specified environment and method.

### 2.8 Metrics

In the paper, we introduced metrics for learning about mobilization feasibility. Here, we show how to compute the spatial mobilization feasibility metric.

To compute the spatial metric, we first need to know the success rate of the given manipulation policy at different base pose deviations. To obtain this statistics, walk through Section 2.3 to run policy evaluation for your desired task across all evaluation scenes with `--randomize_base_init_pose` set to the following values: 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30. Then, walk through Section 2.7 to collect evaluation results for all these settings. Finally, navigate into `mobipi/vis/`, fill in [this dictionary](https://github.com/yjy0625/mobipi/blob/release/mobipi/vis/plot_spatial_metric.py#L8) with your collected mean success rate values and run the following script:

```
python mobipi/vis/plot_spatial_metric.py
```

### 2.9 Visualization Tools

We also provide visualization tools to inspect scenes and method performances.

* **Visualize Topdown Maps of A Scene**: Run `python mobipi/vis/plot_topdown.py`.
* **Visualize Navigation Targets Selected by Baselines and Our Method**: Edit [this line](https://github.com/yjy0625/mobipi/blob/release/mobipi/vis/plot_nav_poses.py#L139), [this line](https://github.com/yjy0625/mobipi/blob/release/mobipi/vis/plot_nav_poses.py#L140), and [this line](https://github.com/yjy0625/mobipi/blob/release/mobipi/vis/plot_nav_poses.py#L144) to adjust environment and method names. Then, use `mobipi/vis/plot_nav_poses.py` to produce a plot.
* **Visualize Animation of Optimization Process in Blender:** first, make sure to have completed at least one evaluation episode using our method. Then, navigate to `mobipi/vis/blender`, follow instructions at the top of the script to run `replay_episode.py`. This should create a subdirectory called `success` in your replay output directory. In this `success` subdirectory, you will see folders with names `replay_*/`. Note down the path to one of these folders and use it to run `render_episode.py`, following the instructions at the top of this file.

### 2.10 Plotting Results

Interested in producing a result plot similar to ours? Check out `mobipi/vis/plot_sim_table.py` and `mobipi/vis/plot_real_table.py`.

### Navgation:
##  下载robocasa导航演示数据集，包含了
```
https://public.boxcloud.com/d/1/b1!MAgh8V6WX8wT4BH1CjOm5X2B6fbFrJS5S67Dnkk64osI1dDKDymdt8rJ_XMjPXzoLFVHrNZVOnoSBHJkDuR5_6Bbl51Eqe-_c12IfTCBa6FpwPzvbKF6XVMaL7PdquJCJQ6SVU0aovfdlJ9EaJTqZOeFf-GKrnsUUsVeg9oKGDBeo1vkk4PWYGQjCRgWrEiv1_F-42NguMhk3LgIGctPfTN6Z_HwlK8CkFwDvJv0BYduU-s_wJAX0foP--dkJ0jESDMvbZ5pKmfKRMdau7PrSA-dyR8KDvRpKanndv48ppN26S_sBTw9DyTvcq4BPteOIwsQjt6iY79EVfMtf2N0EV7faTDFYztoUhX83PTby0VrusqyGveClv84tprYyasDtb_DFUMegaG2RyGav6OJcMoG9s0Ag6wqRuPtgbf7od2QXvZHKRIQJzkhKQyaGZKfLP33CtuueWEmD_LJNFAqb-yzy1ja8eVEBfoiHqOeV7FcW9McRWoDADGHkcWQ02HumMt31ebiv-z8wqsKA-g3dKxLDvluWejGWT77IboNDL0pcZNAHodAV9CFa-w3zLwXP0_z1ALqAhK6PbRcgOjIa3xMs5-UIaOO1q9338pt_AWp6z_v8bTJGKv6rgy301BYuKVS9TIdlLqPjxsbPijW9IUug5IaFQoNcw0S_lOwRd2X48nACM3cIGdbW039BtVOw6P6NmLP0Q04Ks76O4jLXqohisobtlPJNCMYZlTW8mUXDA1oLeyEXIG6-stiO8OwHbWdUEJjaFghataieYhZPn1ddqwaVJs29B-bDHtYHUH6IYcYkD1F7wAz_jHWbbsxLSa0zzDaTA7AzWPTrP8wOET384aattHPlOInyHRe8OkSr_fJp4z5pjHuf_FjOpBI16cAm4zgUdvJ9rml4VF1n00tq7goYfkg3stbbN9vyziizcLQDzUBF5x1falgNyetiZQbhJvpShd4RDv6JaZjyZogLraiErKGy6j0Xl7pVJ_GNJG6NBOb-E7Qu_UhrnYWGIRfvImAxJp3HoKItnPHd5jheU0ARknEy7rNpmJkLfHTulZOrHi2HZLVZ68byEJpjRvYTiaSvlnGQ-16XMKOcRfi81F6h9yRE6MpUy9f1_m7NSVBR60mvYrJW4HwZXvORNL71zOGrEzERqVWlgOEDMsrp22P3nm3zvXINyU7bV2tpWrx97_moUeLObivr0hwpNomHnf3oW1ydQaDkf4aqsYeAyASw6emF2CINf0lbv3lP3tZCy74b8ZclK5BahyWyUh5csRmpXxTnRDuHh79PuSVl1lbalYOkUGFulIfytDVdRqABGtdjHOl4DwqSMGWAEkuFtUP1WUTCsmRDGiqu_v1_eSixXHwezqrDPbzK1pnlY34uZtEWwXd9FPyLrcmmxd4Spo-7VfqvPbUQ33Zb7O9cLv7FyNmC4hdPdp420Sc-p8aur24o_V2vU-HJ5xYB06j3j30Zr09MOBes88T4Pw6AbIVV10W-X5W5VO_45JerAfs4Ntjr61XkFw-7yAGNaEUrNyuQXRFn2eOV_qL9Qd_jAEY/download
```

```
https://public.boxcloud.com/d/1/b1!Ff7uVsDY7JsPU9Ky90ztJajuv_GVXq_rXC1yhH2jev8QWrUqs1gcVnGugP5guP33EcALnWGgaKLkn7Cs34qs78qBljNPgsEN_SNVfmX7dTTXD3wxTAP07nPg_19-wrTnQeYSY5118_F4a0G949UA4u4bzLSsGw3LKWjUcCPc_W0TT0DVpjKzHHbbbGcBjvcNysIMP8rq2uJ04oH5wWrtheCNu4XWATxlGV2p6pp45tZIykkANIgMW5wGP0aKNirMo0JMyROhZPYbJUnqyJA8QsePpPF9hnepgpGCe_piDYPeioXWifXFieSuGLvi_3z2MaJCM8rpGHHmBQontZrsfJlszW8jo1d1woq4d6sIJtmK8kNuJuML_VHRn-j-7Qoxefnums9Y1FhPHv2WLWx_N1SYKDzpnLHuIJ4sduLnHR4CtDtLmT5WhexNdvQcQ4Z_RJ99ShbemVCXedAhNVCxUGEUd6X5XNnyfON1x7zf092wjZ9_H-crMqaM1-1LA0QP35nWtfl2jnbqqYE5dfQ2gk1iC-U2D07NYM5f_NtZbHjWLn7cxAJhFOpw6mk346j4bSuXolDbDAdcYotkBR4AUEJec990V4XIKjj7YIsWLgmH1pL_biiuXXvvd2S68jkUEx_6ra2-QmTU5bTrH2sSqZshGs_VTJdP7_Wp6iPP10YLn3fj3u_Ipr1HNvcTlb3kpi2ljMaE_wWqmZMYuygFi2nmJ3KU27BUpC_DzUTcyWYnYJoM-62qrWWx3NYteS4g4orpfyPzmmJYA4V6_XTZXVqYZxB3tnudqxZIIX81Vfv1xH-pE2syTRPAOER5aQyvlFPgn_FJ8wcSfyUn2v52v5FwurE6Q0hBCpMoDXVFAykqQVrJrD6A887F0vYTNBQSBeR8DJjgefP5PhfNF--iMJptQa9IWvvdDk08B9yYZFfV34v7SHMZlgyesmPu34F56zfoBmqZfrzz2ZM6a6hxBvdbqhVuyNAHHFEEe50RYVWYjL_PPoe68oC5u4X5kR4yUaySWrUcp2uxAIAvkX2RBHBjgo308gpLUhmAqIKx3WJCqiz0kxiyDoEEZJ2ndrt0HIxbdUgUWkw_nXg_rgpRKMuRcHRnIAbKVnwX_mHMgdWQ7KUdHhW6DMnla90olMTSl3JdVSsOuZjqPRXY4wA4vsnmYBcz9Ji7tg8fklpuoZ2jJAo-tTeFdkgPJZhrOtrnLLyNFgI6DzclGcENdpHgb8lufFWQOm682F3IIETKg6orIITWFnY_lf0vz0F4tgRClR_Fr7TVmgWCiGMgerd-MdL1W_FyDigMkwiMWeoDGGUoEnzRDcXelPznYQxJlmuPjFDA4k1ggVA3r6csi8dfg7gWizel6hQHzaLwlyGx0pZoIPwYGcQ1HY9E5ma7r2a_qF0AWh3XzPDvARMlZ3MYe2sAQAsprc6nnN_PNbmrjh9V8f47GIu3DLxFjpqowGb9YK9m0TJtPsETv1qz8ykUHZkAGgivWxLyaXymemMzVUDdibDRfeycdG-SX8x-/download
```

```
python robocasa/robocasa/scripts/playback_dataset.py --dataset /media/zk/Elements/sim/robocasa/demos/demo.hdf5
```


https://github.com/user-attachments/assets/872cd02d-6557-462b-a5b1-e8be55b1b11e

## 制定导航目标"ovens"
### 更改demo_task.py代码，添加导航任务，第101行
```
            ("PrepareCoffee", "make coffee"),
            ("NavigateKitchen","ovens")
```
### 执行程序，选择菜单第14个任务为导航任务
```
python /media/zk/Elements/sim/robocasa/robocasa/demos/demo_tasks.py
```


https://github.com/user-attachments/assets/d41efe0d-b6a8-448c-be4f-830ec517c7b5


| 类型                        | 数量         | 说明                                                                 |
|-----------------------------|--------------|----------------------------------------------------------------------|
| 文件格式                    | HDF5         | 包含机器人操作任务的演示数据                                         |
| 总体结构                    | 分为 `data` 和 `mask` 两部分 | `data` 存放具体演示数据，`mask` 存放数据划分信息             |
| Episodes（演示/轨迹）       | 至少 90 个   | 来自 `mask/90_demos`，每个 episode 是一次完整的任务执行过程         |
| Transitions（时间步/示例）  | 15,500 个    | 所有 episode 中的 `(obs, action)` 对数量，可用于行为克隆或模仿学习  |
| 单个 Episode 示例（如 demo_0） | 175 个时间步 | 动作数据 shape: `(175, 12)`，表示 175 个动作，每个动作 12 维       |



---

# 🧠 在 RoboTHOR 中生成带语言指令的数据集

本项目介绍如何使用 [AI2-THOR](https://github.com/allenai/ai2thor) 和 [RoboTHOR](https://github.com/allenai/robothon) 框架生成包含自然语言指令的任务数据集（如“请去卧室”、“打开冰箱”），可用于训练和评估基于视觉与语言的导航策略（Vision-and-Language Navigation, VLN）。

---

## 📦 1. 环境准备

### 安装依赖

确保你已经安装了以下库：

```bash
pip install ai2thor robo-thor
```

> ⚠️ 推荐使用虚拟环境（如 `conda` 或 `venv`）来隔离依赖。

---

## 🛠️ 2. 编写生成脚本

创建一个 Python 脚本（例如 `generate_language_dataset.py`），并添加如下内容作为基础模板：

```python
import ai2thor.controller
from ai2thor.platform import CloudRendering
import json

# 初始化控制器
controller = ai2thor.controller.Controller(
    platform=CloudRendering,
    scene="FloorPlan1",
)

# 开始录制
controller.start()

# 示例指令与对应动作映射
instructions_and_actions = [
    {"instruction": "Go to the bedroom.", "action": "go_to", "target": "bedroom"},
    {"instruction": "Open the fridge.", "action": "open_object", "target": "fridge"},
]

dataset = []

for item in instructions_and_actions:
    print(f"Instruction: {item['instruction']}")

    # 执行动作（此处为示例函数，需自行实现）
    if item["action"] == "go_to":
        success = move_to_room(controller, item["target"])
    elif item["action"] == "open_object":
        success = open_specific_object(controller, item["target"])

    # 保存一条记录
    dataset.append({
        "instruction": item["instruction"],
        "action_type": item["action"],
        "target": item["target"],
        "success": success
    })

# 停止录制
controller.stop()

# 保存为 JSON 文件
with open("language_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)
```

> 💡 注意：你需要自己定义 `move_to_room()` 和 `open_specific_object()` 函数，或者对接 AI2-THOR 提供的 API 来执行真实操作。

---

## 🧪 3. 示例数据结构

生成的 JSON 数据格式如下所示：

```json
[
  {
    "instruction": "Go to the bedroom.",
    "action_type": "go_to",
    "target": "bedroom",
    "success": true
  },
  {
    "instruction": "Open the fridge.",
    "action_type": "open_object",
    "target": "fridge",
    "success": true
  }
]
```

---

## 📊 4. 可选增强功能（建议扩展）

| 功能 | 描述 |
|------|------|
| 图像观测 | 添加 RGB 图像、深度图等图像数据 |
| 多语言支持 | 添加中文、西班牙语等多语言指令 |
| 场景多样性 | 遍历多个房间布局（`FloorPlan1` 到 `FloorPlan30`） |
| 路径记录 | 记录机器人移动轨迹（位置、方向） |
| 目标检测标签 | 添加目标对象的边界框或类别标签 |
| 强化学习接口 | 将数据集封装成 Gym 兼容接口 |

---

## 🚀 5. 运行命令

运行你的脚本以生成数据集：

```bash
python generate_language_dataset.py
```

生成的文件将保存为 `language_dataset.json`。

---

## ✅ 6. 成功后你可以做什么？

- 将数据集用于训练 VLN（Vision-and-Language Navigation）模型；
- 构建指令跟随机器人原型系统；
- 发布自己的数据集并用于论文或开源项目；
- 扩展到更多任务类型，如“拿取物品”、“打开抽屉”等。

---

## 📚 参考资料

- [AI2-THOR GitHub](https://github.com/allenai/ai2thor)
- [RoboTHOR Challenge](https://www.aicrowd.com/challenges/robothor-challenge-2022)
- [Allen AI Documentation](https://ai2thor.allenai.org/)

---




## 3 License

This codebase is licensed under the terms of the MIT License.

## 4 Acknowledgements

* The simulation code is based on [RoboCasa](https://robocasa.ai/), [RoboMimic](https://robomimic.github.io/), and [MimicGen](https://mimicgen.github.io/).
* Code from [LeLaN](https://learning-language-navigation.github.io/) is used in baseline implementations.
* Neural rendering implementation is based on [NerfStudio](https://docs.nerf.studio/).
