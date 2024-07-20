# CrackMamba: Topology-aware Mamba for Crack Segmentation in Structures

Official repository for: CrackMamab: Topology-aware Mamba for Crack Segmentation in Structures

## Notes
The full code will be made public after the paper is fully accepted.


## Installation

**Step-1:** Create a new conda environment & install requirements

```shell
conda create -n crack_mamba python=3.10
conda activate crack_mamba

pip install torch==2.0.1 torchvision==0.15.2
pip install causal-conv1d==1.1.1
pip install mamba-ssm
pip install torchinfo timm numba
```

**Step-2:** Install CrackMamba

```shell
git clone https://github.com/shengyu27/CrackMamba
cd CrackMamba
pip install -e .
```

## Prepare data & Pretrained model

**Dataset:**  

CrackSeg9K:

We use the same data & processing strategy following [SwinUMamba](https://github.com/JiarunLiu/Swin-UMamba). 
We downloaded version 2.0 of the dataset and re-screened it, see the file in the folder for the exact dataset split:

`/CrackMamba/dataset/train.txt`

`/CrackMamba/dataset/test.txt`

Download dataset from [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/EGIEBY) and put them into the data folder. Then preprocess the dataset with following command:

```shell
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

SewerCrack:

The dataset was derived from a series of original CCTV videos provided by a municipal authority in the southern United States. The process involved extracting frames from these videos and manually annotating them. Due to privacy concerns and copyright restrictions, we regret that we are unable to provide a public access link to this dataset.

CHASE_DB1:

For your convenience we have provided the data for this `dataset` in the dataset folder for you to download.

It's worth noting that the pixel values in mask are only 0 and 1, so they look all black when visualized. If any subsequent manipulation is required, you can handle it yourself.

**Pretrained model:** 

You can download the model weights `here(coming soon)`.

## Training & Evaluate

Using the following command to train & evaluate CrackMamba

```shell
bash scripts/train_Crack.sh
```

Here, You can configure the content of the script, such as folder address, dataset number, etc., to train on different datasets.

## Acknowledgements

We thank the authors of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [SwinUMamba](https://github.com/JiarunLiu/Swin-UMamba/tree/main), [Mamba](https://github.com/state-spaces/mamba) and [VMamba](https://github.com/MzeroMiko/VMamba) for making their valuable code & data publicly available.


