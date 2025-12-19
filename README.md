# MARTIN: Mobile Autonomous Remover of Trash In the eNvironment

MARTIN is a robotics project designed to autonomously detect and collect roadside litter.  
It integrates **SYBIL** (Single-class YOLO Based Identifier of Litter), a computer vision model trained to identify trash using single-class bounding boxes.

---

## 🚀 Features
- Autonomous litter detection and localization using YOLO-based computer vision.
- Modular design with SYBIL as the vision subsystem.
- Conda-based reproducible environment (`requirements.txt`).
- Organized dataset structure for training, validation, and testing.
- Experiment tracking and model outputs stored in `runs/`.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/MARTIN.git
```

Set up SYBIL environment:

```bash
cd PATH/TO/MARTIN/SYBIL
conda create -n SYBIL_env python=3.10.12
conda activate SYBIL_env
pip install -r requirements.txt
```
IMPORTANT: CUDA may need to be installed manually depending on GPU hardware:
```bash
pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118 --no-deps
```

After that step, please ensure that (necessary for smooth ROS exchange):
```bash
ml_dtypes==0.5.3
numpy==2.2.6
onnx==1.19.1
opencv-python==4.12.0.88
protobuf==6.33.1
```

Set up ROS environment:
```bash
cd PATH/TO/MARTIN/ROS
conda create -n ROS_env python=3.10.12
conda activate ROS_env
pip install -r requirements.txt
```

---

## 📂 Project Structure
```bash
MARTIN/
├── SYBIL/                # Vision subsystem
│   ├── images/           # Dataset images; downloaded from Drive
│   ├── labels/           # Dataset labels
│   ├── splits/           # Defines fold splits
│   ├── runs/             # Training outputs
│   ├── package_testing/  # Hardware/GPU test scripts
│   ├── pre_testing/      # Scripts that need[ed] to be run before training
│   ├── post_training/    # Scripts that can be run given a trained model
|   ├── requirements.txt  # SYBIL Environment dependencies
│   └── *.py              # Core training scripts
├── ROS/                  # ROS subsystem
│   └── requirements.txt  # ROS Environment dependencies
|── README.md             # Project overview
|── .gitignore            # sets what needs to be ignored for Git
```