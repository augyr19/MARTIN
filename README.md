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

### Clone the repository:

```bash
git clone https://github.com/<your-username>/MARTIN.git
```

### Set up SYBIL environment:

See ```SYBIL/flow.md```


### Set up ROS environment:
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
│   └── flow.md           # Explains SYBIL architecture and workflow
├── ROS/                  # ROS subsystem
│   └── requirements.txt  # ROS Environment dependencies
|── README.md             # Project overview
|── .gitignore            # sets what needs to be ignored for Git
```