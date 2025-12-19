# 📂 SYBIL Structure
```bash
SYBIL/
├── flow.md                   # Contains SYBIL architecture and work flow
├── images/                   # Dataset images; downloaded from Drive
├── labels/                   # Dataset labels; downloaded from Drive
├── splits/                   # Defines fold splits
├── runs/                     # Training outputs
├── package_testing/          # Hardware test scripts
│   ├── checking_available_cameras.py
│   ├── GPU_test.py       
├── pre_testing/              # Scripts that need[ed] to be run before training
│   ├── COCO_YOLO_conversion.py
│   ├── verifying_k_fold.py
│   ├── yaml_path_corrector.py
├── post_training/            # Scripts that can be run given a trained model
│   ├── converting_pt_to_onnx.py
│   ├── model_eval.py   
│   ├── testing_SYBIL_IRL.py      
├── requirements.txt          # SYBIL Environment dependencies
└── Kfold_Optimizer_Code.py   # Core training script
```

# 📄 Code Execution Flow
This section will explain what each script does, what needs to be changed, and in which order they need to be run.

> [IMPORTANT] Before running any scripts, ensure your current working directory is set to the ```SYBIL``` folder. Open the ```SYBIL``` folder in your IDE before executing the code to ensure paths resolve correctly.

## 1.0 SYBIL Environment

### 1.1 Install all items
```bash
cd PATH/TO/MARTIN/SYBIL
conda create -n SYBIL_env python=3.10.12
conda activate SYBIL_env

pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118 --no-deps

pip install -r requirements.txt
```

### 1.3 Confirm ROS compatibility
Run ```conda list``` in your terminal. Confirm that:

```bash
ml_dtypes==0.5.3
numpy==2.2.6
onnx==1.19.1
opencv-python==4.12.0.88
protobuf==6.33.1
```

### 1.4 GPU_test.py
Run ```SYBIL/package_testing/GPU_test.py``` to ensure GPU is recognized.

## 2.0 Pre-Training
Before running any code, download the images and labels from the Google Drive:

https://drive.google.com/file/d/14YxYl2jsR-mImZsj_k05MjDsgw8qRoUj/view?usp=drive_link

Place the ```images``` and ```labels``` folders exactly as shown in the structure above.

### 2.1 COCO_YOLO_conversion.py
The PlastOPol dataset was downloaded in the COCO format. To adjust the format to the YOLO accepted structure: ```COCO_YOLO_conversion.py``` was used. If you get the dataset from the Google Drive, you do not need to run this code, as the Google Drive dataset is already YOLO formatted. It is included for reference and reproducibility. If you intend to run this code for whatever reason, you will need to correct the file paths to your specific computer.

### 2.2 verifying_k_fold.py
The new training method uses K-fold validation. To implement this in YOLO, the ```splits``` folder was created, which contains ```.txt``` files that specify the file paths to the images for each fold. The labels are assumed to be in a file path identical to the images but under a ```labels``` folder (instead of ```images```). Running ```verifying_k_fold.py``` will correct the ```splits/*.txt``` file paths for your specific computer automatically. You do not need to change any variables. This code will also ensure the folds are unique to prevent data leakage.


### 2.3 yaml_path_corrector.py
The ```yamls/*.yaml``` files are necessary to tell YOLO where to for the training, validation, and test data for each fold. Running ```yaml_path_corrector.py``` will correct the paths specified in those file paths for your specific computer automatically. You do not need to change any variables.

### 2.4 Computer Settings
Before running the training code, optimize your computer settings for best performance. Then, set your battery preferences so that your device never goes to sleep. I would also recommend plugging in your computer and setting the max charge to 55% to avoid straining your battery.

## 3.0 Training
The ```Kfold_Optimizer_Code.py``` file runs a Bayesian Optimizer on the first of 5 folds. Using those hyperparameters, it then trains 5 different models (one for each fold), then averages the results and records the average values in the ```K-Fold_Results.csv```, which will automatically be created in the ```SYBIL``` folder.

### 3.1 Change model size
You will need to change the line:

```modelsize = "yolov8n"   # change this when testing different model sizes```

Depending which model size you wish to train. The options are ```[n,s,m,l,x]```.

### 3.2 Optimizing Batch Size
YOLO automatically optimizes your batch size based on your hardware (largely your GPU VRAM). Run the training code with:

```YOLO_optimized_batch_size = 0.75  # Adjust based on GPU memory```

To calculate what batch size will use 75% of your VRAM.

If this line is left unchanged, YOLO will perform this calculation for every fold, which is inefficient, as this number will likely not change for a given model size. To speed up the training process, look in your terminal for where YOLO outputs the batch size. Will look something like:

```AutoBatch: Using batch-size 24 for CUDA:0 3.00G/4.00G (75%) ✅```

 Once the code outputs that value, stop the code, change the ```YOLO_optimized_batch_size```, and rerun the code.

### 3.3 Run the code
This will take a while (multiple hours). The code is set to ```verbose=True```, so you will get terminal output to ensure you are still making progress.

This code will automatically create several ```yolov8X.pt``` files in your ```SYBIL``` folder, which you can ignore. It will also record its training runs in a ```SYBIL/runs/``` file. The ```SYBIL/runs/BOKfolds/``` records the Bayesian Optimization trials, and the ```SYBIL/runs/Kfolds/``` records the fold models.

### 3.4 Recording Results
Once the training has finished, open ```K-Fold_Results.csv``` or look in the terminal for the average values and append them to the ```training_results.csv``` in the Google Drive.

If you are done training for a while, be sure to reset your computer settings for battery efficiency.

## 4.0 Post-Training
These files assume that a model has already been trained.

### 4.1 Using model with a webcam
The code files needed to debug and load up a model on your computer's camera.

#### 4.1.1 checking_available_cameras.py
Run ```SYBIL/package_testing/checking_available_cameras.py``` to scan your computer for a valid camera and to reveal the numbers assigned to specific cameras (if you have multiple). You do not need to change any variables.

#### 4.1.2 testing_SYBIL_IRL.py
Run ```testing_SYBIL_IRL.py``` to load a specified model onto your computer camera (or attached usb camera). The bounding boxes and classification confidence are included in frame.

### 4.2 Testing Generalization of Final Model
The ```model_eval.py``` file will train the final model on the full training set and evaluate it on the test set to ensure generalizability. It doesn't perform as advertised right now, which is by design.

> [IMPORTANT] This file should only be run AFTER a model has been selected based on the results of ```training_results.csv```. Running them before that point will result in peeking.

### 4.3 Converting to Onnx
After a final model has been selected, run ```converting_pt_to_onnx.py``` to convert the YOLO formatted model into something ROS can use. You will need to change the file path to the best model.