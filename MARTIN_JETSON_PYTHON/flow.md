# 📂 MARTIN_JETSON_PYTHON Structure
```bash
MARTIN_JETSON_PYTHON/
├── flow.md                   # Contains MARTIN_JETSON_PYTHON architecture and work flow
├── models/                   # Scripts to initialize computer vision models
│   ├── SYBIL.py     
├── nodes/                    # Scripts that are intended to function as ROS nodes
│   ├── SYBIL_node.py         
│   ├── SYBIL_node_w_camera.py
├── sensors/                  # Scripts that correctly import + wrap hardware + sensors
│   ├── RealSense.py   
├── utils/                    # Scripts that contain functions + logics for data manipulation
│   ├── depth_ops.py
│   ├── filtering.py        
```

# 📄 Code Execution Flow
This section will explain what each script does, what needs to be changed, and in which order they need to be run.

> [IMPORTANT] Before running any scripts, ensure your current working directory is set to the ```MARTIN_JETSON_PYTHON``` folder.

## Sensors
Scripts that wrap hardware sensors

### RealSense.py
1. Initialize and configure the RealSense pipeline:
* __init__(self, warmup_frames: int = 10)
* __get_frames__(self)
* __get_intrinsics__(self)
* __get_depth_scale__(self):
* __stop__(self)


## Models
Scripts to initialize Computer Vision models

### SYBIL.py
Creates a class to initialize instances of the SYBIL litter detection model and establishes the function used to evaluate frames for litter:
* __init__(self, weights_path: str)
* __infer__(self, frame: np.ndarray, conf_threshold: float = 0.531)

## utils
Scripts that contain functions + logics for data manipulation

### depth_ops.py
Contains the functions to transition a bounding box output from SYBIL to (X,Y,Z) coordinates that can be used for arm manipulation and approximate real-world bounding box size for future filtering:
* __depth_at_pixel__(depth_frame, x, y, depth_scale)
* __bbox_to_xyz_xywh__(bbox_xywh, depth_frame, intrinsics, depth_scale)
* __bbox_real_world_size__(bbox_xyxy, bbox_xywh, depth_frame, intrinsics, depth_scale)

### filtering.py
Not yet written

## nodes
Scripts that tie together different supporting scripts to achieve MARTIN's function

### SYBIL_node.py
Full execution of SYBIL with bounding boxes converted to real-world coordinates



### SYBIL_node_w_camera.py
A slight modification of SYBIL_node that displays the annotated frame on the computer screen


