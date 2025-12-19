from ultralytics import YOLO

# 1. Load your trained model
# REPLACE THIS PATH with your actual trained model file (e.g., runs/train/yolov8n_custom/weights/best.pt)
model_path = "C:/Users/brand/Documents/College/2025/MARTIN/SYBIL/runs/final/yolov8m_best_full_retrain/weights/best.pt"
model = YOLO(model_path)

# 2. Export the model to ONNX format
# The 'format' argument specifies the output format (ONNX)
# The 'opset' should be a modern value like 12 or higher (17 is current standard for PyTorch exports)
model.export(format='onnx', opset=17) 

# The output file will be saved in the same directory as your model, e.g., 'yolov8n.onnx'
print("✅ YOLO Model successfully exported to ONNX format!")