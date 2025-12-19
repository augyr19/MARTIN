import os
import csv
from pathlib import Path
from ultralytics import YOLO

def evaluate_models():
    # Base paths
    base_dir = Path(r"C:\Users\brand\Documents\College\2025\Capstone\YOLO_PlastOpol_DataSet")
    final_dir = base_dir / "runs" / "final"
    yaml_path = base_dir / "plastopol.yaml"

    # Model sizes to evaluate - only medium in this case
    # other options to add are: "n", "s", "l", "x",
    model_sizes = ["m"]

    # Output CSV
    csv_path = final_dir / "test_results.csv" # Change if working with validation set

    # Prepare CSV header
    header = [
        "model",
        "model_size_MB",
        "num_parameters",
        "mAP50",
        "mAP50-95",
        "preprocess_ms",
        "inference_ms",
        "loss_ms",
        "postprocess_ms"
    ]

    results = []

    for size in model_sizes:
        model_name = f"yolov8{size}_best_full_retrain"
        weights_path = final_dir / model_name / "weights" / "best.pt"

        if not weights_path.exists():
            print(f"Skipping {model_name}, weights not found.")
            continue

        print(f"Evaluating {model_name}...")

        # Load model
        model = YOLO(str(weights_path))

        # Collect model info
        num_params = sum(p.numel() for p in model.model.parameters())
        model_size_MB = os.path.getsize(weights_path) / (1024**2)

        # Evaluate on test set
        metrics = model.val(
            data=str(yaml_path),
            split="test", # or "val" if needed on validation set
            conf=0.001,
            batch=4, # Adjust batch size as needed
            device=0, # Use GPU 0
            verbose=True,
        )

        # Extract metrics
        mAP50_95 = metrics.box.map
        mAP50 = metrics.box.map50

        # Speed breakdown (ms per image)
        preprocess_ms = metrics.speed['preprocess']
        inference_ms = metrics.speed['inference']
        loss_ms = metrics.speed['loss']
        postprocess_ms = metrics.speed['postprocess']

        results.append([
            model_name,
            round(model_size_MB, 3),
            num_params,
            round(mAP50_95, 4),
            round(mAP50, 4),
            round(preprocess_ms, 2),
            round(inference_ms, 2),
            round(loss_ms, 2),
            round(postprocess_ms, 2)
        ])

    # Write results to CSV
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results)

    print(f"Evaluation complete. Results saved to {csv_path}")


if __name__ == "__main__":
    evaluate_models()
