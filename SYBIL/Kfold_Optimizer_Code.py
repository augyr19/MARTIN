import optuna
from ultralytics import YOLO
import os
import psutil
import csv
from pathlib import Path



# Settings
YOLO_optimized_batch_size = 0.75  # Adjust based on GPU memory
modelsize = "yolov8n"   # change this when testing different model sizes
ROOT = Path.cwd() # Assumes they opened the SYBIL folder
base_path = ROOT / "yamls"
folds = [1, 2, 3, 4, 5]
csv_file = "K-Fold_Results.csv"

def get_yaml_path(fold_id):
    return os.path.join(base_path, f"fold{fold_id}.yaml")

# Optuna objective (only run on fold1)
def objective(trial, yaml_file, fold_id):
    lr = trial.suggest_loguniform("lr0", 1e-5, 1e-2)
    wd = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    model = YOLO(modelsize + ".pt")
    results = model.train(
        data=yaml_file,
        imgsz=640,
        epochs=15,
        batch=YOLO_optimized_batch_size,
        lr0=lr,
        weight_decay=wd,
        optimizer="AdamW",
        device=0,
        project="runs/BOKfolds",  # Bayesian Optimization K-Folds
        name=f"fold{fold_id}_trial{trial.number}",
        patience=10,
        conf=0.001,
        fraction=0.30,
        verbose=True
    )

    precision, recall, map50, map5095 = results.mean_results()
    return map50

def main():
    # Step 1: Bayesian optimization on Fold 1
    yaml_file_fold1 = get_yaml_path(1)
    print("\n🔁 Running Bayesian Optimization on Fold 1...")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, yaml_file_fold1, 1), n_trials=15)
    best_params = study.best_trial.params
    print(f"✅ Best params from Fold 1: {best_params}")

    # Step 2: Train all folds with best params
    outer_scores = []
    per_image_times = []

    for fold_id in folds:
        yaml_file = get_yaml_path(fold_id)
        print(f"\n🚀 Training Fold {fold_id} with best params...")

        model = YOLO(modelsize + ".pt")
        results = model.train(
            data=yaml_file,
            imgsz=640,
            epochs=100,
            batch=YOLO_optimized_batch_size,
            lr0=best_params["lr0"],
            weight_decay=best_params["weight_decay"],
            optimizer="AdamW",
            device=0,
            project="runs/Kfolds",
            name=f"{modelsize}_fold{fold_id}_best",
            patience=10,
            conf=0.001,
            fraction=1.0,
            verbose=True
        )

        precision, recall, map50, map5095 = results.mean_results()
        print(f"📊 Fold {fold_id} Results: mAP50={map50:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
        outer_scores.append(map50)

        # Extract per-image timings (YOLO reports these in results.speed)
        speed = results.speed
        per_image_time = speed['preprocess'] + speed['inference'] + speed['postprocess']
        per_image_times.append(per_image_time)

    # Step 3: Aggregate results
    mean_score = sum(outer_scores) / len(outer_scores)
    std_score = (sum((x - mean_score) ** 2 for x in outer_scores) / len(outer_scores)) ** 0.5
    avg_time_per_image = sum(per_image_times) / len(per_image_times)

    # Model size in MB
    model_size_mb = os.path.getsize(modelsize + ".pt") / (1024 * 1024)

    # Memory usage (approximate peak during run)
    memory_used_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

    print("\n==============================")
    print(f"✅ K-Fold Evaluation Complete")
    print(f"Average mAP50 across folds: {mean_score:.4f} ± {std_score:.4f}")
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Memory used: {memory_used_mb:.2f} MB")
    print(f"Avg per-image time: {avg_time_per_image:.4f} ms")
    print("==============================")

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model Size", "Mean mAP50", "Std mAP50", "Model Size (MB)", "Memory Used (MB)", "Avg Time per Image (ms)"])
        writer.writerow([modelsize, mean_score, std_score, model_size_mb, memory_used_mb, avg_time_per_image])

if __name__ == "__main__":
    main()
