import os
from pathlib import Path

# --- Configuration ---

BASE_DIR = r"C:/Users/brand/Documents/College/2025/MARTIN/SYBIL/"
LABELS_DIR = BASE_DIR + r"labels"
SPLITS_DIR = BASE_DIR + r"splits"


def run_sanity_check():
    # 1. Count unique .txt label files
    all_label_files = [f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')]
    total_unique_labels = len(all_label_files)
    
    print("--- General Stats ---")
    print(f"Total unique label files found: {total_unique_labels}")

    # Helper to read paths from split files and return set of filenames
    def get_filenames_from_split(file_name):
        path = os.path.join(SPLITS_DIR, file_name)
        if not os.path.exists(path):
            return set()
        with open(path, 'r') as f:
            # We take the stem (filename without extension) to compare images to labels
            return {Path(line.strip()).stem for line in f if line.strip()}

    # Get the constant test set
    test_set = get_filenames_from_split("test.txt")
    print(f"Test set size: {len(test_set)}")
    print("----------------------\n")

    all_passed = True

    # 2. Check each fold (1 through 5)
    for i in range(1, 6):
        train_file = f"train_fold-{i}.txt"
        val_file = f"val_fold-{i}.txt"
        
        train_set = get_filenames_from_split(train_file)
        val_set = get_filenames_from_split(val_file)
        
        print(f"🔎 Checking Fold {i}...")
        
        # Check A: Summation
        current_fold_total = len(train_set) + len(val_set) + len(test_set)
        if current_fold_total == total_unique_labels:
            print(f"  ✅ Sum Match: {len(train_set)} (Train) + {len(val_set)} (Val) + {len(test_set)} (Test) = {total_unique_labels}")
        else:
            print(f"  ❌ Sum Mismatch: Expected {total_unique_labels}, got {current_fold_total}")
            all_passed = False

        # Check B: Uniqueness (No Overlap)
        train_val_overlap = train_set.intersection(val_set)
        train_test_overlap = train_set.intersection(test_set)
        val_test_overlap = val_set.intersection(test_set)
        
        if not train_val_overlap and not train_test_overlap and not val_test_overlap:
            print(f"  ✅ Uniqueness: No overlaps detected.")
        else:
            all_passed = False
            if train_val_overlap: print(f"  ❌ Data Leakage: Train & Val share {len(train_val_overlap)} images!")
            if train_test_overlap: print(f"  ❌ Data Leakage: Train & Test share {len(train_test_overlap)} images!")
            if val_test_overlap: print(f"  ❌ Data Leakage: Val & Test share {len(val_test_overlap)} images!")
        print("")

    if all_passed:
        print("🎉 All checks passed! Your dataset splits are clean and ready for K-Fold training.")
    else:
        print("⚠️ Sanity check failed. Please review the errors above before starting your training.")

if __name__ == "__main__":
    run_sanity_check()