import os
from pathlib import Path

# --- Dynamic Path Resolution ---
# This assumes you are running the script from within the SYBIL folder or a subfolder
# It finds the 'SYBIL' directory in your current path and sets it as the root.
def get_project_root():
    current_path = Path.cwd()
    for parent in [current_path] + list(current_path.parents):
        if parent.name == "SYBIL":
            return parent
    return current_path # Fallback to CWD if 'SYBIL' not in path parents

ROOT = get_project_root()
LABELS_DIR = ROOT / "labels"
SPLITS_DIR = ROOT / "splits"
IMAGES_DIR = ROOT / "images"

def fix_split_file_paths():
    """Rewrites the .txt files in /splits to match the current user's absolute path."""
    print(f"🛠️  Aligning split files to current root: {ROOT}")
    split_files = [f for f in os.listdir(SPLITS_DIR) if f.endswith('.txt')]
    
    for split_file in split_files:
        file_path = SPLITS_DIR / split_file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if not line.strip(): continue
            # Extract just the filename from the old path (handles / or \)
            filename = Path(line.strip()).name
            # Construct the new path based on the CURRENT user's SYBIL/images folder
            new_path = IMAGES_DIR / filename
            new_lines.append(str(new_path) + "\n")
            
        with open(file_path, 'w') as f:
            f.writelines(new_lines)
    print("✅ Split files successfully updated to current machine paths.\n")

def run_sanity_check():
    # 1. Count unique .txt label files
    if not LABELS_DIR.exists():
        print(f"❌ Error: Labels directory not found at {LABELS_DIR}")
        return

    all_label_files = [f for f in os.listdir(LABELS_DIR) if f.endswith('.txt')]
    total_unique_labels = len(all_label_files)
    
    print("--- General Stats ---")
    print(f"Project Root: {ROOT}")
    print(f"Total unique label files found: {total_unique_labels}")

    def get_filenames_from_split(file_name):
        path = SPLITS_DIR / file_name
        if not path.exists():
            return set()
        with open(path, 'r') as f:
            return {Path(line.strip()).stem for line in f if line.strip()}

    test_set = get_filenames_from_split("test.txt")
    print(f"Test set size: {len(test_set)}")
    print("----------------------\n")

    all_passed = True
    for i in range(1, 6):
        train_set = get_filenames_from_split(f"train_fold-{i}.txt")
        val_set = get_filenames_from_split(f"val_fold-{i}.txt")
        
        print(f"🔎 Checking Fold {i}...")
        current_fold_total = len(train_set) + len(val_set) + len(test_set)
        
        # Check A: Summation
        if current_fold_total == total_unique_labels:
            print(f"  ✅ Sum Match: {len(train_set)}T + {len(val_set)}V + {len(test_set)}Test = {total_unique_labels}")
        else:
            print(f"  ❌ Sum Mismatch: Total labels {total_unique_labels} vs split sum {current_fold_total}")
            all_passed = False

        # Check B: Uniqueness
        if not train_set.intersection(val_set) and not train_set.intersection(test_set):
            print(f"  ✅ Uniqueness: No overlaps.")
        else:
            print(f"  ❌ Data Leakage detected in Fold {i}!")
            all_passed = False
        print("")

    if all_passed:
        print("🎉 Sanity check passed!")

if __name__ == "__main__":
    # First, fix the paths so the .txt files work on THIS computer
    fix_split_file_paths()
    # Then, run the check
    run_sanity_check()