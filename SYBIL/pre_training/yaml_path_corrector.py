import os
from pathlib import Path

# --- Dynamic Path Resolution ---
def get_project_root():
    current_path = Path.cwd()
    for parent in [current_path] + list(current_path.parents):
        if parent.name == "SYBIL":
            return parent
    return current_path

ROOT = get_project_root()
YAMLS_DIR = ROOT / "yamls"
SPLITS_DIR = ROOT / "splits"

def fix_yaml_paths():
    """Rewrites paths in .yaml files to match the current user's local directory."""
    if not YAMLS_DIR.exists():
        print(f"❌ Error: YAML directory not found at {YAMLS_DIR}")
        return

    yaml_files = [f for f in os.listdir(YAMLS_DIR) if f.endswith('.yaml')]
    print(f"📂 Found {len(yaml_files)} YAML files in {YAMLS_DIR}")

    for y_file in yaml_files:
        yaml_path = YAMLS_DIR / y_file
        
        with open(yaml_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            # Check for keys that typically hold paths
            if line.strip().startswith(('path:', 'train:', 'val:', 'test:')):
                key = line.split(':')[0]
                
                # Logic for the root 'path' key
                if key == 'path':
                    new_lines.append(f"path: {ROOT}\n")
                
                # Logic for specific split files
                else:
                    # Get the filename (e.g., train_fold-1.txt) regardless of old structure
                    filename = Path(line.split(':')[-1].strip()).name
                    # Construct the new local absolute path to the splits folder
                    new_path = SPLITS_DIR / filename
                    new_lines.append(f"{key}: {new_path}\n")
            else:
                # Keep nc, names, etc. as they are
                new_lines.append(line)
        
        with open(yaml_path, 'w') as f:
            f.writelines(new_lines)
            
    print(f"✅ All YAML files updated to root: {ROOT}")

if __name__ == "__main__":
    fix_yaml_paths()