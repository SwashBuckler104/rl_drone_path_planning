# Requirements & Setup Guide
## RL Drone Path Planning — Module 6

---

## Target Platform
- **OS:** Ubuntu 22.04 LTS  (also tested on Windows 10/11)
- **Python:** 3.10 or 3.11
- **Hardware:** CPU only — no GPU required

---

## 1. System Packages

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip
# OpenCV needs these system libs
sudo apt install -y libgl1-mesa-glx libglib2.0-0
# Tkinter for live matplotlib windows (optional, used by render_mode="human")
sudo apt install -y python3-tk
```

### Windows 10 / 11

Ensure Python 3.10 or 3.11 is installed from https://python.org.  
During installation check **Add Python to PATH**.

No extra system packages needed — all dependencies are handled via pip.

---

## 2. Virtual Environment

Creating a virtual environment keeps this project isolated from your
system Python.  **Always do this before installing packages.**

### Ubuntu / macOS

```bash
# Navigate to the project directory
cd rl_drone_path_planning

# Create the venv
python3 -m venv venv

# Activate it
source venv/bin/activate

# Your prompt should now show (venv)
```

### Windows

```powershell
cd rl_drone_path_planning

python -m venv venv

# Activate
venv\Scripts\activate

# PowerShell may require: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## 3. Install Python Dependencies

With the venv activated:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on PyTorch:**  
> Stable-Baselines3 requires PyTorch.  The pip install above will
> automatically pull in the CPU version of PyTorch.  If you have an
> Nvidia GPU and want GPU acceleration, install PyTorch separately first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> ```

Expected install time: 2–5 minutes (mainly downloading PyTorch).

---

## 4. Package Version Summary

| Package            | Version Required | Purpose                          |
|--------------------|-----------------|----------------------------------|
| gymnasium          | ≥ 0.29.1        | RL environment API               |
| stable-baselines3  | ≥ 2.3.0         | PPO algorithm implementation     |
| numpy              | ≥ 1.24          | Array operations                 |
| opencv-python      | ≥ 4.8.0         | Map/image processing             |
| matplotlib         | ≥ 3.7.0         | Visualisation and plots          |
| Pillow             | ≥ 10.0          | GIF animation output             |
| tqdm               | ≥ 4.65          | Progress bars                    |
| torch              | (auto via SB3)  | Neural network backend for PPO   |

---

## 5. Generate Maps

Before running any training or testing script, generate the PGM map files:

```bash
# From inside rl_drone_path_planning/
python maps/generate_map.py
```

Expected output:
```
Generating maps...
  Saved: maps/training_map.pgm  (40×40)
  Saved: maps/simple_map.pgm    (20×20)

All maps generated and verified.
```

> The training scripts will auto-generate the map if you forget this step,
> but running it explicitly confirms the maps/ directory is writable.

---

## 6. Verify Installation

Run this quick smoke test to confirm everything is installed correctly:

```bash
python -c "
import gymnasium, stable_baselines3, numpy, cv2, matplotlib, PIL
print('gymnasium     :', gymnasium.__version__)
print('stable-baselines3 :', stable_baselines3.__version__)
print('numpy         :', numpy.__version__)
print('opencv        :', cv2.__version__)
print('matplotlib    :', matplotlib.__version__)
print('Pillow        :', PIL.__version__)
print()
print('All dependencies found — ready to train!')
"
```

Then verify the environment itself:

```bash
python envs/drone_env.py
```

Expected output (last line): `Smoke-test complete.`

---

## 7. Run the Full Project

```bash
# Step 1 — Generate maps (one-time setup)
python maps/generate_map.py

# Step 2 — Train the PPO agent  (~5–15 min on CPU)
python train_ppo.py

# Step 3 — Test with dynamic obstacle
python test_rl.py

# Step 4 — Compare A* and RL side-by-side
python compare_astar_rl.py
```

All output images are saved to `outputs/`.

---

## 8. Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError: gymnasium` | venv not activated | `source venv/bin/activate` |
| `libGL.so.1: cannot open` | OpenCV missing system lib | `sudo apt install libgl1-mesa-glx` |
| `No trained model found` | Ran test_rl before training | Run `python train_ppo.py` first |
| `AssertionError: Start position is on an obstacle` | Map auto-generation placed walls over start | Check that `(2,2)` is free in the generated map |
| `cannot connect to X server` (matplotlib) | Headless server, no display | Set `MPLBACKEND=Agg` or use `--no-obstacle` flag |

---

## 9. Deactivate the Virtual Environment

```bash
deactivate
```

---

## Hardware Notes

- **CPU:** This project runs entirely on CPU.  Training 300,000 steps
  takes roughly **5–15 minutes** on a modern quad-core CPU.
- **RAM:** ~500 MB peak during training.
- **Disk:** ~1 GB for PyTorch + SB3.
- **GPU:** Not required, but if you have one, SB3 uses it automatically
  when PyTorch is installed with CUDA support.
