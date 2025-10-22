# Reinforcement Learning for Foosball

## (previously existing readme)
Design and development of a professional foosball playing AI.
The directory is organized under 3 teams - Mechanical Assets, Simulation, and AI agents


## Setup Python environment
```
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```
.venv/bin/mjpython - <<'PY'
import time, mujoco
from mujoco.viewer import launch_passive
m = mujoco.MjModel.from_xml_path("foosball_sim/v2/foosball_sim.xml")
d = mujoco.MjData(m)
with launch_passive(m, d) as v:
    t0 = time.time()
    while v.is_running() and time.time() - t0 < 60:
        mujoco.mj_step(m, d)
        v.sync()
PY
```


## Camera Calibration

We calibrate the phone camera with a printed checkerboard using OpenCV. This produces:
- cameraCalibration/calib.yaml — intrinsics, distortion, and metrics (human-readable)
- cameraCalibration/calib.npz — NumPy bundle (intrinsic matric K, dist, mean reprojection error, etc.)
- cameraCalibration/undistorted_preview.jpg — visual check

**To run:**
```
cd cameraCalibration
python calibrate.py
```

**Our current results:**
```
Image size: 3024×4032
Mean reprojection error: 0.613 px
RMS (OpenCV): 4.717 px
K =
[[2787.81 0.00 1622.20]
[ 0.00 2823.81 2225.39]
[ 0.00 0.00 1.00]]
dist (k1 k2 p1 p2 k3) = [0.2875, -0.9437, 0.0116, 0.0049, 0.8478]
```

**What this means**
- **Mean reprojection error ~0.61 px** → good for general vision/pose tasks.
- **Intrinsics**: `fx≈2788 px`, `fy≈2824 px` (same order as image width/height), and the principal point `(cx≈1622, cy≈2225)` is near image center.
- **Distortion**: radial terms (k1,k2,k3) are significant (typical for phone lenses); tangential terms (p1,p2) are small, meaning model is reasonable.

**Notes**
- Intrinsics are valid **only for this resolution** (3024×4032). Recalibrate if you change resolution or device.


### Interpreting quality

- **Mean reprojection error:** ≤ **0.7 px** is good.
- **Visual check:** `undistorted_preview.jpg` should show straight table rails and checkerboard lines with no bending.

### Troubleshooting

- **“Could not find corners”**
  - Some provided images present this error even with the (9,6) included; this is likely due to human error. 
  - Ensure the entire inner-corner grid is visible (not occluded by rods/walls).
  - Reduce glare, keep the sheet flat, and use sharp images (no motion blur).



















.
