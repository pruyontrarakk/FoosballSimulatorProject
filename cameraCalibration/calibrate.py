#!/usr/bin/env python3
import cv2, numpy as np, glob, os, yaml
from pathlib import Path

# -------- CONFIG --------
IMG_GLOBS = [
    "calibration_data/*.JPG", "calibration_data/*.JPEG",
    "calibration_data/*.jpg", "calibration_data/*.jpeg",
    "calibration_data/*.png", "calibration_data/*.PNG"
]
PATTERN = (9, 6)         # inner corners: (cols, rows). Swap to (6,9) if needed.
SQUARE_SIZE = 25.0       # mm (use your real square size)
OUT_YAML = "calib.yaml"
OUT_NPZ  = "calib.npz"
PREVIEW  = "undistorted_preview.jpg"
# ------------------------

def find_images():
    imgs = []
    for g in IMG_GLOBS:
        imgs.extend(glob.glob(g))
    return sorted(imgs)

def detect_corners(gray):
    # Use the SB detector (far more robust)
    flags = cv2.CALIB_CB_EXHAUSTIVE
    res = cv2.findChessboardCornersSB(gray, PATTERN, flags=flags)
    if isinstance(res, tuple):
        ok, corners = res
    else:
        ok, corners = (res is not None), res
    if not ok or corners is None or len(corners) != PATTERN[0]*PATTERN[1]:
        return False, None
    # Optional subpixel refine
    cv2.cornerSubPix(
        gray, corners, (11,11), (-1,-1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    )
    return True, corners

def build_objp(pattern, square):
    objp = np.zeros((pattern[0]*pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * float(square)
    return objp

def main():
    image_paths = find_images()
    print(f"Found {len(image_paths)} images.")
    if not image_paths:
        return

    objp = build_objp(PATTERN, SQUARE_SIZE)
    objpoints, imgpoints = [], []
    image_size = None
    accepted = 0

    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] cannot read {p}")
            continue
        if image_size is None:
            h, w = img.shape[:2]
            image_size = (w, h)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # light preproc helps with glare
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        gray = cv2.equalizeHist(gray)

        ok, corners = detect_corners(gray)
        if not ok:
            print(f"Could not find corners in {p}")
            continue

        objpoints.append(objp.copy())
        imgpoints.append(corners)
        accepted += 1

        # quick visual check while iterating
        vis = img.copy()
        cv2.drawChessboardCorners(vis, PATTERN, corners, True)
        cv2.imshow("detected", vis)
        cv2.waitKey(30)

    cv2.destroyAllWindows()
    print(f"Accepted {accepted} images.")

    if accepted < 5:
        print("[ERROR] Need at least 5 good detections. Aborting.")
        return

    # ----- CALIBRATION -----
    print("\nRunning calibration...")
    # flags = cv2.CALIB_RATIONAL_MODEL  # enable for strong distortion lenses
    flags = 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCameraExtended(
        objpoints, imgpoints, image_size, None, None, flags=flags
    )[:5]  # ignore extra outputs from Extended

    # Per-image reprojection errors
    per_image_errs = []
    total_err, total_pts = 0.0, 0
    for i, obj in enumerate(objpoints):
        proj, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], K, dist)
        err = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        per_image_errs.append(float(err))
        total_err += err * len(proj)
        total_pts += len(proj)
    mean_err = float(total_err / total_pts)

    # ----- REPORT -----
    print("\nCalibration results")
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    print(f"RMS (OpenCV): {rms:.6f} px")
    print(f"Mean reprojection error: {mean_err:.6f} px")
    print("K (camera matrix):\n", K)
    print("dist (k1 k2 p1 p2 k3 ...):\n", dist.ravel())

    # ----- SAVE -----
    data = {
        "camera_matrix": K.tolist(),
        "dist_coeffs": dist.squeeze().tolist(),
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "rms": float(rms),
        "mean_reprojection_error": mean_err,
        "per_image_errors": per_image_errs,
        "pattern_inner_corners": {"cols": PATTERN[0], "rows": PATTERN[1]},
        "square_size_mm": float(SQUARE_SIZE),
    }
    with open(OUT_YAML, "w") as f:
        yaml.dump(data, f, sort_keys=False)
    np.savez(OUT_NPZ, K=K, dist=dist, rvecs=np.array(rvecs, dtype=object), tvecs=np.array(tvecs, dtype=object))
    print(f"\nSaved: {OUT_YAML} and {OUT_NPZ}")

    # ----- UNDISTORT PREVIEW -----
    first = cv2.imread(image_paths[0])
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, image_size, alpha=0.0, newImgSize=image_size)
    und = cv2.undistort(first, K, dist, None, newK)
    cv2.imwrite(PREVIEW, und)
    print(f"Undistorted preview -> {PREVIEW}")

if __name__ == "__main__":
    main()
