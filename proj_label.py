#!/usr/bin/env python3
import cv2
import numpy as np
from pathlib import Path
import yaml

# ───────────────────────────────────────── Config ─────────────────────────────────────────

# Camera models: 0,2,3,4,5 = plumb_bob; 1,6,7 = fisheye
CAMERA_MODELS = {
    0: "plumb_bob", 1: "fisheye",
    2: "plumb_bob", 3: "plumb_bob",
    4: "plumb_bob", 5: "plumb_bob",
    6: "fisheye",   7: "fisheye",
}

# Folder names for your 8 cameras (same order as Ks / Ds / P0..P7 / Tr_0..Tr_7)
CAM_DIRS = [
    "port_a_cam_0",
    "port_a_cam_1",
    "port_b_cam_0",
    "port_b_cam_1",
    "port_c_cam_0",
    "port_c_cam_1",
    "port_d_cam_0",
    "port_d_cam_1",
]


ROOT = Path("/.../PanopticCUDAL")

# SemanticKITTI color map yaml
YAML_FP = Path("/../semantic-kitti-api/config/semantic-kitti.yaml")

# Which sequence + frame range to process
SEQ = "06"
CENTER_FRAME = 442        # will process CENTER_FRAME-3 .. CENTER_FRAME+2
FRAME_RADIUS = 3          # i.e. range(-FRAME_RADIUS, FRAME_RADIUS)

# Intrinsics (K) and distortion (D) for your 8 cameras
Ks = [
    np.array([[1827.48989, 0, 925.91346], [0, 1835.88358, 642.07154], [0, 0, 1]]),
    np.array([[968.79741, 0, 954.078232], [0, 974.917054, 664.433449], [0, 0, 1]]),
    np.array([[1845.22591, 0, 936.35196], [0, 1853.86544, 650.10661], [0, 0, 1]]),
    np.array([[1864.72501, 0, 970.12232], [0, 1873.55503, 650.17816], [0, 0, 1]]),
    np.array([[1868.79534, 0, 942.08755], [0, 1879.58294, 664.88899], [0, 0, 1]]),
    np.array([[1886.15844, 0, 998.18086], [0, 1897.20158, 647.67828], [0, 0, 1]]),
    np.array([[973.643281, 0, 955.433506], [0, 976.532409, 652.994446], [0, 0, 1]]),
    np.array([[968.010756, 0, 959.928453], [0, 976.214308, 653.013925], [0, 0, 1]]),
]

Ds = [
    np.array([-0.260735, 0.046071, 0.001173, -0.000154, 0.0]),
    np.array([-0.037439, -0.004983, 0.004292, -0.002258]),
    np.array([-0.288548, 0.104497, -0.001729, 0.00302, 0.0]),
    np.array([-0.286262, 0.102781, 0.002619, 0.001749, 0.0]),
    np.array([-0.293808, 0.114099, -0.003534, 0.001313, 0.0]),
    np.array([-0.294376, 0.125442, -0.002546, -0.000282, 0.0]),
    np.array([-0.036805, -0.007166, -0.005873, 0.006247]),
    np.array([-0.042778, -0.001466, 0.004273, -0.000596]),
]


# ───────────────────────────────────── Helper functions ───────────────────────────────────

def load_semkitti_color_map(yaml_path: Path):
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    cmap = data.get("color_map", {})
    # yaml should be {int_str: [B,G,R]} for OpenCV
    return {int(k): tuple(v) for k, v in cmap.items()}


def load_point_cloud(fp: Path):
    return np.fromfile(str(fp), dtype=np.float32).reshape(-1, 4)[:, :3]


def load_labels(fp: Path):
    return np.fromfile(str(fp), dtype=np.uint32)


def parse_calib_txt(fp: Path):
    txt = fp.read_text().strip().splitlines()
    data = {}
    for line in txt:
        line = line.strip()
        if not line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        vals = [float(x) for x in v.strip().split()]
        data[k] = vals

    Ps, Trs = [], []
    for i in range(8):
        P_key = f"P{i}"
        Tr_key = f"Tr_{i}"
        if P_key not in data or Tr_key not in data:
            raise KeyError(f"Missing {P_key} or {Tr_key} in calib file {fp}")
        P = np.array(data[P_key]).reshape(3, 4)
        Tr = np.eye(4)
        Tr[:3, :] = np.array(data[Tr_key]).reshape(3, 4)
        Ps.append(P)
        Trs.append(Tr)
    return Ps, Trs


def get_rectify_map(img, K, D, model: str):
    h, w = img.shape[:2]
    if model == "fisheye":
        newK = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=0.0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), newK, (w, h), cv2.CV_32FC1
        )
    else:
        newK, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 0)
        map1, map2 = cv2.initUndistortRectifyMap(
            K, D, np.eye(3), newK, (w, h), cv2.CV_32FC1
        )
    return map1, map2


def project_lidar(points, labels, P, Tr, img, color_map):
    h, w = img.shape[:2]

    # LiDAR -> homogeneous -> camera
    pts_hom = np.hstack([points, np.ones((points.shape[0], 1), dtype=points.dtype)])
    pts_cam = (Tr @ pts_hom.T).T[:, :3]

    # keep only points in front of camera
    mask = pts_cam[:, 2] > 0
    pts_cam = pts_cam[mask]
    labels = labels[mask]
    if pts_cam.shape[0] == 0:
        return img.copy()

    # pinhole projection using P (we only use K = P[:, :3])
    K = P[:, :3]
    uv = np.zeros((pts_cam.shape[0], 2), dtype=np.float32)
    uv[:, 0] = (pts_cam[:, 0] / pts_cam[:, 2]) * K[0, 0] + K[0, 2]
    uv[:, 1] = (pts_cam[:, 1] / pts_cam[:, 2]) * K[1, 1] + K[1, 2]
    uv = np.round(uv).astype(int)

    # keep points inside image
    valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv = uv[valid]
    labels = labels[valid]

    out = img.copy()
    for (x, y), l in zip(uv, labels):
        sem_label = l & 0xFFFF
        if sem_label == 0:
            continue  # skip unlabeled
        color = color_map.get(sem_label, (255, 255, 255))
        cv2.circle(out, (x, y), 2, color, -1)

    return out


# ────────────────────────────────────────── main ──────────────────────────────────────────

def main():
    print("[INFO] Starting LiDAR→camera projection")

    if not ROOT.exists():
        raise FileNotFoundError(f"ROOT does not exist: {ROOT}")

    if not YAML_FP.exists():
        raise FileNotFoundError(f"SemanticKITTI yaml not found: {YAML_FP}")

    color_map = load_semkitti_color_map(YAML_FP)
    print(f"[INFO] Loaded color map with {len(color_map)} entries")

    seq_dir = ROOT / "sequences" / SEQ
    if not seq_dir.exists():
        raise FileNotFoundError(f"Sequence dir not found: {seq_dir}")

    calib_fp = seq_dir / "calib.txt"
    if not calib_fp.exists():
        raise FileNotFoundError(f"calib.txt not found: {calib_fp}")

    Ps, Trs = parse_calib_txt(calib_fp)
    print("[INFO] Parsed calib.txt, got 8 P and 8 Tr matrices")

    out_root = ROOT / "projection" / SEQ
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output root: {out_root}")

    for offset in range(-FRAME_RADIUS, FRAME_RADIUS):
        frame_id = CENTER_FRAME + offset
        frame_str = f"{frame_id:06d}"
        print(f"\n[FRAME] {frame_str}")

        bin_fp = seq_dir / "velodyne" / f"{frame_str}.bin"
        label_fp = seq_dir / "labels" / f"{frame_str}.label"

        if not bin_fp.exists() or not label_fp.exists():
            print(f"  [SKIP] Missing velodyne or label for frame {frame_str}")
            continue

        points = load_point_cloud(bin_fp)
        labels = load_labels(label_fp)
        if points.shape[0] != labels.shape[0]:
            print(f"  [WARN] Mismatch points({points.shape[0]}) vs labels({labels.shape[0]}) - skipping")
            continue

        for cam_idx, cam_dir in enumerate(CAM_DIRS):
            img_fp = seq_dir / cam_dir / f"{frame_str}.png"
            if not img_fp.exists():
                print(f"  [MISS IMG] {img_fp}")
                continue

            img = cv2.imread(str(img_fp), cv2.IMREAD_COLOR)
            if img is None:
                print(f"  [WARN] Failed to read image: {img_fp}")
                continue

            model = CAMERA_MODELS[cam_idx]
            K = Ks[cam_idx]
            D = Ds[cam_idx]

            map1, map2 = get_rectify_map(img, K, D, model)
            rectified = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

            overlay = project_lidar(points, labels, Ps[cam_idx], Trs[cam_idx], rectified, color_map)

            save_dir = out_root / cam_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"labelproj_{frame_str}.png"

            cv2.imwrite(str(save_path), overlay)
            print(f"  [OK] cam {cam_idx} ({cam_dir}) → {save_path}")

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
