import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

learning_map_inv = {
    0: 10, 1: 11, 2: 15, 3: 18, 4: 20, 5: 30, 6: 31, 7: 32,
    8: 40, 9: 44, 10: 48, 11: 49, 12: 50, 13: 51, 14: 70, 15: 71,
    16: 72, 17: 80, 18: 81, 19: 0
}

color_map = {
    0: [0, 0, 0], 10: [100, 150, 245], 11: [100, 230, 245],
    15: [30, 60, 150], 18: [80, 30, 180], 20: [0, 0, 255],
    30: [255, 30, 30], 31: [255, 40, 200], 32: [150, 30, 90],
    40: [255, 0, 255], 44: [255, 150, 255], 48: [75, 0, 75],
    49: [175, 0, 75], 50: [255, 200, 0], 51: [255, 120, 50],
    70: [0, 175, 0], 71: [135, 60, 0], 72: [150, 240, 80],
    80: [255, 240, 150], 81: [255, 0, 0]
}

def load_and_map_label(label_path, is_prediction=False):
    raw = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
    raw_ids = raw & 0xFFFF
    if is_prediction:
        raw_ids = np.vectorize(lambda x: learning_map_inv.get(x, 0))(raw_ids)
    return raw_ids

def visualize_angled_bev(points, labels, save_path, elev=15, azim=200, zoom=1.8):
    colors = np.array([color_map.get(l, [0, 0, 0]) for l in labels]) / 255.0
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=0.5)

    ax.view_init(elev=elev, azim=azim)
    ax.axis('off')

    half_x, half_y, half_z = 40 / zoom, 20 / zoom, 3 / zoom
    ax.set_xlim(0, 2 * half_x)
    ax.set_ylim(-half_y, half_y)
    ax.set_zlim(-0.5, half_z)
    ax.set_box_aspect((2 * half_x, 2 * half_y, half_z + 0.5))
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--velodyne_dir', required=True, help='Directory of .bin files')
    parser.add_argument('--label_dir', required=True, help='Directory of .label files')
    parser.add_argument('--vis', choices=['gt', 'pred'], required=True, help='gt or pred')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--video_name', default='output.mp4', help='Output video filename')
    parser.add_argument('--fps', type=int, default=5, help='Frames per second for video')
    parser.add_argument('--elev', type=float, default=15, help='Elevation angle for 3D view')
    parser.add_argument('--azim', type=float, default=200, help='Azimuth angle for 3D view')
    parser.add_argument('--zoom', type=float, default=1.8, help='Zoom factor (larger means closer)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    bin_files = {f[:-4]: f for f in os.listdir(args.velodyne_dir) if f.endswith('.bin')}
    label_files = {f[:-6]: f for f in os.listdir(args.label_dir) if f.endswith('.label')}
    common_keys = sorted(set(bin_files.keys()) & set(label_files.keys()), key=lambda x: int(x))

    images = []
    for stem in common_keys:
        bin_path = os.path.join(args.velodyne_dir, bin_files[stem])
        label_path = os.path.join(args.label_dir, label_files[stem])
        out_path = os.path.join(args.out_dir, f'{stem}.png')

        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        labels = load_and_map_label(label_path, is_prediction=(args.vis == 'pred'))
        visualize_angled_bev(points, labels, out_path, elev=args.elev, azim=args.azim, zoom=args.zoom)
        images.append(imageio.imread(out_path))

    video_path = os.path.join(args.out_dir, args.video_name)
    imageio.mimsave(video_path, images, fps=args.fps)
    print(f"Saved video to {video_path}")

if __name__ == '__main__':
    main()