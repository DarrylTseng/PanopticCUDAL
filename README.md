# Panoptic-CUDAL: Rural Australia Point Cloud Dataset in Rainy Conditions

<div align="center">
<a href="https://scholar.google.com/citations?user=IjXrc6AAAAAJ&hl=en">Tzu-Yun (Darryl) Tseng</a><sup>1</sup>, 
<a href="https://scholar.google.com/citations?user=xJW2v3cAAAAJ&hl=en">Alexey Nekrasov</a><sup>2</sup>, 
<a href="https://scholar.google.com/citations?hl=en&user=wEWN7RcAAAAJ">Malcolm Burdorf</a><sup>2</sup>,  
<a href="https://scholar.google.com/citations?user=ZcULDB0AAAAJ&hl=en">Bastian Leibe</a><sup>2</sup>,
<a href="https://scholar.google.com/citations?user=wT0QEpQAAAAJ&hl=en">Julie Stephany Berrio Perez</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=SElTcXQAAAAJ&hl=en">Mao Shan</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?hl=en&user=edmTNUQAAAAJ">Zhenxing Ming</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=LNqaebYAAAAJ&hl=en">Stewart Worrall</a><sup>1</sup>

<sup>1</sup>The University of Sydney <sup>2</sup>RWTH Aachen University

[Panoptic Autonomous Driving Dataset for Rural Rainy Environments](https://arxiv.org/abs/2503.16378v2). This paper is accepted to [IEEE ITSC 2025](https://ieee-itsc.org/2025/).


<p align="center">
  <img src="combined_8views_teaser.gif" alt="PanopticCUDAL teaser" width="800">
  <br>
  <em>PanopticCUDAL teaser: 8 synchronized views from rural Australian driving.</em>
</p>

</div>
<br><br>

## About

Existing autonomous driving datasets are pre- dominantly oriented towards well-structured urban settings and favourable weather conditions, leaving the complexities of rural environments and adverse weather conditions largely unaddressed. Although some datasets encompass variations in weather and lighting, bad weather scenarios do not appear often. Rainfall can significantly impair sensor functionality, introducing noise and reflections in LiDAR and camera data and reducing the system’s capabilities for reliable environmental perception and safe navigation. This paper introduces the Panoptic-CUDAL dataset, a novel dataset purpose-built for panoptic segmentation in rural areas subject to rain. By recording high-resolution LiDAR, camera, and pose data, Panoptic-CUDAL offers a diverse, information-rich dataset in a challenging scenario. We present the analysis of the recorded data and provide baseline results for panoptic, se- mantic segmentation, and 3D occupancy prediction methods on LiDAR point clouds.

## Data
- Images, `calib.txt`, metadata (`.pkl`) on [Hugging Face](https://huggingface.co/datasets/DarrylT/PanopticCUDAL/tree/main).
- LiDAR, labels, poses on [RWTH server](https://omnomnom.vision.rwth-aachen.de/data/panoptic-cudal/).

The dataset is organized in a directory structure in SemanticKITTI format:
```tree
├── 30/
│   ├── poses.txt
│   ├── calib.txt
│   ├── labels/
│   │     ├ 000000.label
│   │     └ 000001.label
|   ├── port_a_cam_0/
|   |     ├ 000000.png
|   |     ├ 000001.png
|   .
|   .
│   └── port_b_cam_1/
|   |     ├ 000000.png
|   |     ├ 000001.png
|   ├── port_d_cam_1/
|   |     ├ 000000.png
|   |     ├ 000001.png
|   |
│   └── velodyne/
│         ├ 000000.bin
│         └ 000001.bin
├── 31/
├── 32/
.
.
.
└── 41/
```
We closely follow the Semantic KITTI dataset structure and MMdetection3D to create .pkl files, and it should be possible to load it as is.

## Usage

To project point clouds on images, please run our code implemented on [SemanticKITTI API](https://github.com/PRBonn/semantic-kitti-api).
```bash
python proj_label.py
```

For inference, our code is based on [FRNet](https://github.com/Xiangxu-0103/FRNet).
```bash
python visualize.py --velodyne_dir /path/to/PanopticCUDAL/sequences/34/velodyne --label_dir /path/to/FRNet/predictions/34 --vis pred --out_dir /path/to/output/vis/04 --video_name pred_04.mp4 --fps 10
```

## Acknowledgement

We gratefully acknowledge the open-source communities whose tools, datasets and code underpinned this work.

## BibTeX
```
@article{tseng2025panoptic,
  title={Panoptic-CUDAL Technical Report: Rural Australia Point Cloud Dataset in Rainy Conditions},
  author={Tseng, Tzu-Yun and Nekrasov, Alexey and Burdorf, Malcolm and Leibe, Bastian and Berrio, Julie Stephany and Shan, Mao and Ming, Zhenxing and Worrall, Stewart},
  journal={IEEE International Conference on Intelligent Transportation Systems (ITSC)},
  url={https://arxiv.org/abs/2503.16378v2}
  year={2025}
}
```

