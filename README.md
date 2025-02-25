# Multi-Scale Robotic Crack Measurement in Concrete Structures

[![Paper DOI](https://img.shields.io/badge/DOI-10.1016/j.autcon.2023.105215-blue)](https://doi.org/10.1016/j.autcon.2023.105215)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 Overview
This repository contains the source code and datasets for the **Multi-Scale Robotic Crack Measurement** approach introduced in the paper:

> **A Multi-Scale Robotic Approach for Precise Crack Measurement in Concrete Structures**  
> *Ali Ghadimzadeh Alamdari, Arvin Ebrahimkhanlou*  
> Published in *Automation in Construction (2024)*  
> [🔗 Read the Paper](https://doi.org/10.1016/j.autcon.2023.105215)

The repository provides:
- **ROS Package** for controlling a robotic arm for crack inspection.
- **CNN Model** for detecting and segmenting cracks using computer vision.
- **Experimental Data** including LiDAR scans, images, and point clouds.
- **Analysis Scripts** for evaluating performance and generating visualizations.

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/A-Multi-Scale-Robotic-Crack-Measurement.git
cd A-Multi-Scale-Robotic-Crack-Measurement
```

### 2️⃣ ROS Package Setup
The **ROS package** is built inside `catkin_ws`. Ensure you have **ROS (Melodic/Noetic)** installed.

```bash
cd ros_package
catkin_make
source devel/setup.bash
```

Run the robotic inspection:
```bash
roslaunch crack_inspection robot.launch
```

### 3️⃣ Running the CNN Model
The crack detection model is based on **U-Net**. To test the model:
```bash
cd cnn_model
python run_inference.py --image sample.jpg
```

To train the model:
```bash
python train.py --dataset dataset/
```

## 📂 Repository Structure
```yaml
├── ros_package/       # ROS package for robotic crack inspection
├── cnn_model/         # CNN model for crack detection
├── data/              # Experimental and simulation datasets
├── results/           # Analysis and output visualizations
├── docs/              # Documentation and supplementary files
├── README.md          # Main project documentation
└── LICENSE            # License information
```

## 📊 Results and Visualizations

### 📌 Crack Detection Output
<div align="center">
    <img src="results/crack_detection_example.png" alt="Crack Detection Example" width="500">
</div>

### 📌 LiDAR-Based 3D Reconstruction
<div align="center">
    <img src="results/point_cloud_example.png" alt="3D Reconstruction" width="500">
</div>

## 📜 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## 📧 Contact
For inquiries, feel free to contact:  
📩 **Ali Ghadimzadeh Alamdari** – [ag4328@drexel.edu](mailto:ag4328@drexel.edu)  
📩 **Arvin Ebrahimkhanlou** – [ae628@drexel.edu](mailto:ae628@drexel.edu)
