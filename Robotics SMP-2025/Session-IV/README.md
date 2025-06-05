## Fire and Smoke Detection Model Training

This repository contains a Jupyter Notebook (`FireSmokeDetectionModelTraining.ipynb`) that demonstrates the process of downloading a fire and smoke detection dataset, training a YOLOv8 model, and performing inference.

### 1\. Dataset Download using Roboflow

The dataset for training the YOLO model is efficiently downloaded and prepared using the Roboflow Python SDK.

#### What this notebook does for dataset download:

  * Authenticates with your Roboflow account using an API key.
  * Fetches the **Fire Detection v3** project (version 6) from the `touatimed2` workspace.
  * Downloads the dataset in **YOLOv5 format**, which is also compatible with **Ultralytics YOLOv8**.

Roboflow simplifies dataset management, labeling, versioning, and formatting for many popular ML frameworks, including YOLO.

#### References:

  * **Roboflow Website**: [https://roboflow.com](https://roboflow.com)
  * **Roboflow Docs**: [https://docs.roboflow.com](https://docs.roboflow.com)

### 2\. Ultralytics YOLO - Overview

**Ultralytics YOLO** is an advanced and user-friendly object detection framework that supports multiple YOLO versions, with a strong emphasis on **YOLOv8**, its latest and most powerful iteration.

#### Official Resources:

| Resource               | Link                                                                             |
| :--------------------- | :------------------------------------------------------------------------------- |
| 🚀 Ultralytics Website | [ultralytics.com](https://ultralytics.com)                                       |
| 📚 YOLOv8 Docs         | [docs.ultralytics.com](https://docs.ultralytics.com)                             |
| 💻 GitHub Repo         | [github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| 📦 PyPI Package        | [pypi.org/project/ultralytics](https://pypi.org/project/ultralytics)             |

#### YOLOv8 Pre-trained Models:

The following pre-trained models are available:
| Model       | Size   | Speed    | Accuracy | Download Link                                                                             |
| :---------- | :----- | :------- | :------- | :---------------------------------------------------------------------------------------- |
| **YOLOv8n** | Nano   | Fastest  | Low      | [`yolov8n.pt`](https://www.google.com/search?q=%5Bhttps://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt%5D\(https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt\)) |
| **YOLOv8s** | Small  | Faster   | Medium   | [`yolov8s.pt`](https://www.google.com/search?q=%5Bhttps://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt%5D\(https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt\)) |
| **YOLOv8m** | Medium | Balanced | Higher   | [`yolov8m.pt`](https://www.google.com/search?q=%5Bhttps://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt%5D\(https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt\)) |
| **YOLOv8l** | Large  | Slower   | High     | [`yolov8l.pt`](https://www.google.com/search?q=%5Bhttps://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt%5D\(https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt\)) |
| **YOLOv8x** | XLarge | Slowest  | Highest  | [`yolov8x.pt`](https://www.google.com/search?q=%5Bhttps://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt%5D\(https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt\)) |

#### Supported Tasks:

| Task            | Command / Usage Example                                           |
| :-------------- | :---------------------------------------------------------------- |
| Detection       | `YOLO("yolov8n.pt").train(data="data.yaml")`                      |
| Segmentation    | `YOLO("yolov8n-seg.pt").train(data="data.yaml", task="segment")`  |
| Classification  | `YOLO("yolov8n-cls.pt").train(data="data.yaml", task="classify")` |
| Pose Estimation | `YOLO("yolov8n-pose.pt").train(data="data.yaml", task="pose")`    |

### 3\. Model Training

The notebook initializes a YOLOv8n model and trains it for fire and smoke detection. The model is trained on a dataset with 2 classes (fire and smoke).


### 4\. Inference

The notebook also contains code to perform inference using the trained model on video data.

### 5\. Technologies Used

The notebook utilizes the following Python libraries:

  * `roboflow`
  * `ultralytics`
  * `torch`
  * `os`
  * `pathlib`

### 6\. Setup and Usage

To run this Jupyter notebook:

1.  **Clone the repository** (if this notebook is part of a larger GitHub repository).
2.  **Install necessary libraries**:
    ```bash
    pip install roboflow ultralytics torch
    ```
    Ensure you have the correct PyTorch version that supports your CUDA (GPU) setup if you plan to use a GPU.
3.  **Obtain a Roboflow API key**: You will need a Roboflow API key to download the dataset. Replace it in the notebook with your actual API key.

5.  **Run cells sequentially** to download the dataset, load the model, train it, and perform inference.