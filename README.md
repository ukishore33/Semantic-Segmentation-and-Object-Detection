 # Object Detection and Semantic Segmentation Project

This project implements object detection and semantic segmentation on images and videos using Python, OpenCV, PyTorch, and pre-trained YOLO and DeepLab models.

## Repository Structure

- **functions.py:** This file contains all the helper functions necessary for object detection and segmentation tasks.

- **detect_objects_img.ipynb:** This Jupyter notebook detects objects in images, analyzes how performance decreases with noise, and performs semantic segmentation. In the last step both object detection and semantic segmentation are brought together.

- **detect_objects_mp4.ipynb:** This Jupyter notebook processes the file `video_1.mp4` in the folder `videos` and detects objects in it. The processed video is saved to `videos/video_1_preprocessed.mp4`.

There are two directories in the repository:

1) **images/original:** This directory contains images that are processed by the `detect_objects_img.ipynb` notebook.

2) **videos:** This directory contains a video that is processed by the `detect_objects_mp4.ipynb` notebook. The processed video is saved in the same folder with the name `video_1_preprocessed.mp4`.

Have fun!
