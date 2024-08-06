# SCRFD Face Detection

This repository contains an implementation of face detection using the SCRFD model, which is a fast and lightweight face detection model designed for edge devices. The implementation utilizes the ONNX format for the model and leverages OpenCV for image and video processing.

## Features

- **Efficient Face Detection**: Detects faces with high accuracy and speed.
- **Support for Keypoints**: Detects facial landmarks for better face alignment.
- **Image and Video Processing**: Supports detection on images, video files, and live camera feeds.
- **ONNX Model**: Utilizes the ONNX format for broad compatibility and deployment options.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- ONNXRuntime

### Install Dependencies

You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```
## Usage
You can use this script to detect faces in images, videos, or live camera feeds. Below are the instructions on how to use it:

1. Image Detection
To run face detection on a single image:

```bash
python crop.py --image path_to_image.jpg --output output_image.jpg
```
2. Video Detection
To run face detection on a video file:

```bash
python crop.py --video path_to_video.mp4 --output output_video.mp4
```
3. Live Camera Detection
To run face detection using your webcam:

```bash
python crop.py --camera 0 --output output_video.mp4
```
Note: Replace 0 with the appropriate camera index if you have multiple cameras.

Command-Line Arguments
- --image: Path to the image file.
- --video: Path to the video file.
- --camera: Camera index for capturing video.
- --output: Path to save the output image or video.

Sure! Here is the README content in Markdown format:

markdown
Copy code
# SCRFD Face Detection

This repository contains an implementation of face detection using the SCRFD model, which is a fast and lightweight face detection model designed for edge devices. The implementation utilizes the ONNX format for the model and leverages OpenCV for image and video processing.

## Features

- **Efficient Face Detection**: Detects faces with high accuracy and speed.
- **Support for Keypoints**: Detects facial landmarks for better face alignment.
- **Image and Video Processing**: Supports detection on images, video files, and live camera feeds.
- **ONNX Model**: Utilizes the ONNX format for broad compatibility and deployment options.

## Requirements

- Python 3.x
- OpenCV
- NumPy
- ONNXRuntime

### Install Dependencies

You can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```
Example requirements.txt:

numpy
opencv-python
onnxruntime
# Usage
You can use this script to detect faces in images, videos, or live camera feeds. Below are the instructions on how to use it:

1. Image Detection
To run face detection on a single image:

```bash
python script_name.py --image path_to_image.jpg --output output_image.jpg
```
2. Video Detection
To run face detection on a video file:

```bash
python script_name.py --video path_to_video.mp4 --output output_video.mp4
```
3. Live Camera Detection
To run face detection using your webcam:

```bash
python script_name.py --camera 0 --output output_video.mp4
```
Note: Replace 0 with the appropriate camera index if you have multiple cameras.

Command-Line Arguments
--image: Path to the image file.
--video: Path to the video file.
--camera: Camera index for capturing video.
--output: Path to save the output image or video.
# Output
The script will output the processed images or videos with detected faces and keypoints drawn. The results can be saved to disk or displayed in a window.

# Acknowledgments
[InsightFace](https://github.com/deepinsight/insightface) for the SCRFD model.

[ONNXRuntime](https://github.com/microsoft/onnxruntime) for providing the inference engine.
