# Pedestrian Detection

This pedestrian detection project utilizes OpenCV to detect and track pedestrians in a video stream. It employs a pre-trained Haar cascade classifier for pedestrian detection, dynamically adjusting parameters based on estimated pedestrian dimensions. Bounding boxes are drawn around detected pedestrians, and their count is displayed on each frame.

Key features include:

- Utilization of a pre-trained Haar cascade classifier for pedestrian detection.
- Dynamic adjustment of detection parameters based on estimated pedestrian dimensions.
- Visualization of bounding boxes around detected pedestrians, with count labels.
- Easy-to-use script for processing video files or live video streams.

## Dependencies

- OpenCV (`pip install opencv-python`)

## Usage

1. Clone the repository.
2. Ensure OpenCV is installed.
3. Run the script with a video file path as input (`python pedestrian_detection.py`).
4. Press 'q' to quit the application.

## Dataset

- The provided sample video file ("peoplebg.mp4") is sourced from [this YouTube video](https://youtu.be/ORrrKXGx2SE?si=jrLi9xuQdiMcoEwX), and used for demonstration purposes. However, the script can be easily adapted to work with other video files or live video streams.

**Note**: Fine-tuning of detection parameters may be required depending on the specific application and environmental conditions.
