# Point Cloud Generation from segmented Range Image using Ouster SDK and ROS Packages
This repository demonstrates the process of generating a segmented ROI pointcloud from reflective and range images captured by the Ouster LiDAR sensor. The pipeline utilizes the Ouster SDK, OpenCV, Open3D and ROS packages to convert sensor data into meaningful 3D point clouds.

## YOLO-Pose Detection on LiDAR Reflective Image for Generating Segmentation Mask
The image below showcases YOLO-Pose detection applied on a LiDAR reflective image. The detected object's pose is accurately identified and the bounding box will be used as a mask.

![YOLO-Pose Detection](https://github.com/user-attachments/assets/d7fe6e54-3cfe-499e-a499-9b0540914fe8)

---

## LiDAR Point Cloud Data (Original)
The following image represents the raw point cloud data generated from the Ouster LiDAR sensor. This serves as the baseline for further processing and segmentation.

![Original LiDAR Point Cloud](https://github.com/user-attachments/assets/39cdf8b6-e985-4fb5-a80f-c662503ad038)

---

## Segmented Point Cloud Data
This image illustrates the segmented point cloud data of a detected object. The segmentation was achieved by mapping the reflective image to the corresponding range image. Utilizing a lookup table derived from the sensor's metadata I mapped the segmented range_image pixels to the 3d point cloud.

![Segmented Point Cloud]![segmented_pc](https://github.com/user-attachments/assets/e7785f50-7ddb-4b20-9015-536f1ce21406)


---

## Combined Point Cloud: Original and Segmented
The image below shows the combined point cloud, including the original and segmented point clouds. This visualization highlights the effectiveness of the segmentation process in isolating objects within the 3D space.

![Combined Point Cloud]![original_segmented_PC](https://github.com/user-attachments/assets/a68db6b8-6b53-429f-a5ec-2ef4ad023d0a)


---
## Conclusion
This project demonstrates 3D point cloud segmentation by utilizing object detection on its 2D images, which is less computationally intensive and provides fast and accurate results.
