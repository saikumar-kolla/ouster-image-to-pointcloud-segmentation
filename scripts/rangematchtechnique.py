"""Description: This script uses the logic where the ROI_range image values are used to match the original point cloud values 
 and then publish the matched points as PointCloud2 message."""


import os
import rospy
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointField
from std_msgs.msg import Header
from cv_bridge import CvBridge
from ouster.sdk import client
import std_msgs.msg
import json
import numpy as np
import pandas as pd
import cv2
from ultralytics import YOLO
import time
import open3d as o3d

class HandSegmentation:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = YOLO('/home/kolla/Downloads/yolo11x-pose.pt')

        # Subscribe to image, range image, and point cloud topics from rosbag
        self.image_sub = rospy.Subscriber("/ouster/reflec_image", Image, self.image_callback)
        self.pcl_sub = rospy.Subscriber("/ouster/points", PointCloud2, self.store_pcl)
        self.range_image_sub = rospy.Subscriber("/ouster/range_image", Image, self.range_image_callback)
        self.metadata_sub = rospy.Subscriber("/ouster/metadata", std_msgs.msg.String, self.metadata_callback)

        # Publisher for segmented hand point cloud
        self.chest_pub = rospy.Publisher("/filtered_hand_points", PointCloud2, queue_size=10)

        # Store latest point cloud and range image
        self.latest_pcl = None
        self.latest_range_image = None

        # Store sensor metadata
        self.sensor_info = None
        self.metadata = None
        self.xyzlut = None  # XYZ lookup table


        # Timeout parameters for metadata
        self.metadata_received = False
        self.metadata_timeout = 30
        self.metadata_start_time = time.time()


        # Counter for logging frequency
        self.log_counter = 0

    def metadata_callback(self, msg):
        """Process metadata from /ouster/metadata"""
        try:
            self.metadata = json.loads(msg.data)
            metadata_str = json.dumps(self.metadata)
            self.sensor_info = client.SensorInfo(metadata_str)
            self.xyzlut = client.XYZLut(self.sensor_info)
            self.metadata_received = True
            rospy.loginfo("Sensor metadata loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Error processing metadata: {e}")

    def store_pcl(self, pcl_msg):
        """Store the latest point cloud message"""
        self.latest_pcl = pcl_msg
        



    def range_image_callback(self, msg):
        try:
            range_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_range_image = range_image*4
            self.log_counter += 1
            if self.log_counter % 10 == 0:
                rospy.loginfo(f"Received range image of shape: {range_image.shape}")

        except Exception as e:
            rospy.logerr(f"Error processing range image: {e}")

    def visualize_keypoints(self, frame, keypoints):
        for i, (x, y, confidence) in enumerate(keypoints):
            if confidence > 0.5:  # Only consider keypoints with high confidence
                cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow("Keypoints", frame)
        cv2.waitKey(1)  # Use waitKey(1) to avoid blocking

    def image_callback(self, msg):


        try:
            
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rospy.loginfo(f"Received image of shape: {frame.shape}")

            results = self.model(frame, show=False)

            combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            detected = False

            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.data[0].cpu().numpy()
                    rospy.loginfo(f"Keypoints shape: {keypoints.shape}")

                    if keypoints.shape[0] == 0:
                        rospy.logwarn("No keypoints detected in this frame.")
                        continue

                    # Extract hand region mask
                    mask = self.get_chest_region_mask(keypoints, frame.shape[:2])
                    if mask is not None:
                        combined_mask = cv2.bitwise_or(combined_mask, mask)
                        detected = True

            if detected:
                self.visualize_mask(combined_mask)
                rospy.loginfo("Chest region mask created successfully.")

                if self.latest_pcl is not None and self.latest_range_image is not None:
                    filtered_points = self.extract_filtered_points(combined_mask, self.latest_range_image, self.latest_pcl)
                    self.chest_pub.publish(filtered_points)
                 
                    rospy.loginfo("Filtered chest points published successfully.")
                    
                    # # Convert ROS PointCloud2 to Open3D point cloud
                    # points_list = list(pc2.read_points(filtered_points, field_names=("x", "y", "z"), skip_nans=True))
                    # rospy.loginfo(f"Number of points in the point cloud: {len(points_list)}")
                    # o3d_cloud = o3d.geometry.PointCloud()
                    # o3d_cloud.points = o3d.utility.Vector3dVector(np.array(points_list))
                    # # Visualize the point cloud using Open3D
                    # o3d.visualization.draw_geometries([o3d_cloud])
                   

                    
                    # # Define the save location and ensure the directory exists
                    # file_path = os.path.expanduser("~/mannequin_detection/extracted_clouds/front/filtered_chest_points.ply")
                    # os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    # # Save the point cloud to a file
                    # o3d.io.write_point_cloud(file_path, o3d_cloud)
                    # rospy.loginfo(f"Filtered chest points saved successfully to {file_path}.")
                

                else:
                    rospy.logwarn("No point cloud or range image data available.")

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
            rospy.logerr(f"Error details: {type(e).__name__}, {str(e)}")

    def get_chest_region_mask(self, keypoints, image_shape):
        """Create a mask for the chest region using keypoints"""
        left_shoulder_idx = 5
        right_shoulder_idx = 6
        left_hip_idx = 11
        right_hip_idx = 12

        # Get keypoint coordinates
        left_shoulder = keypoints[left_shoulder_idx][:2]
        right_shoulder = keypoints[right_shoulder_idx][:2]
        left_hip = keypoints[left_hip_idx][:2]
        right_hip = keypoints[right_hip_idx][:2]

        # Expand the polygon by a margin to cover more points

        chest_polygon = np.array([
            left_shoulder,
            right_shoulder,
            right_hip,
            left_hip
        ], dtype=np.int32)

        # Create a blank mask
        mask = np.zeros(image_shape[:2], dtype=np.uint8)

        # Fill the polygon to create the chest mask
        cv2.fillPoly(mask, [chest_polygon], 255)

        return mask

    def visualize_mask(self, mask):
        # Convert mask to a three-channel image with color
        mask_visual = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_visual[mask > 0] = [0, 0, 255]  # Red color for the mask
        cv2.imshow("Chest Mask", mask_visual)
        cv2.waitKey(1)  # Use waitKey(1) to avoid blocking



    def extract_filtered_points(self, mask, range_image, pcl_msg):
        rospy.loginfo("Extracting filtered points with all fields...")
        if not self.sensor_info or not self.xyzlut:
            rospy.logwarn("Metadata or lookup table not ready.")
            return

        try:
            # Ensure the mask and range image have the same dimensions
            if mask.shape != range_image.shape:
                rospy.logerr("Mask and range image dimensions do not match.")
                return

            # Step 1: Apply the mask to the range image
            mask_bool = mask.astype(bool)
            roi_range_image = np.where(mask_bool,range_image, np.nan)

            # Step 2: Destagger the range image
            destaggered_image = client.destagger(self.sensor_info, roi_range_image, inverse=True)
            flattened_image = destaggered_image.flatten()

            # # printing range values
            # print(destaggered_image)

            # Get the field information from the original point cloud
            fields = pcl_msg.fields
            field_names = [field.name for field in fields]
            rospy.loginfo(f"Original point cloud fields: {field_names}")

            # Convert original point cloud to structured numpy array
            original_cloud_points = list(pc2.read_points(pcl_msg, skip_nans=False, field_names=field_names))
            dtype_list = []
            for field in fields:
                if field.datatype == PointField.FLOAT32:
                    dtype_list.append((field.name, np.float32))
                elif field.datatype == PointField.FLOAT64:
                    dtype_list.append((field.name, np.float64))
                elif field.datatype == PointField.UINT32:
                    dtype_list.append((field.name, np.uint32))
                elif field.datatype == PointField.INT32:
                    dtype_list.append((field.name, np.int32))
                elif field.datatype == PointField.UINT8:
                    dtype_list.append((field.name, np.uint8))
                elif field.datatype == PointField.INT8:
                    dtype_list.append((field.name, np.int8))
                elif field.datatype == PointField.UINT16:
                    dtype_list.append((field.name, np.uint16))
                elif field.datatype == PointField.INT16:
                    dtype_list.append((field.name, np.int16))
                else:
                    rospy.logwarn(f"Unsupported field datatype: {field.datatype}")
                    dtype_list.append((field.name, np.float32))  # Default to float32 if unsupported

            original_cloud_array = np.array(original_cloud_points, dtype=dtype_list)

            # Create DataFrames for easy matching
            original_cloud_df = pd.DataFrame(original_cloud_array)
            print(original_cloud_df.shape)
            #print(original_cloud_df.head(10))
            matched_rows = original_cloud_df[original_cloud_df['range'].isin(flattened_image)]


            # Publishing Matched Rows as PointCloud2
            if not matched_rows.empty:
                rospy.loginfo(f"Matched {len(matched_rows)} points. Publishing filtered point cloud.")
                
                # Convert matched rows DataFrame to structured array
                matched_points = matched_rows.to_records(index=False)

                # Define PointCloud2 fields
                pc2_fields = []
                offset = 0
                for field in fields:
                    if field.name in matched_rows.columns:
                        pc2_fields.append(PointField(name=field.name, offset=offset, datatype=field.datatype, count=1))
                        offset += np.dtype(dtype_list[field_names.index(field.name)][1]).itemsize

                # Create PointCloud2 message
                header = pcl_msg.header
                filtered_cloud = pc2.create_cloud(header, pc2_fields, matched_points)
                return filtered_cloud
                
            

        except Exception as e:
            rospy.logerr(f"Error extracting filtered points: {e}")
            rospy.logerr(f"Error details: {type(e).__name__}, {str(e)}")
            return None





def ros_thread():
    HandSegmentation()
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('hand_segmentation_node', log_level=rospy.DEBUG)
    ros_thread()
