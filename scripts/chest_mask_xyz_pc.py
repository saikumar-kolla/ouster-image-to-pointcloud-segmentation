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
from scipy.spatial import cKDTree

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
                    
                    # Convert ROS PointCloud2 to Open3D point cloud
                    points_list = list(pc2.read_points(filtered_points, field_names=("x", "y", "z"), skip_nans=True))
                    rospy.loginfo(f"Number of points in the point cloud: {len(points_list)}")
                    o3d_cloud = o3d.geometry.PointCloud()
                    o3d_cloud.points = o3d.utility.Vector3dVector(np.array(points_list))
                    # Visualize the point cloud using Open3D
                    o3d.visualization.draw_geometries([o3d_cloud])
                   

                    
                    # Define the save location and ensure the directory exists
                    file_path = os.path.expanduser("~/mannequin_detection/extracted_clouds/front/filtered_chest_points.ply")
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    # Save the point cloud to a file
                    o3d.io.write_point_cloud(file_path, o3d_cloud)
                    rospy.loginfo(f"Filtered chest points saved successfully to {file_path}.")
                

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
        """Extract filtered points using the mask and range image"""

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
            roi_range_image = np.where(mask_bool,range_image,np.nan)

            # Step 2: Destagger the range image
            destaggered_image = client.destagger(self.sensor_info, roi_range_image, inverse=True)

            # Step 3: Clean the NaN values before passing to xyzlut
            destaggered_image_clean = np.nan_to_num(destaggered_image, nan=0)

            # step 4: Convert the masked range image to XYZ points
            xyz_points = self.xyzlut(destaggered_image_clean).reshape(-1, 3)
            valid_points = xyz_points[~np.isnan(xyz_points).any(axis=1)]
            print(valid_points.shape)
            
       
            # Ensure there are valid points before creating the point cloud message
            if len(valid_points) == 0:
                rospy.logwarn("No valid points found in the filtered point cloud.")
                return


            # Define point cloud fields
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),

            ]

            # Create a PointCloud2 message and publish
            header = Header(stamp=pcl_msg.header.stamp, frame_id="os_sensor")
            pc_msg = pc2.create_cloud(header, fields, valid_points)
            rospy.loginfo("Filtered point cloud published successfully.")

            return pc_msg
        

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
