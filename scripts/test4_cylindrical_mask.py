import threading
import rospy
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointField
from cv_bridge import CvBridge
from ouster.sdk import client
import std_msgs.msg
from std_msgs.msg import Header
import json
import numpy as np
import cv2
from ultralytics import YOLO
import time
import open3d as o3d
import pandas as pd
import os

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
        self.hand_pub = rospy.Publisher("/filtered_hand_points", PointCloud2, queue_size=1)

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

        # Left arm keypoint indices (shoulder, elbow, wrist)
        self.left_hand_indices = [5, 7, 9]

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
            # https://github.com/ouster-lidar/ouster-ros/issues/250#issuecomment-1801350711 read this for more info on multiplying by 4
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
                    mask = self.get_hand_region_mask(keypoints, frame.shape[:2])
                    if mask is not None:
                        combined_mask = cv2.bitwise_or(combined_mask, mask)
                        detected = True

            if detected:
                self.visualize_mask(combined_mask)
                rospy.loginfo("Hand region mask created successfully.")

                if self.latest_pcl is not None and self.latest_range_image is not None:
                    self.extract_filtered_points(combined_mask, self.latest_range_image, self.latest_pcl)
                    
                    


                

                else:
                    rospy.logwarn("No point cloud or range image data available.")

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
            rospy.logerr(f"Error details: {type(e).__name__}, {str(e)}")

    def get_hand_region_mask(self, keypoints, image_shape):
        try:
            # Extract keypoints
            shoulder_x, shoulder_y = keypoints[5, :2]
            elbow_x, elbow_y = keypoints[7, :2]
            wrist_x, wrist_y = keypoints[9, :2]
            confidences = keypoints[[5, 7, 9], 2]

            if np.all(confidences > 0.5):
                mask = np.zeros(image_shape[:2], dtype=np.uint8)

                # Calculate dynamic thickness
                upper_arm_length = np.linalg.norm(np.array([shoulder_x, shoulder_y]) - np.array([elbow_x, elbow_y]))
                forearm_length = np.linalg.norm(np.array([elbow_x, elbow_y]) - np.array([wrist_x, wrist_y]))
                thickness = int(0.1 * (upper_arm_length + forearm_length))

                # Ensure thickness is within valid range
                if thickness <= 0:
                    thickness = 1  # Set to default value if out of range

                # Convert keypoints to integers
                shoulder = (int(shoulder_x), int(shoulder_y))
                elbow = (int(elbow_x), int(elbow_y))
                wrist = (int(wrist_x), int(wrist_y))

                # Calculate perpendicular vectors for thickness
                def perpendicular_vector(p1, p2, thickness):
                    direction = np.array([p2[1] - p1[1], p1[0] - p2[0]])
                    direction = direction / np.linalg.norm(direction) * thickness
                    return direction

                # Calculate polygon points for upper arm
                upper_arm_perp = perpendicular_vector(shoulder, elbow, thickness)
                upper_arm_polygon = np.array([
                    shoulder + upper_arm_perp,
                    shoulder - upper_arm_perp,
                    elbow - upper_arm_perp,
                    elbow + upper_arm_perp
                ], dtype=np.int32)

                # Calculate polygon points for forearm
                forearm_perp = perpendicular_vector(elbow, wrist, thickness)
                forearm_polygon = np.array([
                    elbow + forearm_perp,
                    elbow - forearm_perp,
                    wrist - forearm_perp,
                    wrist + forearm_perp
                ], dtype=np.int32)

                # Draw polygons to create the mask
                cv2.fillConvexPoly(mask, upper_arm_polygon, 255)
                cv2.fillConvexPoly(mask, forearm_polygon, 255)

                # Extend the mask to cover the hand
                direction = np.array([wrist_x - elbow_x, wrist_y - elbow_y])
                direction = direction / np.linalg.norm(direction)
                hand_tip = np.array([wrist_x, wrist_y]) + 5 * direction
                hand_tip = (int(hand_tip[0]), int(hand_tip[1]))

                # Draw a circle at the hand tip
                cv2.circle(mask, hand_tip, 5, 255, -1)

                rospy.loginfo(f"Created mask with shape {mask.shape}")
                return mask
            else:
                rospy.logwarn("Low confidence in hand keypoints.")
                return None
        except Exception as e:
            rospy.logerr(f"Error in get_hand_region_mask: {e}")
            return None


    def visualize_mask(self, mask):
        # Convert mask to a three-channel image with color
        mask_visual = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_visual[mask > 0] = [0, 0, 255]  # Red color for the mask
        cv2.imshow("Hand Mask", mask_visual)
        cv2.waitKey(1)  # Use waitKey(1) to avoid blocking

    def extract_filtered_points(self, mask, range_image, pcl_msg):

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

            #checking mask coverage
            print(f"Mask Coverage: {mask_bool.sum()} / {mask_bool.size}")


            ''' Passing staggered range image to xyz function so that we can better visualize the cloud in rviz '''

            # roi_range_image = np.where(mask_bool,range_image,0)

            # # Step 2: stagger the range image
            # destaggered_image = client.destagger(self.sensor_info, roi_range_image, inverse=True)

            # # Pass the cleaned image to XYZLut without nan and if possible in unit32 format
            # xyz_points = self.xyzlut(destaggered_image).reshape(-1, 3)

            ''' Directly passing destaggered range image to xyz function so that we can match with original point cloud
            which was destaggered when published'''
            roi_range_image = np.where(mask_bool,range_image,0)
            xyz_points = self.xyzlut(roi_range_image).reshape(-1, 3)

            # Filter out invalid points post-conversion
            valid_xyz = xyz_points[~np.isnan(xyz_points).any(axis=1)]
   
            # Create a DataFrame for valid xyz points
            valid_xyz_df = pd.DataFrame(valid_xyz, columns=['x', 'y', 'z'])
            print("shape",valid_xyz_df.shape)
            # valid_xyz_df.to_csv('valid_xyz_df.csv', index=False)

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
            rospy.loginfo("converted pcl_msg to structured numpy array") 

            print(f"{dtype_list}")

            # Create a DataFrame for the original cloud
            original_cloud_df = pd.DataFrame(original_cloud_array)
            print("shape",original_cloud_df.shape)
            original_cloud_df.to_csv('original_cloud_df.csv', index=False)

            # Find rows in valid_xyz_df where all values are zero
            zero_rows_indices = valid_xyz_df[(valid_xyz_df == 0.0).all(axis=1)].index

            # Set corresponding rows in original_cloud_df to zero
            original_cloud_df.iloc[zero_rows_indices] = 0

            original_cloud_filtered = original_cloud_df
            # # Save matched rows to a CSV file
            # original_cloud_filtered.to_csv('original_segmented_cloud.csv', index=False)

            # Publishing Matched Rows as PointCloud2
            if not original_cloud_filtered.empty:
                

                # Visualize the segmented original cloud and also range image cloud in open3d
                matched_points_np = original_cloud_filtered[['x', 'y', 'z']].to_numpy(dtype=np.float32)
                valid_xyz_np = valid_xyz_df.to_numpy(dtype=np.float32)

                original_segmented_cloud = o3d.geometry.PointCloud()
                range_image_cloud = o3d.geometry.PointCloud()
                original_segmented_cloud.points = o3d.utility.Vector3dVector(matched_points_np)
                range_image_cloud.points = o3d.utility.Vector3dVector(valid_xyz_np)

                # Add Colors (R, G, B) in range [0, 1]
                original_segmented_cloud_color = [1, 0, 0]  # Red for matched cloud
                range_image_cloud_color = [0, 1, 0]  # Green for range image cloud

                original_segmented_cloud.colors = o3d.utility.Vector3dVector(np.tile(original_segmented_cloud_color, (matched_points_np.shape[0], 1)))
                range_image_cloud.colors = o3d.utility.Vector3dVector(np.tile(range_image_cloud_color, (valid_xyz_np.shape[0], 1)))

                o3d.visualization.draw_geometries([original_segmented_cloud, range_image_cloud,])


                # Convert matched rows DataFrame to structured array
                filtered_points = original_segmented_cloud.to_records(index=False)

                # Define PointCloud2 fields
                pc2_fields = [
                    PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1),
                    PointField('intensity', 12, PointField.FLOAT32, 1),
                    PointField('t', 16, PointField.UINT32, 1),
                    PointField('reflectivity', 16, PointField.UINT16, 1),
                    PointField('ring', 20, PointField.UINT16, 1),
                    PointField('ambient', 22, PointField.UINT16, 1),
                    PointField('range', 26, PointField.UINT32, 1)
                ]

                # Create PointCloud2 message
                header = pcl_msg.header
                filtered_cloud = pc2.create_cloud(header, pc2_fields, filtered_points)
                self.hand_pub.publish(filtered_cloud)
                rospy.loginfo("Filtered chest points published successfully.")
            else:
                rospy.logwarn("No points matched. No point cloud published.")
                return None    
            

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
