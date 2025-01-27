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
                    filtered_points = self.extract_filtered_points(combined_mask, self.latest_range_image, self.latest_pcl)
                    self.hand_pub.publish(filtered_points)
                        
                        # # Convert ROS PointCloud2 to Open3D point cloud
                    points_list = list(pc2.read_points(filtered_points, field_names=("x", "y", "z"), skip_nans=True))
                    rospy.loginfo(f"Number of points in the point cloud: {len(points_list)}")
                    o3d_cloud = o3d.geometry.PointCloud()
                    o3d_cloud.points = o3d.utility.Vector3dVector(np.array(points_list))
                    # Visualize the point cloud using Open3D
                    o3d.visualization.draw_geometries([o3d_cloud])
                   

                    
                    # Define the save location and ensure the directory exists
                    file_path = os.path.expanduser("~/mannequin_detection/extracted_clouds/front/filtered_hand_points.ply")
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)

                    # Save the point cloud to a file
                    o3d.io.write_point_cloud(file_path, o3d_cloud)
                    rospy.loginfo(f"Filtered chest points saved successfully to {file_path}.")


                

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
        rospy.loginfo("Extracting filtered points...")
        if not self.sensor_info or not self.xyzlut:
            rospy.logwarn("Metadata or lookup table not ready.")
            return

        try:
            #Step 1: Destagger the range image first
            mask_bool = mask.astype(bool)
            roi_range_image = np.where(mask_bool,np.nan, range_image)
            rospy.loginfo(f"ROI range image shape: {roi_range_image.shape}")
            rospy.loginfo(f"range image shape: {range_image.shape}")
            destaggered_image = client.destagger(self.sensor_info, roi_range_image, inverse=True)

            # Step 2: Convert the range image to XYZ points
            xyz_points = self.xyzlut(destaggered_image).reshape(-1, 3)

            # Step 3: Apply the mask to the XYZ points
           

            # Step 4: Filter out NaN values
            valid_points = xyz_points[~np.isnan(xyz_points).any(axis=1)]
            # Ensure there are valid points before creating the point cloud message
            if len(valid_points) == 0:
                rospy.logwarn("No valid points found in the filtered point cloud.")
                return

            # Convert valid points to float32 to avoid type conversion issues
            #valid_points = valid_points.astype(np.float32)

            rospy.loginfo(f"Publishing {len(valid_points)} unique valid XYZ points.")


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
            rospy.logerr(f"Error filtering points: {e}")




def ros_thread():
    HandSegmentation()
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('hand_segmentation_node', log_level=rospy.DEBUG)
    ros_thread()
