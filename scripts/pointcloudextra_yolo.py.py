import rospy
from sensor_msgs.msg import Image, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
from ouster.sdk import client
import std_msgs.msg
import json
import numpy as np
import cv2
from ultralytics import YOLO
import time

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
        self.hand_pub = rospy.Publisher("/left_hand_points", PointCloud2, queue_size=1)

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
            #rospy.loginfo(f"Received range image of shape: {range_image.shape}")
            self.latest_range_image = range_image
            self.log_counter += 1
            if self.log_counter % 10 == 0:
                rospy.loginfo(f"Received range image of shape: {range_image.shape}")

        except Exception as e:
            rospy.logerr(f"Error processing range image: {e}")

#visualize keypoints on a image for checking

   # def visualize_keypoints(self, frame, keypoints):
    #    for i, (x, y, confidence) in enumerate(keypoints):
     #       if confidence > 0.5:  # Only consider keypoints with high confidence
      #          cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
       #         cv2.putText(frame, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        #cv2.imshow("Keypoints", frame)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rospy.loginfo(f"Received image of shape: {frame.shape}")

            results = self.model(frame, show=True)
            for result in results:
                if hasattr(result, 'keypoints') and result.keypoints is not None:
                    keypoints = result.keypoints.data[0].cpu().numpy()
                    rospy.loginfo(f"Keypoints shape: {keypoints.shape}")

                    if keypoints.shape[0] == 0:
                        rospy.logwarn("No keypoints detected in this frame.")
                        continue

                    mask, center = self.get_hand_region_mask(keypoints, frame.shape[:2])
                    if mask is None:
                        rospy.logwarn("Failed to create hand region mask.")
                        continue

                    rospy.loginfo(f"Mask shape: {mask.shape}, Center: {center}")

                    if self.latest_pcl is not None and self.latest_range_image is not None:
                        hand_points = self.extract_hand_points(mask, self.latest_range_image)
                        if hand_points is not None and len(hand_points) > 0:
                            header = self.latest_pcl.header
                            hand_pcl_msg = pc2.create_cloud_xyz32(header, hand_points)
                            self.hand_pub.publish(hand_pcl_msg)
                            rospy.loginfo(f"Published {len(hand_points)} points for left hand")
                    else:
                        rospy.logwarn("No point cloud or range image data available.")
            #self.visualize_keypoints(frame, keypoints)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
            rospy.logerr(f"Error details: {type(e).__name__}, {str(e)}")

    def get_hand_region_mask(self, keypoints, image_shape):
        try:
            hand_points = keypoints[self.left_hand_indices]
            confidences = hand_points[:, 2]
            print(confidences)

            if np.all(confidences > 0.5):
                wrist_x, wrist_y = hand_points[2, :2]
                elbow_x, elbow_y = hand_points[1, :2]
                arm_length = np.linalg.norm([wrist_x - elbow_x, wrist_y - elbow_y])
                radius = int(arm_length * 0.3)

                mask = np.zeros(image_shape, dtype=bool)
                y, x = np.ogrid[:image_shape[0], :image_shape[1]]
                mask[(x - int(wrist_x))**2 + (y - int(wrist_y))**2 <= radius**2] = True
                rospy.loginfo(f"Created mask with shape {mask.shape}, radius {radius}")
                return mask, (wrist_x, wrist_y)
            else:
                rospy.logwarn("Low confidence in hand keypoints.")
                return None, None
        except Exception as e:
            rospy.logerr(f"Error in get_hand_region_mask: {e}")
            return None, None

    def extract_hand_points(self, mask, range_image):
        if not self.metadata_received or self.xyzlut is None:
            rospy.logwarn("Metadata or XYZLut not ready.")
            return None

        try:
            cloud = self.xyzlut(range_image)
            hand_points = cloud[mask]
            rospy.loginfo(f"Extracted {hand_points.shape[0]} hand points")
            return hand_points
        except Exception as e:
            rospy.logerr(f"Error extracting hand points: {e}")
            rospy.logerr(f"Error details: {type(e).__name__}, {str(e)}")
            return None


if __name__ == '__main__':
    rospy.init_node('hand_segmentation_node')
    hand_seg = HandSegmentation()
    rospy.spin()
