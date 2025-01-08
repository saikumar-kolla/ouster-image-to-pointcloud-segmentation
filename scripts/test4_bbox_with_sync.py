import threading
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
import cv2
from ultralytics import YOLO
import time
import open3d as o3d
import message_filters

class ObjectSegmentation:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = YOLO('/home/kolla/Downloads/yolo11x-pose.pt')

        # Subscribe to synchronized topics
        image_sub = message_filters.Subscriber("/ouster/reflec_image", Image)
        pcl_sub = message_filters.Subscriber("/ouster/points", PointCloud2)
        range_image_sub = message_filters.Subscriber("/ouster/range_image", Image)

        # Synchronize the topics
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [image_sub, pcl_sub, range_image_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.combined_callback)

        # Metadata subscriber
        self.metadata_sub = rospy.Subscriber("/ouster/metadata", std_msgs.msg.String, self.metadata_callback)

        # Publisher for filtered point cloud
        self.filtered_pub = rospy.Publisher("/filtered_points", PointCloud2, queue_size=1)

        # Store latest data
        self.latest_pcl = None
        self.latest_range_image = None

        # Sensor metadata
        self.sensor_info = None
        self.metadata = None
        self.xyzlut = None

        # Processing flag to prevent overlapping calls
        self.processing = False

    def metadata_callback(self, msg):
        try:
            self.metadata = json.loads(msg.data)
            metadata_str = json.dumps(self.metadata)
            self.sensor_info = client.SensorInfo(metadata_str)
            self.xyzlut = client.XYZLut(self.sensor_info)
            rospy.loginfo("Sensor metadata loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Error processing metadata: {e}")

    def combined_callback(self, image_msg, pcl_msg, range_image_msg):
        if self.processing:
            return

        self.processing = True

        try:
            self.image_callback(image_msg)
            self.store_pcl(pcl_msg)
            self.range_image_callback(range_image_msg)
        except Exception as e:
            rospy.logerr(f"Error in combined callback: {e}")
        finally:
            self.processing = False

    def store_pcl(self, pcl_msg):
        self.latest_pcl = pcl_msg

    def range_image_callback(self, msg):
        try:
            range_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.latest_range_image = range_image*4
            rospy.loginfo(f"Received range image of shape: {range_image.shape}")
        except Exception as e:
            rospy.logerr(f"Error processing range image: {e}")

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rospy.loginfo(f"Received image of shape: {frame.shape}")

            results = self.model.track(frame, show=True)
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    box = result.boxes[0]
                    mask = self.get_bbox_mask(box.xyxy[0].cpu().numpy(), frame.shape[:2])

                    if mask is None:
                        rospy.logwarn("Failed to create bounding box mask.")
                        continue

                    self.visualize_mask(mask)
                    if self.latest_pcl is not None and self.latest_range_image is not None:
                        self.extract_filtered_points(mask, self.latest_range_image, self.latest_pcl)
                    else:
                        rospy.logwarn("No point cloud or range image data available.")
                    break
                
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def get_bbox_mask(self, bbox, image_shape):
        try:
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(image_shape[1], int(x2)), min(image_shape[0], int(y2))
            mask = np.zeros(image_shape, dtype=np.uint8)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
            rospy.loginfo(f"Created bounding box mask with bounds: ({x1}, {y1}), ({x2}, {y2})")
            return mask
        except Exception as e:
            rospy.logerr(f"Error creating bounding box mask: {e}")
            return None

    def visualize_mask(self, mask):
        mask_visual = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        mask_visual[mask > 0] = [0, 0, 255]
        cv2.imshow("Mask", mask_visual)
        cv2.waitKey(1)

    def extract_filtered_points(self, mask, range_image, pcl_msg):
        if not self.sensor_info or not self.xyzlut:
            rospy.logwarn("Metadata or lookup table not ready.")
            return

        try:
            mask_bool = mask.astype(bool)
            roi_range_image = np.where(mask_bool, range_image, np.nan)
            destaggered_image = client.destagger(self.sensor_info, roi_range_image)
            xyz_points = self.xyzlut(destaggered_image).reshape(-1, 3)
            valid_points = xyz_points[~np.isnan(xyz_points).any(axis=1)]

            rospy.loginfo(f"Publishing {len(valid_points)} valid XYZ points.")
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
            ]
            header = Header(stamp=pcl_msg.header.stamp, frame_id="os_sensor")
            pc_msg = pc2.create_cloud(header, fields, valid_points)
            self.filtered_pub.publish(pc_msg)
            rospy.loginfo("Filtered point cloud published successfully.")

        except Exception as e:
            rospy.logerr(f"Error filtering points: {e}")

if __name__ == '__main__':
    rospy.init_node('object_segmentation_node', log_level=rospy.DEBUG)
    ObjectSegmentation()
    rospy.spin()
