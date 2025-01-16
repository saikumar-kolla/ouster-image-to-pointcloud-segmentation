import rospy
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import String
import numpy as np
from ouster.sdk import client
import sensor_msgs.point_cloud2 as pc2
import json
from cv_bridge import CvBridge

class RangeToPCD:
    def __init__(self):
        rospy.init_node('range_to_pcd', anonymous=True)
        
        self.metadata = None
        self.range_image = None
        self.sensor_info = None
        self.xyzlut = None
        self.bridge = CvBridge()
        
        # Subscribers
        self.metadata_sub = rospy.Subscriber('/ouster/metadata', String, self.metadata_callback)
        self.range_sub = rospy.Subscriber('/ouster/range_image', Image, self.range_callback)
        
        # Publisher
        self.pcd_pub = rospy.Publisher('/range_pcd',PointCloud2,queue_size=10)
        
    def metadata_callback(self, msg):
        try:
            metadata = json.loads(msg.data)
            metadata_str = json.dumps(metadata)
            self.sensor_info = client.SensorInfo(metadata_str)
            self.xyzlut = client.XYZLut(self.sensor_info)
            rospy.loginfo("Sensor metadata loaded successfully.")
        except Exception as e:
            rospy.logerr(f"Error processing metadata: {e}")
        
    def range_callback(self, msg):
        try:
            range_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.range_image = range_image*4
            self.publish_point_cloud()
        except Exception as e:
            rospy.logerr(f"Error processing range image: {e}")
        
    def publish_point_cloud(self):
        if self.sensor_info is not None and self.range_image is not None:
            try:
                # Destagger the range image
                destaggered_image = client.destagger(self.sensor_info, self.range_image, inverse=True)
                
                # Convert range image to XYZ points
                xyz_points = self.xyzlut(destaggered_image).reshape(-1, 3)
                
                # Filter out invalid points
                valid_points = xyz_points[~np.isnan(xyz_points).any(axis=1)]
                
                # Create PointCloud2 message
                header = rospy.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = 'os_sensor'
                point_cloud = pc2.create_cloud_xyz32(header, valid_points)
                
                # Publish the point cloud
                self.pcd_pub.publish(point_cloud)
                rospy.loginfo("Point cloud published successfully.")
            except Exception as e:
                rospy.logerr(f"Error converting range image to point cloud: {e}")

if __name__ == '__main__':
    try:
        node = RangeToPCD()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
