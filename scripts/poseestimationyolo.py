import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('/home/kolla/Downloads/yolo11x-pose.pt')

# Initialize ROS node
rospy.init_node('yolo_pose_estimation', anonymous=True)

# Create a CvBridge instance
bridge = CvBridge()

def image_callback(msg):
    """
    Callback function to process images received from the ROS topic.
    """
    try:
        # Convert ROS Image message to OpenCV format
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Run YOLO inference with visualization enabled
        results = model(frame, show=True)  # Direct visualization of keypoints and skeleton
        
        # Optionally, log or process the keypoints
        for result in results:
            if hasattr(result, 'keypoints') and result.keypoints is not None:
                print("Keypoints:", result.keypoints)

    except Exception as e:
        rospy.logerr(f"Error processing image: {e}")

# Subscribe to the ROS image topic
image_topic = "/ouster/reflec_image"
rospy.Subscriber(image_topic, Image, image_callback)

# Keep the node alive
rospy.loginfo("YOLO Pose Estimation Node Started...")
rospy.spin()
