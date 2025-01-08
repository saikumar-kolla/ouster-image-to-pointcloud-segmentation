import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('/home/kolla/Downloads/yolo11x-seg.pt')

# Initialize ROS node
rospy.init_node('yolo_realtime_detection', anonymous=True)

# Create a CvBridge instance
bridge = CvBridge()

def image_callback(msg):
    """
    Callback function to process images received from the ROS topic.
    """
    try:
        # Convert ROS Image message to OpenCV format
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Run YOLO inference
        results = model(frame, show=True)  # Set show=True to display results

        # Optionally, save or process the results
        for result in results:
            print(result.boxes)  # Example: print bounding boxes

    except Exception as e:
        rospy.logerr(f"Error processing image: {e}")

# Subscribe to the topic
image_topic = "/ouster/reflec_image"
rospy.Subscriber(image_topic, Image, image_callback)

# Keep the node alive
rospy.loginfo("YOLO Real-Time Detection Node Started...")
rospy.spin()
