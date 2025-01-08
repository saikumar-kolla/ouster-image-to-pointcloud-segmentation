import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

# Specify the directory where you want to save the images
SAVE_DIR = '/home/kolla/mannequin_detection/reflec_images'

class ImageSaver:
    def __init__(self):
        """Initialize the node, subscriber, and required objects."""
        # Create the save directory if it doesn't exist
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
            rospy.loginfo(f"Directory '{SAVE_DIR}' created successfully.")
        else:
            rospy.loginfo(f"Directory '{SAVE_DIR}' already exists.")

        # Initialize the node
        rospy.init_node('image_saver_node', anonymous=True)

        # Set up the subscriber to the /ouster/reflec_image topic
        self.subscriber = rospy.Subscriber('/ouster/reflec_image', Image, self.callback)

        # CvBridge is used to convert ROS Image messages to OpenCV images
        self.bridge = CvBridge()

        # Counter to keep track of the image filenames
        self.image_count = 0

    def callback(self, msg):
        """Callback function for the subscriber."""
        try:
            # Convert the ROS Image message to a CV2 image (BGR)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Create a unique filename for the image
            image_filename = os.path.join(SAVE_DIR, f'image_{self.image_count:04d}.jpg')
            
            # Save the image using OpenCV
            cv2.imwrite(image_filename, cv_image)
            
            rospy.loginfo(f"Saved image: {image_filename}")
            
            # Increment the image count for the next image
            self.image_count += 1
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def run(self):
        """Run the ROS event loop."""
        rospy.loginfo("Image saver node is running...")
        rospy.spin()


if __name__ == '__main__':
    try:
        image_saver = ImageSaver()
        image_saver.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Node terminated.")
