import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2
import time
import threading
import pyttsx3
from ultralytics import YOLO
from collections import Counter
from pynput import keyboard
from std_msgs.msg import Bool

class RealSenseSubscriber(Node):
    def __init__(self):
        super().__init__('realsense_subscriber')

        self.image_subscription = self.create_subscription(
            Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.depth_subscription = self.create_subscription(
            Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)

        self.subscription = self.create_subscription(
            Bool,
            'keypress_status',
            self.monitor_keypress,
            10)
        
        self.publisher_ = self.create_publisher(Bool, 'state_', 10)


        self.cv_bridge = CvBridge()
        self.color_frame = None
        self.depth_frame = None

        self.count_times = 0
        self.screenshot_count = 5
        self.screen_shot_frames = []
        # self.capture_active = True

        # Initialize the YOLO model
        self.model = YOLO('/home/redha/colcon_ws/src/emptyseats/yolov8m_ES_finetuned_v4.pt')
        self.get_logger().info("model is ready")

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        
        self.node_spin = True
        self.final_states = []

    def monitor_keypress(self, msg):
        if msg.data:
            # self.capture_active = True
            self.engine.say("capturing")
            self.engine.runAndWait()
            self.capture_frames()
        else:
            # Step 1: Flatten the list
            flattened = [item for sublist in self.final_states for item in sublist]
            zero_indices = [i + 1 for i, value in enumerate(flattened) if value == 0]
            self.engine.say(f"{zero_indices}")
            self.engine.runAndWait()
            self.node_spin = False

    def capture_frames(self):
        capture_counter = self.screenshot_count

        while capture_counter > 0:
            if self.color_frame is not None and self.depth_frame is not None:
                # Process and save the frames (color + depth)
                combined_image = self.process_frames(self.color_frame, self.depth_frame)
                self.screen_shot_frames.append(combined_image)
                self.get_logger().info(f"Captured frame {self.screenshot_count - capture_counter + 1}")
                capture_counter -= 1
                time.sleep(0.4)

        self.get_logger().info("Capture complete")
        # self.capture_active = False
        self.perform_detection()

    def image_callback(self, msg):
        # if not self.capture_active:
        #     return

        color_frame = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        np_frame = np.asanyarray(color_frame)
        self.color_frame = cv2.rotate(np_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    def depth_callback(self, msg):
        # if not self.capture_active:
        #     return

        depth_image = self.cv_bridge.imgmsg_to_cv2(msg, "passthrough")
        depth_image = np.array(depth_image, dtype=np.float32)
        depth = cv2.rotate(depth_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        depth *= 0.001  # Convert depth from millimeters to meters

        self.depth_frame = depth

    def process_frames(self, color_image, depth_image):
        inliers_mask = depth_image <= 2.0
        outliers_mask = depth_image > 2.0

        combined_image = np.zeros_like(color_image)
        combined_image[inliers_mask] = color_image[inliers_mask]
        combined_image[outliers_mask] = [255, 255, 255]  # white for outliers

        # cv2.imshow('Filtered Color Image (White Outliers)', combined_image)
        return combined_image

    def perform_detection(self):
        states = []
        for img in self.screen_shot_frames[0 + self.count_times: 5 + self.count_times]:
            state = self.detection(self.model, img)
            states.append(state)
        
        element_counts = Counter(tuple(x) for x in states)
        final_state = max(element_counts, key=element_counts.get)

        if self.count_times == self.screenshot_count:
            self.final_states.append(final_state[::-1])
        else:
            self.final_states.append(final_state)

        self.get_logger().info(f"Detected states: {states}")
        self.get_logger().info(f"Final states: {self.final_states}")
        # self.engine.say(f"{final_state}")
        # self.engine.runAndWait()
        msg = Bool()
        msg.data = True
        self.publisher_.publish(msg)
        self.count_times += self.screenshot_count

        # if (self.count_times == 10): self.node_spin = False



    def detection(self, model, image):
        results = model(image)

        chair_detections = []
        bag_detections = []
        person_detections = []

        # Iterate over detected objects
        for result in results:
            for box in result.boxes:
                # Extract the bounding box coordinates
                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]
                class_id = int(box.cls[0])
                confidence = box.conf[0].item()  # Convert to a Python float if necessary
                class_name = model.names[class_id]

                # Classify detected objects
                if class_name == "chair":
                    chair_detections.append((x1, y1, x2, y2, class_id, confidence))
                elif class_name == "bag":
                    bag_detections.append((x1, y1, x2, y2, class_id, confidence))
                elif class_name == "person" and confidence > 0.7:
                    person_detections.append((x1, y1, x2, y2, class_id, confidence))

                # # Draw bounding boxes
                # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # label = f"{class_name}"
                # cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # cv2.imshow('Detection Output', image)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()

        return self.process_frame(sorted(chair_detections, key=lambda x: x[0]), 
                                   sorted(bag_detections, key=lambda x: x[0]),
                                   sorted(person_detections, key=lambda x: x[0]))

    def process_frame(self, raw_chairs, raw_bags, raw_people):
        state = [0, 0, 0]
        tolerance = 100
        
        if len(raw_chairs) == 0:
            return [-1, -1, -1]
        
        max_y2 = max([chair[3] for chair in raw_chairs], default=0)
        
        chairs = [chair for chair in raw_chairs if abs(chair[3] - max_y2) <= tolerance]
        bags = raw_bags
        people = raw_people
            
        updated_chairs = chairs    

        for obj in bags + people:
            assigned_flag = False 
            for chair in chairs:
                if abs(obj[0] - chair[0]) < tolerance:
                    assigned_flag = True
                    break
            if not assigned_flag:
                updated_chairs.append(obj)
        
        if len(updated_chairs) != 3:
            self.get_logger().info("Take another image!")
            return [-1, -1, -1]

        sorted_chairs = sorted(updated_chairs, key=lambda x: x[0])
        for obj in bags + people:
            for index, chair in enumerate(sorted_chairs):
                if abs(obj[0] - chair[0]) < tolerance:
                    state[index] = 1
                    break
        
        return state

def main(args=None):
    rclpy.init(args=args)
    node = RealSenseSubscriber()

    try:
        while rclpy.ok() and node.node_spin:
            rclpy.spin_once(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
