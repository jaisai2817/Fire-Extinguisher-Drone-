import cv2
import numpy as np
import pytesseract
import time
from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
import RPi.GPIO as GPIO
from pymavlink import mavutil 
import math
import socket
import argparse
import geopy.distance


# Set Tesseract path if needed (not required for Raspberry Pi)
# pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Connect to the drone
# vehicle = connect('127.0.0.1:14550', wait_ready=True)
def connectMyCopter():
  parser =  argparse.ArgumentParser(description='commands')
  parser.add_argument('--connect')
  args = parser.parse_args()

  connection_string = args.connect
  baud_rate = 57600
  print("\nConnecting to vehicle on: %s" % connection_string)
  vehicle = connect(connection_string,baud=baud_rate,wait_ready=True)
  return vehicle

# Servo motor GPIO setup
SERVO1_PIN = 17  # GPIO pin for Servo 1
SERVO2_PIN = 27  # GPIO pin for Servo 2
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO1_PIN, GPIO.OUT)
GPIO.setup(SERVO2_PIN, GPIO.OUT)
servo1 = GPIO.PWM(SERVO1_PIN, 50)  # 50 Hz PWM frequency
servo2 = GPIO.PWM(SERVO2_PIN, 50)
servo1.start(0)
servo2.start(0)

# Camera configuration
cap = cv2.VideoCapture(0)  # Change if using an external camera (try index 1 or 2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Define movement thresholds
FRAME_CENTER_X = 960  # Assuming a 1920x1080 video frame
FRAME_CENTER_Y = 540
THRESHOLD = 50  # Pixels within this range are considered "centered"
SEARCH_TIME_LIMIT = 60  # Maximum time to search at each location (seconds)

# Define GPS locations
locations = [
    {"lat": 12.971598, "lon": 77.594566, "alt": 10},  # Location 1
    {"lat": 12.972000, "lon": 77.594800, "alt": 10},  # Location 2
    {"lat": 12.973000, "lon": 77.595000, "alt": 10},  # Location 3
    {"lat": 12.974000, "lon": 77.595200, "alt": 10},  # Location 4
    {"lat": 12.975000, "lon": 77.595400, "alt": 10},  # Location 5
]
home_location = {"lat": 12.971598, "lon": 77.594566, "alt": 10}  # Home location

def arm_and_takeoff(aTargetAltitude):
    """
    Arms vehicle and fly to aTargetAltitude.
    """

    print("Basic pre-arm checks")
    # Don't let the user try to arm until autopilot is ready
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

        
    print("Arming motors")
    # Copter should arm in GUIDED mode
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    
    time.sleep(3)

    

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude) # Take off to target altitude

    # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command 
    #  after Vehicle.simple_takeoff will execute immediately).
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)      
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95: #Trigger just below target alt.
            print("Reached target altitude")
            break
        time.sleep(1)
        
def get_dstance(cord1, cord2):
    #return distance n meter
    return (geopy.distance.geodesic(cord1, cord2).km)*1000 

def move_drone(x_offset, y_offset):
    """Adjust drone's position based on detected target"""
    forward_backward = 0.5 if y_offset > THRESHOLD else -0.5 if y_offset < -THRESHOLD else 0
    left_right = 0.5 if x_offset > THRESHOLD else -0.5 if x_offset < -THRESHOLD else 0

    if forward_backward != 0 or left_right != 0:
        print(f"Moving drone: Forward/Backward: {forward_backward}, Left/Right: {left_right}")
        vehicle.send_ned_velocity(forward_backward, left_right, 0)

def detect_number(roi):
    """Recognize number '1' inside the detected colored circle"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)  # Faster with smaller kernel
    _, adaptive_thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY_INV)  # Simple binary thresholding

    # Use Tesseract OCR to detect number 1
    config = '--psm 6 -c tessedit_char_whitelist=1'  # Set whitelist for '1' only
    text = pytesseract.image_to_string(adaptive_thresh, config=config)

    return "1" in text

def detect_and_align():
    """Detect the colored circle with number 1 inside and align the system."""
    print("Starting image detection...")
    start_time = time.time()

    while time.time() - start_time < SEARCH_TIME_LIMIT:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red, Yellow, Black color ranges in HSV
        # Red
        lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
        # Yellow
        lower_yellow, upper_yellow = np.array([20, 100, 100]), np.array([30, 255, 255])
        # Black (using low saturation and value to detect dark colors)
        lower_black, upper_black = np.array([0, 0, 0]), np.array([180, 255, 50])

        # Create masks for each color
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        mask_black = cv2.inRange(hsv, lower_black, upper_black)

        # Combine masks
        mask = cv2.bitwise_or(mask_red, mask_yellow)
        mask = cv2.bitwise_or(mask, mask_black)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area > 2000:  # Increased threshold for long-range detection
                x, y, w, h = cv2.boundingRect(largest_contour)
                roi = frame[y:y+h, x:x+w]

                # Visualize the detection (optional, can remove this later)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Area: {area}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Show the video stream to verify
                cv2.imshow("Video Stream", frame)

                # Detect number "1" inside the red circle
                if detect_number(roi):
                    center_x = x + w // 2
                    center_y = y + h // 2

                    # Calculate offsets from the frame center
                    x_offset = center_x - FRAME_CENTER_X
                    y_offset = FRAME_CENTER_Y - center_y

                    print(f"Colored circle detected with number 1 at position: ({center_x}, {center_y})")
                    print(f"Offsets from center: X: {x_offset}, Y: {y_offset}")
                    return True
                else:
                    print("Colored circle detected but no number 1 inside.")
        else:
            print("No target detected.")

        # Add a break condition to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            break

    return False

def drop_payload():
    """Trigger both servos to release the payload"""
    print("🔽 Dropping payload...")
    servo1.ChangeDutyCycle(7.5)  # Move servo 1 to release position
    servo2.ChangeDutyCycle(7.5)  # Move servo 2 to release position
    time.sleep(1)
    servo1.ChangeDutyCycle(2.5)  # Reset servos
    servo2.ChangeDutyCycle(2.5)
    print("✅ Payload dropped!")

def goto_location(to_lat, to_long): 
        
    print(" Global Location (relative altitude): %s" % vehicle.location.global_relative_frame)
    curr_lat = vehicle.location.global_relative_frame.lat
    curr_lon = vehicle.location.global_relative_frame.lon
    curr_alt = vehicle.location.global_relative_frame.alt

    # set to locaton (lat, lon, alt)
    to_lat = to_lat
    to_lon = to_long
    to_alt = curr_alt

    to_pont = LocationGlobalRelative(to_lat,to_lon,to_alt)
    vehicle.simple_goto(to_pont, groundspeed=3)
    
    to_cord = (to_lat, to_lon)
    while True:
        curr_lat = vehicle.location.global_relative_frame.lat
        curr_lon = vehicle.location.global_relative_frame.lon
        curr_cord = (curr_lat, curr_lon)
        print("curr location: {}".format(curr_cord))
        distance = get_dstance(curr_cord, to_cord)
        print("distance ramaining {}".format(distance))
        if distance <= 2:
            print("Reached within 2 meters of target location...")
            break
        time.sleep(1)

def rtl():
    print("Returning to Launch...")
    vehicle.mode = VehicleMode("RTL")

# Mission Execution
# takeoff(5)

vehicle = connectMyCopter()
time.sleep(2)
arm_and_takeoff(5)

for loc in locations:
    goto_location(loc["lat"], loc["lon"])
    vehicle.mode = VehicleMode("LOITER")
    time.sleep(15)  # Loiter for 15 seconds

    if detect_and_align():
        print("Target detected. Dropping payload.")
        servo1.ChangeDutyCycle(12.5)  # Rotate servo motor 180 degrees for drop mechanism
        time.sleep(1)
        servo1.ChangeDutyCycle(2.5)  # Reset servo motor
    else:
        print("Target not detected. Moving to next location.")

# If all locations have been visited, return home
print("Mission complete! Returning to home.")
# go_to_location(home_location)
rtl()

# Cleanup
cap.release()
servo1.stop()
servo2.stop()
GPIO.cleanup()
vehicle.close()
