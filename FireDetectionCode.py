import cv2
import numpy as np
import tflite_runtime.interpreter as tflite


# Load the TFLite model and allocate tensors
interpreter = tflite.Interpreter(model_path="/home/pi/Downloads/fire_detection_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the image
def preprocess_image(image):
    img = cv2.resize(image, (227, 227))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Initialize video capture
cap = cv2.VideoCapture(0)  # Change the index if you are using an external camera

# Class labels
C = ['fire', 'non fire']

# Main loop to capture frames and perform inference
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_image(frame)

    # Perform inference on the frame
    interpreter.set_tensor(input_details[0]['index'], preprocessed_frame)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Assuming output_data shape is (1, 2)
    predictions = np.squeeze(output_data)  # Squeeze to remove the batch dimension

    # Get the index of the class with the highest probability
    prediction = np.argmax(predictions)

    # Display the prediction on the frame
    label = C[prediction]
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Fire Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
    # Release the capture and close windows
        cap.release()
        cv2.destroyAllWindows()
        print("Input details:", input_details)

