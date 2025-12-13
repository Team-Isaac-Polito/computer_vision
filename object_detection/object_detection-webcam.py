# This script demonstrates how to use the YOLO model for object detection using a webcam.
# It captures video from the webcam, processes each frame with the YOLO model, and displays the detected objects with bounding boxes and labels.
# It's for test purposes only to check if the model is working correctly on camera.


from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("runs/detect/train/weights/best.pt")

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to tensor
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img)
    
    # Process results
    for result in results:
        bbox = result.boxes.xyxy
        labels = [result.names[cls.item()] for cls in result.boxes.cls.int()]
        confs = result.boxes.conf
        for i, conf in enumerate(confs):
            if conf > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = bbox[i]
                label = labels[i]
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'{label}: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('YOLO Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()