#stable
import cv2
import torch

# Load YOLOv5 model from PyTorch Hub
# We're using yolov5s (small) model for decent speed
model = torch.hub.load('yolov5', 'custom',
                        path='yolov5\yolov5s.pt', source ='local') 
model.classes = [0]  # Only detect 'person' class

# Load video file instead of webcam
cap = cv2.VideoCapture(0)  # Make sure people.mp4 is in the same folder or provide the full path

# Create the CSRT tracker
tracker = cv2.TrackerCSRT_create()

# Flags and variables
tracking = False
selection = None
detections_list = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to read frame.")
        break  # Exit when video ends or fails

    height, width = frame.shape[:2]
    frame_center = (width // 2, height // 2)

    if not tracking:
        detections_list = []

        # Run YOLO on the current frame
        results = model(frame)
        detections = results.xyxy[0]

        # Draw and number all detected people
        for idx, (*box, conf, cls) in enumerate(detections):
            x1, y1, x2, y2 = map(int, box)
            detections_list.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {idx + 1}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display prompt to pick a person to lock onto
        cv2.putText(frame, "Press number to lock on (1, 2, 3...)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        key = cv2.waitKey(1) & 0xFF
        if key >= ord('1') and key <= ord(str(len(detections_list))):
            selection = int(chr(key)) - 1
            x1, y1, x2, y2 = detections_list[selection]
            w, h = x2 - x1, y2 - y1
            tracker.init(frame, (x1, y1, w, h))
            tracking = True
            print(f"Locked onto Person {selection + 1}")

    else:
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            person_center = (x + w // 2, y + h // 2)

            dx = person_center[0] - frame_center[0]
            dy = person_center[1] - frame_center[1]
            print(f"Offset from center: dx={dx}, dy={dy}")

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, person_center, 5, (0, 0, 255), -1)
            cv2.line(frame, frame_center, person_center, (255, 0, 255), 2)
        else:
            print("Tracking lost!")
            tracking = False
            tracker = cv2.TrackerCSRT_create()  # Reset tracker

    cv2.circle(frame, frame_center, 5, (255, 255, 0), -1)

    cv2.imshow("Person Selector & Tracker", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


