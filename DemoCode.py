import cv2
import torch
import numpy as np
import random
import math

# Load YOLOv5 model
model = torch.hub.load('yolov5', 'custom', path='yolov5/yolov5s.pt', source='local')
model.classes = [0]  # Only detect people

# Webcam setup
cap = cv2.VideoCapture(0)

# Tracker setup
tracker = cv2.TrackerCSRT_create()
tracking = False
selection = None
car_moving_up = False
car_moving_down = False
mode = 'autonomous'

# LiDAR parameters
lidar_fov = 60
num_rays = 20
ray_length = 200

# Car parameters
car_x, car_y = 160, 240
car_speed = 2

# Static obstacles
obstacles = [(random.randint(100, 300), random.randint(50, 180), 10 + random.randint(0, 15)) for _ in range(5)]

# Display config
cv2.namedWindow("Simulated LiDAR + Person Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Simulated LiDAR + Person Tracking", 1280, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (320, 240))
    height, width = frame.shape[:2]
    frame_center = (width // 2, height // 2)

    # ---------- TRACKING ----------
    if not tracking and mode == 'autonomous':
        detections_list = []
        results = model(frame)
        detections = results.xyxy[0]

        for idx, (*box, conf, cls) in enumerate(detections):
            x1, y1, x2, y2 = map(int, box)
            detections_list.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Person {idx+1}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, "Press number to lock on (1, 2, 3...)", (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        key = cv2.waitKey(1) & 0xFF
        if key >= ord('1') and key <= ord(str(len(detections_list))):
            selection = int(chr(key)) - 1
            x1, y1, x2, y2 = detections_list[selection]
            w, h = x2 - x1, y2 - y1
            tracker.init(frame, (x1, y1, w, h))
            tracking = True
            print(f"Locked onto Person {selection + 1}")

    elif tracking and mode == 'autonomous':
        success, box = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            person_center = (x + w // 2, y + h // 2)
            dx = person_center[0] - frame_center[0]
            dy = person_center[1] - frame_center[1]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.circle(frame, person_center, 5, (0, 0, 255), -1)
            cv2.line(frame, frame_center, person_center, (255, 0, 255), 2)

            if person_center[0] < width // 3:
                car_moving_up = True
                car_moving_down = False
            elif person_center[0] > 2 * width // 3:
                car_moving_up = False
                car_moving_down = True
            else:
                car_moving_up = False
                car_moving_down = False
        else:
            tracking = False
            tracker = cv2.TrackerCSRT_create()

    # ---------- LIDAR + SIMULATION ----------
    sim_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    alert_triggered = False

    if mode == 'autonomous':
        if car_moving_up:
            car_y -= car_speed
        elif car_moving_down:
            car_y += car_speed

    # Draw car
    cv2.circle(sim_frame, (car_x, car_y), 8, (100, 100, 255), -1)

    # LiDAR rays (visual only)
    angles = np.linspace(-lidar_fov / 2, lidar_fov / 2, num=num_rays)
    for angle in angles:
        rad = np.radians(angle)
        end_x = int(car_x + ray_length * np.sin(rad))
        end_y = int(car_y - ray_length * np.cos(rad))
        cv2.line(sim_frame, (car_x, car_y), (end_x, end_y), (0, 255, 0), 1)

    # Draw obstacles
    for (ox, oy, r) in obstacles:
        cv2.circle(sim_frame, (ox, oy), r, (255, 255, 0), -1)

    # ---------- ALERT LOGIC (AUTONOMOUS ONLY) ----------
    if mode == 'autonomous':
        for (ox, oy, r) in obstacles:
            dx = ox - car_x
            dy = oy - car_y
            distance = math.hypot(dx, dy)

            if distance < 100:
                angle = math.degrees(math.atan2(dy, dx))
                relative_angle = angle - 270
                if relative_angle < -180:
                    relative_angle += 360
                elif relative_angle > 180:
                    relative_angle -= 360

                if -lidar_fov / 2 <= relative_angle <= lidar_fov / 2:
                    alert_triggered = True
                    cv2.line(sim_frame, (car_x, car_y), (ox, oy), (0, 0, 255), 2)
                    break

        # Forward dot as virtual obstacle
        forward_dot_y = 10
        dx = 0
        dy = forward_dot_y - car_y
        distance = math.hypot(dx, dy)
        if distance < 100:
            angle = math.degrees(math.atan2(dy, dx))
            relative_angle = angle - 270
            if relative_angle < -180:
                relative_angle += 360
            elif relative_angle > 180:
                relative_angle -= 360

            if -lidar_fov / 2 <= relative_angle <= lidar_fov / 2:
                alert_triggered = True
                cv2.line(sim_frame, (car_x, car_y), (car_x, forward_dot_y), (0, 0, 255), 2)

    # Fixed forward dot
    cv2.circle(sim_frame, (car_x, 10), 5, (255, 255, 0), -1)

    if alert_triggered and mode == 'autonomous':
        cv2.putText(sim_frame, "ALERT: Obstacle Ahead!", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Mode display
    cv2.putText(sim_frame, f"Mode: {mode.upper()}", (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Combine and show
    combined = np.hstack((sim_frame, frame))
    cv2.imshow("Simulated LiDAR + Person Tracking", combined)

    # ---------- KEYBOARD CONTROLS ----------
    key = cv2.waitKey(10) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('a'):
        mode = 'autonomous'
        print("Switched to AUTONOMOUS mode")
    elif key == ord('m'):
        mode = 'manual'
        print("Switched to MANUAL mode")
    elif key == ord('r') and mode == 'autonomous':
        tracking = False
        tracker = cv2.TrackerCSRT_create()
        print("Reset tracker – re-enter person selection")
    elif mode == 'manual':
        if key == 82 or key == ord('w'):
            car_y -= car_speed
        elif key == 84 or key == ord('s'):
            car_y += car_speed

    # Clamp car position
    car_y = np.clip(car_y, 20, 220)

cap.release()
cv2.destroyAllWindows()
