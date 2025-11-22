"""
MINIMAL OBJECT DETECTION DEMO
==============================

This is the absolute simplest version - 15 lines of code!
Perfect for understanding the core concept.

Run this first to see how simple it can be.
Then check the full version for advanced features.
"""

import cv2
from ultralytics import YOLO
import time

# Load the model (downloads on first run, then uses cache)
model = YOLO('yolov8n.pt')

# Open your webcam
cap = cv2.VideoCapture(0)

print("🚀 Object Detection Started!")
print("Press 'q' to quit\n")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run inference (that's it!)
    results = model(frame, conf=0.5)
    
    # Draw results (YOLO does this automatically)
    annotated_frame = results[0].plot()
    
    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
    prev_time = current_time
    
    # Show FPS
    cv2.putText(annotated_frame, f'FPS: {fps:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display
    cv2.imshow('Object Detection', annotated_frame)
    
    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n✅ Done!")
