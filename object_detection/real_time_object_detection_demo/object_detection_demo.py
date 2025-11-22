"""
Real-Time Object Detection to Edge - Demo Application
======================================================

A complete, production-ready Python application demonstrating 
real-time object detection using YOLOv8 on a webcam feed.

Features:
- Real-time object detection on webcam
- FPS counter showing performance
- Confidence threshold filtering
- Detection statistics
- Easy model switching
- Graceful error handling

Author: Created for College Freshers Workshop
Date: 2025
"""

import cv2
from ultralytics import YOLO
import time
import sys
from collections import deque

class ObjectDetectionApp:
    """
    Main application class for real-time object detection.
    
    This class encapsulates all functionality for webcam-based
    object detection using YOLOv8.
    """
    
    def __init__(self, model_name='yolov8n.pt', confidence_threshold=0.5, 
                 display_size=(1280, 720), fps_window=30):
        """
        Initialize the object detection application.
        
        Args:
            model_name (str): YOLOv8 model to use
                - 'yolov8n.pt': nano (fastest, 3MB)
                - 'yolov8s.pt': small (balanced, 13MB)
                - 'yolov8m.pt': medium (accurate, 26MB)
                - 'yolov8l.pt': large (most accurate, 52MB)
            
            confidence_threshold (float): Minimum confidence (0-1) to display detection
                - 0.5: Default, balanced (some false positives, all true positives)
                - 0.7: Strict (fewer false positives, might miss some)
                - 0.3: Loose (more detections, some noise)
            
            display_size (tuple): Size to display video (width, height)
            
            fps_window (int): Number of frames for FPS averaging
        """
        
        print("🚀 Initializing Object Detection Application...")
        print(f"📦 Loading model: {model_name}")
        
        try:
            self.model = YOLO(model_name)
            print(f"✅ Model loaded successfully!")
            print(f"   Available classes: {len(self.model.names)} objects")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("   Make sure you have internet for first-time model download")
            sys.exit(1)
        
        self.confidence_threshold = confidence_threshold
        self.display_size = display_size
        
        # FPS calculation
        self.fps_window = deque(maxlen=fps_window)
        self.prev_time = 0
        
        # Statistics tracking
        self.frame_count = 0
        self.total_detections = 0
        self.class_counts = {}
        
        print(f"⚙️  Configuration:")
        print(f"   Confidence Threshold: {confidence_threshold*100:.0f}%")
        print(f"   Display Size: {display_size[0]}x{display_size[1]}")
        print("\n📹 Starting webcam...")
    
    def calculate_fps(self):
        """Calculate frames per second with averaging."""
        current_time = time.time()
        
        if self.prev_time > 0:
            delta = current_time - self.prev_time
            if delta > 0:
                fps = 1 / delta
                self.fps_window.append(fps)
        
        self.prev_time = current_time
        
        # Return average FPS over the window
        return sum(self.fps_window) / len(self.fps_window) if self.fps_window else 0
    
    def add_info_panel(self, frame, fps, detection_count):
        """
        Add information panel to the video frame.
        
        Shows:
        - FPS counter
        - Frame number
        - Number of objects detected
        - Confidence threshold
        """
        
        # Create a semi-transparent panel at the top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Add text information
        text_color = (0, 255, 0)  # Green
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                    font, font_scale, text_color, thickness)
        
        # Frame count
        cv2.putText(frame, f'Frame: {self.frame_count}', (10, 60), 
                    font, font_scale, text_color, thickness)
        
        # Detection count
        cv2.putText(frame, f'Objects: {detection_count}', (10, 90), 
                    font, font_scale, text_color, thickness)
        
        # Confidence threshold
        cv2.putText(frame, f'Conf: {self.confidence_threshold*100:.0f}%', (250, 30), 
                    font, font_scale, text_color, thickness)
        
        return frame
    
    def run(self, camera_index=0):
        """
        Run the real-time object detection application.
        
        Args:
            camera_index (int): Webcam index (0 = default camera)
        
        Keyboard Controls:
            - 'q': Quit application
            - 's': Save current frame
            - 'r': Reset statistics
            - '+': Increase confidence threshold
            - '-': Decrease confidence threshold
        """
        
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam!")
            print("   Make sure your camera is connected and working")
            return
        
        print("✅ Webcam opened successfully!")
        print(f"\n🎬 Press 'q' to quit, 's' to save frame")
        print(f"   '+' to increase confidence, '-' to decrease\n")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error reading frame from camera")
                    break
                
                frame_count += 1
                self.frame_count = frame_count
                
                # Resize for processing (speeds up inference)
                # Note: Original kept for display quality
                display_frame = cv2.resize(frame, self.display_size)
                
                # Run inference
                results = self.model(display_frame, conf=self.confidence_threshold)
                
                # Count detections
                detection_count = 0
                for result in results:
                    detection_count += len(result.boxes)
                    
                    # Count by class
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1
                
                self.total_detections += detection_count
                
                # Draw annotations (YOLO handles this automatically)
                annotated_frame = results[0].plot()
                
                # Calculate and display FPS
                fps = self.calculate_fps()
                
                # Add custom info panel
                annotated_frame = self.add_info_panel(annotated_frame, fps, detection_count)
                
                # Display the frame
                cv2.imshow('Real-Time Object Detection - YOLOv8', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n✋ Quitting application...")
                    break
                
                elif key == ord('s'):
                    filename = f'detection_frame_{frame_count}.jpg'
                    cv2.imwrite(filename, annotated_frame)
                    print(f"📸 Frame saved: {filename}")
                
                elif key == ord('r'):
                    self.total_detections = 0
                    self.class_counts = {}
                    print("🔄 Statistics reset")
                
                elif key == ord('+'):
                    self.confidence_threshold = min(1.0, self.confidence_threshold + 0.1)
                    print(f"✅ Confidence threshold increased to {self.confidence_threshold*100:.0f}%")
                
                elif key == ord('-'):
                    self.confidence_threshold = max(0.0, self.confidence_threshold - 0.1)
                    print(f"✅ Confidence threshold decreased to {self.confidence_threshold*100:.0f}%")
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            self.print_statistics()
    
    def print_statistics(self):
        """Print detection statistics from the session."""
        
        print("\n" + "="*50)
        print("📊 SESSION STATISTICS")
        print("="*50)
        
        print(f"\n📹 Video Statistics:")
        print(f"   Total Frames Processed: {self.frame_count}")
        print(f"   Total Detections: {self.total_detections}")
        
        if self.frame_count > 0:
            avg_detections = self.total_detections / self.frame_count
            print(f"   Avg Objects per Frame: {avg_detections:.2f}")
        
        if self.class_counts:
            print(f"\n🎯 Objects Detected by Class:")
            for class_name, count in sorted(self.class_counts.items(), 
                                           key=lambda x: x[1], reverse=True):
                percentage = (count / self.total_detections * 100) if self.total_detections > 0 else 0
                print(f"   {class_name:15s}: {count:4d} ({percentage:5.1f}%)")
        
        print(f"\n⚙️  Model Configuration:")
        print(f"   Model Name: yolov8n (Nano)")
        print(f"   Model Size: ~3 MB")
        print(f"   Confidence Threshold: {self.confidence_threshold*100:.0f}%")
        print(f"   Display Resolution: {self.display_size[0]}x{self.display_size[1]}")
        
        print("\n" + "="*50)


def main():
    """
    Main entry point for the application.
    
    Creates and runs an ObjectDetectionApp instance with default settings.
    """
    
    print("\n" + "="*60)
    print("🤖 REAL-TIME OBJECT DETECTION TO EDGE - DEMO")
    print("="*60)
    print("\nThis application demonstrates how modern AI models can")
    print("run on edge devices (your laptop, phone, Raspberry Pi)")
    print("with real-time performance!\n")
    
    # Create application instance
    # Note: First run will download the model (~3MB for yolov8n)
    app = ObjectDetectionApp(
        model_name='yolov8n.pt',      # Nano model - fastest for demo
        confidence_threshold=0.5,      # 50% confidence threshold
        display_size=(1280, 720),      # 720p display
        fps_window=30                  # Average FPS over 30 frames
    )
    
    # Run the application
    app.run(camera_index=0)
    
    print("\n✅ Application closed successfully!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
