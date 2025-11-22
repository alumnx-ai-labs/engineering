# Real-Time Object Detection to Edge - Setup & Usage Guide

## Quick Start (5 minutes)

### 1. Install Dependencies

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install ultralytics opencv-python numpy
```

### 2. Run the Demo

```bash
python object_detection_demo.py
```

### 3. That's it!
- Your webcam will open in a window
- Bounding boxes will appear around detected objects
- Press 'q' to quit

---

## What You'll See

When you run the application:

1. **Model Loading (2-3 seconds)**
   - First run: Downloads YOLOv8 nano model (~3MB)
   - Subsequent runs: Loads from cache (instant)

2. **Webcam Feed**
   - Live video from your camera
   - Bounding boxes around detected objects
   - Class labels (e.g., "person", "cat", "car")
   - Confidence scores (e.g., "0.95" = 95% confident)

3. **Info Panel (top left)**
   - FPS: Frames processed per second (target: 20-30)
   - Frame: Current frame number
   - Objects: Number of objects detected in this frame
   - Conf: Current confidence threshold

4. **Real-time Processing**
   - No internet required after first download
   - Works completely locally on your device
   - This is "edge computing" in action!

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| **q** | Quit application |
| **s** | Save current frame as JPEG |
| **r** | Reset detection statistics |
| **+** | Increase confidence threshold (stricter) |
| **-** | Decrease confidence threshold (looser) |

### Understanding Confidence Threshold

- **Lower (0.3-0.4)**: Detects more objects, some false positives
  - Use when: You don't want to miss objects
  - Example: Security surveillance

- **Default (0.5)**: Balanced approach, good for general use
  - Use when: You want good balance
  - Example: Demo, general applications

- **Higher (0.7-0.8)**: Detects fewer objects, very confident
  - Use when: You want only strong detections
  - Example: Safety-critical applications

---

## Model Options

The demo uses `yolov8n.pt` (nano) by default. You can switch models:

```python
# In object_detection_demo.py, change the model_name parameter:

app = ObjectDetectionApp(
    model_name='yolov8s.pt'  # Change here
)
```

### Available Models

| Model | Size | Speed | Accuracy | Best For |
|-------|------|-------|----------|----------|
| **yolov8n.pt** | 3 MB | ⚡⚡⚡⚡ Very Fast | ⭐⭐ | **Laptops, Phones** |
| yolov8s.pt | 13 MB | ⚡⚡⚡ Fast | ⭐⭐⭐ | Balanced |
| yolov8m.pt | 26 MB | ⚡⚡ Medium | ⭐⭐⭐⭐ | High Accuracy |
| yolov8l.pt | 52 MB | ⚡ Slow | ⭐⭐⭐⭐⭐ | Maximum Accuracy |
| yolov8x.pt | 133 MB | 🐢 Very Slow | ⭐⭐⭐⭐⭐ | Cloud Only |

**For this demo: Stick with `yolov8n.pt`** (nano)
- Fastest on CPU
- Still accurate for most objects
- Quick demo responsiveness

---

## Expected Performance

### On Laptop CPU (Intel i5/i7 or AMD equivalent)
- **FPS**: 15-35 frames per second
- **Latency**: 30-65 milliseconds per frame
- **Model Load Time**: 2-3 seconds

### What Affects Performance?
1. **CPU Speed**: Faster processor = more FPS
2. **Image Resolution**: Higher resolution = slower
3. **Number of Objects**: More objects = slightly slower
4. **Model Size**: Larger model = slower inference
5. **Background Processes**: Close unnecessary programs

### Real-World Comparison
```
Your Laptop (CPU):     20-30 FPS  ← This demo
Desktop with GPU:      50-100 FPS
Smartphone GPU:        10-30 FPS
Raspberry Pi 4:        5-15 FPS
Cloud Server:          100+ FPS (but 200-500ms latency)
```

---

## Customization Examples

### Example 1: Use a More Accurate Model

```python
# Requires download of larger model, uses more CPU
app = ObjectDetectionApp(
    model_name='yolov8m.pt',  # Medium model
    confidence_threshold=0.6
)
app.run()
```

### Example 2: Stricter Confidence (Security Camera)

```python
# Only show very confident detections
app = ObjectDetectionApp(
    model_name='yolov8n.pt',
    confidence_threshold=0.75  # 75% confidence
)
app.run()
```

### Example 3: Custom Display Size

```python
# For projector or large screen
app = ObjectDetectionApp(
    model_name='yolov8n.pt',
    display_size=(1920, 1080)  # Full HD
)
app.run()
```

### Example 4: Different Camera

```python
# If you have multiple cameras
app = ObjectDetectionApp()
app.run(camera_index=1)  # Use second camera (0=first, 1=second)
```

---

## Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'ultralytics'"

**Solution:**
```bash
pip install ultralytics
```

**If that doesn't work:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

pip install ultralytics opencv-python numpy
```

---

### ❌ "Could not open webcam" or "Webcam not found"

**Possible causes:**
1. No camera connected
2. Camera in use by another application
3. Wrong camera index

**Solutions:**
```bash
# Check available cameras (Linux/macOS)
ls /dev/video*

# Try different camera indices
# In Python: app.run(camera_index=0), app.run(camera_index=1), etc.
```

**On Windows:**
- Close Zoom, Teams, Discord, OBS if running
- Restart the application
- Check Device Manager for camera status

---

### ❌ "Model downloading failed" or "No internet"

**Cause:** First run needs to download model

**Solution:**
```bash
# Download model manually
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# This will download and cache the model
# Subsequent runs won't need internet
```

**If no internet available:**
- Download on another machine
- Transfer model file to: `~/.cache/ultralytics/` directory
- Or: Use offline installation package

---

### ❌ "Low FPS" or "Application is laggy"

**Causes & Solutions:**

1. **Wrong model selected**
   ```python
   # Change to nano model (fastest)
   model_name='yolov8n.pt'
   ```

2. **Display size too large**
   ```python
   # Reduce resolution
   display_size=(640, 480)  # Instead of (1280, 720)
   ```

3. **High confidence threshold causing delay**
   - This shouldn't cause latency, but try:
   ```python
   confidence_threshold=0.5
   ```

4. **Other programs using CPU**
   - Close: Chrome, VS Code, Discord, streaming apps
   - Use: Task Manager (Windows) or Activity Monitor (macOS)

5. **Older laptop with slow CPU**
   - Expected: 10-15 FPS is normal
   - Consider: Using Raspberry Pi with GPU acceleration

---

### ❌ "Camera image is upside down or rotated"

**Solution:**
```python
# Add this in the run() method before inference
frame = cv2.rotate(frame, cv2.ROTATE_180)  # Or other rotation options
```

---

### ❌ "Detections are inaccurate" or "Missing objects"

**Why this happens:**
- Lighting conditions (dark, backlighting)
- Object partially hidden/occluded
- Small objects at distance
- Object not in training data

**Solutions:**

1. **Improve lighting**
   - Move closer to light source
   - Adjust room lighting

2. **Lower confidence threshold**
   ```python
   confidence_threshold=0.3  # More detections
   ```

3. **Use larger model (more accurate)**
   ```python
   model_name='yolov8s.pt'  # Or yolov8m.pt
   ```

4. **Get object closer to camera**
   - Small/distant objects are harder to detect

5. **Remember: This is not perfect!**
   - Even humans miss things sometimes
   - ML models are probabilistic, not deterministic

---

## Performance Metrics to Track

When you run the demo, you'll see:

1. **FPS (Frames Per Second)**
   - Shown in top-left corner
   - Higher = better processing speed
   - 20+ FPS = good for real-time

2. **Objects per Frame**
   - Average number of detected objects
   - Increases as scene becomes complex

3. **Session Statistics**
   - Printed after you quit (press 'q')
   - Shows total detections by class
   - Useful for analyzing session

### What To Look For

```
Expected output after 30 seconds:
- Total Frames: ~600 (at 20 FPS)
- Total Detections: ~1000-3000 (depending on what's visible)
- Average Objects: 2-5 per frame
- Most Common: person (if human in frame)
```

---

## Advanced: Running on Raspberry Pi

If you have a Raspberry Pi:

```bash
# Install on Raspberry Pi OS
sudo apt-get install python3-pip python3-venv
python3 -m venv venv
source venv/bin/activate

# Install with GPU support (optional, if you have Pi with GPU)
pip install ultralytics opencv-python

# Run
python3 object_detection_demo.py
```

**Performance on Raspberry Pi 4:**
- **CPU only**: 5-10 FPS
- **With GPU**: 15-25 FPS (requires GPU acceleration setup)

---

## Advanced: Recording Detections

To save video with detections:

```python
# Add this in the run() method:
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('detection_output.mp4', fourcc, 30, self.display_size)

# In the loop after drawing annotations:
out.write(annotated_frame)

# In finally block:
out.release()
```

---

## Common Classes Detected

The model is trained on COCO dataset with 80 classes. Common ones:

**People:**
- person

**Animals:**
- dog, cat, bird, horse, cow, elephant, bear, zebra

**Vehicles:**
- car, truck, bus, motorcycle, bicycle, train, airplane

**Objects:**
- bottle, cup, backpack, laptop, phone, keyboard, mouse

**Furniture:**
- chair, table, bed, couch, desk

Full list available at:
https://github.com/ultralytics/yolov8/blob/main/ultralytics/cfg/datasets/coco.yaml

---

## Safety & Privacy Notes

✅ **This application is private:**
- All processing happens locally on your device
- No data sent to cloud or external servers
- No images stored (unless you press 's' to save)
- Works completely offline after first model download

⚠️ **Use responsibly:**
- Don't record others without consent
- Be aware of privacy laws in your region
- Consider where you deploy this technology

---

## Next Steps: What to Try

1. **Modify the code:**
   - Add object counting
   - Filter for specific classes
   - Log detections to CSV

2. **Use your own images:**
   - Process image files instead of webcam
   - Detect in video files

3. **Train your own model:**
   - Collect custom dataset
   - Fine-tune YOLOv8 on your data
   - Detect custom objects

4. **Deploy to mobile:**
   - Convert to TensorFlow Lite
   - Build iOS/Android app
   - Use GitHub: pjreddie/darknet

5. **Explore other models:**
   - Pose estimation (detect body joints)
   - Segmentation (pixel-level detection)
   - Tracking (follow objects across frames)

---

## Resources

**Official Documentation:**
- YOLOv8: https://docs.ultralytics.com
- OpenCV: https://docs.opencv.org
- PyTorch: https://pytorch.org/docs

**GitHub Repositories:**
- YOLOv8: https://github.com/ultralytics/yolov8
- OpenCV: https://github.com/opencv/opencv

**Learning Resources:**
- Fast.ai Practical Deep Learning: https://course.fast.ai
- TensorFlow Tutorials: https://www.tensorflow.org/tutorials
- Roboflow Blog: https://blog.roboflow.com

**Community:**
- YOLOv8 Discussions: https://github.com/ultralytics/yolov8/discussions
- Stack Overflow: Tag `yolo` and `opencv`
- Reddit: r/computervision, r/MachineLearning

---

## Support & Questions

If you encounter issues:

1. **Check the Troubleshooting section above**
2. **Search GitHub issues:** https://github.com/ultralytics/yolov8/issues
3. **Visit Stack Overflow** with tag `yolo`
4. **Read the documentation** in code comments

---

## License & Attribution

**YOLOv8:** GPL-3.0 License (https://github.com/ultralytics/yolov8)

**OpenCV:** Apache 2.0 License

**This Demo Code:** Free to use and modify for educational purposes

---

## Conclusion

You now have everything needed to:
- ✅ Understand object detection concepts
- ✅ Run real-time detection on your device
- ✅ Modify and experiment with the code
- ✅ Deploy to different platforms
- ✅ Build your own applications

**Remember:** Edge computing is the future. Understanding how to build efficient AI systems that run locally is increasingly important in industry.

Happy coding! 🚀
