# Quick Reference Card - Session at a Glance

## 📋 Pre-Session Checklist (30 min before)

- [ ] Run: `python object_detection_demo.py` (test once)
- [ ] Close: Zoom, Teams, Discord, VS Code (frees camera)
- [ ] Check: Projector/display resolution
- [ ] Have: Backup video ready
- [ ] Test: Your microphone
- [ ] Know: Where the '+' and '-' keys are (confidence adjustment)

## ⏱️ 60-Minute Session Timeline

```
0:00-5:00   Hook & Questions (Slides 1-2)
5:00-17:00  Core Concepts (Slides 3-12)
           • What is detection? (2 min)
           • Why edge? (3 min)
           • ML basics (2 min)
           • CNNs (3 min)
           • Models (1 min)
           • Pipeline (1 min)
           • Optimization (1 min)
           • Apps (2 min)
           • Challenges (1 min)

17:00-35:00 LIVE DEMO (15-18 min)
           • Start application
           • Show detection on multiple objects
           • Show +/- controls
           • Point out FPS counter
           • Demonstrate real-time processing

35:00-45:00 Code & Performance (Slides 14-18, 9 min)
           • Setup code (2 min)
           • Main loop (2 min)
           • Advanced features (2 min)
           • Performance metrics (2 min)
           • Deployment (1 min)

45:00-60:00 Summary & Q&A (Slides 19-20, 5 min)
           • Key takeaways (2 min)
           • Resources (1 min)
           • Questions (2 min)
```

## 🎮 Demo Controls

| Key | Result | Demo Timing |
|-----|--------|-------------|
| **q** | Quit | ~35:00 (after ~15-18 min) |
| **s** | Save frame | Show ~25:00 |
| **r** | Reset stats | Show ~28:00 |
| **+** | Higher confidence | Show ~30:00 |
| **-** | Lower confidence | Show ~32:00 |

## 📊 Key Statistics to Mention

- **FPS:** 20-35 on CPU (watch the live counter)
- **Latency:** 30-50 milliseconds per frame
- **Model size:** 3 MB (fits on phone)
- **Classes:** 80 different objects detected
- **Training:** Trained on 1.2M images
- **Speed:** No internet needed (edge advantage!)

## 💬 Key Phrases to Use

### Opening
> "Today you'll understand technology in your phone, cars, and security cameras"

### During Hook
> "How does your phone recognize your face in 0.1 seconds?"

### Introducing Concept
> "Object detection isn't just saying 'there's a dog'—it's saying where in the image"

### Before Demo
> "Here's where it gets real. This is actual code working right now"

### During Demo
> "Watch the FPS counter—this is running on my laptop CPU, no cloud"

### After Demo
> "15 lines of Python. That's all this takes with modern frameworks"

### Closing
> "You understand the concepts AND can build it. Go experiment"

## 🔴 If Something Goes Wrong

| Problem | Quick Fix |
|---------|-----------|
| Webcam won't open | Close other apps, restart demo |
| Very low FPS | It's okay, explain system load |
| Model won't download | Show backup video |
| Confidence control freezes | Kill and restart, continue |
| Forgot a key point | "Great question, let me research that" |

**Golden Rule:** Stay calm. Live demos are risky. Audiences expect hiccups.

## 📌 Slide Quick Summaries

| Slide | Title | Key Message | Time |
|-------|-------|-------------|------|
| 1 | Title | You'll build AI that runs on devices | 1 min |
| 2 | Hook | How does your phone recognize you? | 2 min |
| 3 | What is Detection | WHERE, not just WHAT | 2 min |
| 4 | Why Edge | Speed + Privacy + Reliability | 3 min |
| 5 | ML Basics | Pattern matching from examples | 2 min |
| 6 | CNNs | Like scanning with a magnifying glass | 3 min |
| 7 | Models | YOLO is fast, MobileNet is tiny | 1 min |
| 8 | Pipeline | Preprocessing → Inference → Post-processing | 1 min |
| 9 | Optimization | 4x smaller, 2x faster, keep accuracy | 1 min |
| 10 | Frameworks | TensorFlow Lite, ONNX, we use ONNX | 1 min |
| 11 | Applications | Phones, cars, security, robots, health | 2 min |
| 12 | Challenges | Accuracy trade-offs, diverse environments | 1 min |
| 13 | Demo Overview | What you're about to see | 1 min |
| 14-16 | Code | 15 lines, super simple | 5 min |
| 17 | Performance | 30-50ms latency, 20-30 FPS on CPU | 2 min |
| 18 | Deployment | Phone, web, IoT, server—different but same | 2 min |
| 19 | Takeaways | Concepts + Code + Inspiration | 2 min |
| 20 | Resources | GitHub, docs, tutorials, communities | 1 min |

## 🎯 Student Learning Goals

By end of workshop, students can:

✅ Explain object detection (what and where)
✅ Understand edge computing (local processing > cloud)
✅ Visualize how CNNs work (layers learn features)
✅ Name real applications (phones, cars, security)
✅ Code real-time detection (15 lines!)
✅ Deploy to devices (phone, Raspberry Pi)

## 💻 Installation Commands (For Distribution)

```bash
# Quick install
pip install -r requirements.txt

# Or manually
pip install ultralytics opencv-python numpy

# Run simple version
python minimal_demo.py

# Run full version
python object_detection_demo.py
```

## 📚 Resources to Share

- **YOLOv8 Docs:** https://docs.ultralytics.com
- **GitHub:** https://github.com/ultralytics/yolov8
- **TensorFlow Lite:** https://tensorflow.org/lite
- **OpenCV:** https://opencv.org
- **Fast.ai:** https://course.fast.ai

## 🎬 Demo Talking Points

### Startup Phase (2-3 sec)
"Loading the model... this is one-time. After this, it's cached."

### Detecting Objects (5-10 min)
"Notice the bounding boxes? That's the network identifying where objects are."

### Multiple Objects (2-3 min)
"It detects several at once. See? Person, cup, laptop—all simultaneously."

### Confidence Scores (1-2 min)
"That number is how confident the network is. 0.95 = 95% sure."

### FPS Counter (1-2 min)
"Watch that FPS number—30 frames per second on my laptop CPU. No cloud involved."

### Adjusting Threshold (2-3 min)
"With +, we get stricter. With -, we get looser. Trade-off between recall and precision."

### Performance (1-2 min)
"Cloud would take 200-500ms. We did it in 30-50ms. That's the edge advantage."

## 🎁 What to Give Students

After session:
- [ ] All files (GitHub link or download)
- [ ] Slides PDF
- [ ] Code files (both versions)
- [ ] Setup guide
- [ ] Resource list

## ✨ Pro Tips

1. **Narrate the demo:** Don't let silence happen
2. **Make eye contact:** Even during demo
3. **Have backup video:** Just in case
4. **Manage energy:** Change medium every 5 min
5. **Ask questions:** "Why do you think...?"
6. **Be honest:** "I don't know, but let's figure it out"
7. **Celebrate:** "You now understand AI"

## 🚨 Timing Hacks

- **Ahead of schedule?** Show code variations, discuss fine-tuning
- **Behind schedule?** Skip some deployment details, keep demo and code
- **Way behind?** Reduce Q&A, but keep core message

## 🎓 After Session

- [ ] Send files to students within 24 hours
- [ ] Answer follow-up emails
- [ ] Offer office hours
- [ ] Ask for feedback
- [ ] Reflect on what worked

## 📞 Emergency Contacts (For Info)

- **YOLOv8 Questions:** https://github.com/ultralytics/yolov8/issues
- **OpenCV Questions:** Stack Overflow tag `opencv`
- **Deployment Help:** TensorFlow Lite docs
- **General ML:** Fast.ai forums

---

**You're ready! Print this out and keep it handy during the session.** 🚀
