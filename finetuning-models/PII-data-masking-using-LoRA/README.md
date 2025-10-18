# 🦥 Fine-Tuning Gemma 3 270M for PII Masking

A powerful and efficient solution for masking Personally Identifiable Information (PII) in text using fine-tuned Gemma 3 270M model with QLoRA and Unsloth.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/pii-masking-gemma3/blob/main/PII_Masking_Gemma_3_270M.ipynb)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Notebook Structure](#notebook-structure)
- [Configuration](#configuration)
- [Training Results](#training-results)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This Jupyter notebook fine-tunes the Gemma 3 270M instruction-tuned model to automatically detect and mask PII in text. The model learns to replace sensitive information like names, emails, phone numbers, and addresses with appropriate placeholder tags (e.g., `[NAME]`, `[EMAIL]`, `[PHONE]`).

**Key Highlights:**
- ✅ Runs on free Google Colab T4 GPU
- ✅ Uses efficient QLoRA (4-bit quantization) for low memory usage
- ✅ Powered by Unsloth for 2x faster training
- ✅ Trained on 400K+ PII examples from `ai4privacy/pii-masking-400k`
- ✅ Includes pre/post evaluation and loss visualization
- ✅ Interactive notebook format with detailed explanations

## ✨ Features

- **Easy to Use**: Just click "Run All" in Google Colab
- **Efficient Training**: QLoRA with 4-bit quantization reduces memory footprint
- **Fast Fine-Tuning**: Unsloth optimization for faster training
- **Comprehensive Evaluation**: Pre and post-training model comparison
- **Loss Visualization**: Training and evaluation loss plots with matplotlib
- **Interactive Testing**: Test on custom examples within the notebook
- **Easy Deployment**: Save LoRA adapters for quick inference
- **Multiple Test Cases**: Evaluate on both training and validation data

## 📦 Requirements

### Hardware
- **Minimum**: GPU with 8GB VRAM (e.g., Google Colab T4) ✅ **FREE**
- **Recommended**: GPU with 16GB+ VRAM for faster training

### Software (Auto-installed in Colab)
- Python 3.8+
- CUDA-compatible GPU
- All dependencies installed automatically in the notebook

## 🚀 Quick Start

### Option 1: Google Colab (Recommended - No Setup Required!)

1. **Open the notebook**
   - Click the "Open in Colab" badge above, or
   - Go to [Google Colab](https://colab.research.google.com/)
   - File → Open notebook → GitHub tab
   - Enter: `yourusername/pii-masking-gemma3`
   - Select `PII_Masking_Gemma_3_270M.ipynb`

2. **Enable GPU**
   - Runtime → Change runtime type
   - Hardware accelerator: **T4 GPU**
   - Click Save

3. **Run the notebook**
   - Click Runtime → Run all (or press Ctrl+F9)
   - The notebook will automatically:
     - Install all dependencies
     - Load the model
     - Train on PII data
     - Show results and visualizations

4. **Wait for training to complete** (~45-60 minutes)

5. **Download your fine-tuned model**
   - After training, the LoRA adapters are saved to `gemma-3-pii-masker/`
   - Download: Click the folder icon → Right-click folder → Download

### Option 2: Local Jupyter Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pii-masking-gemma3.git
cd pii-masking-gemma3
```

2. **Install Jupyter and dependencies**
```bash
pip install jupyter
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Launch Jupyter**
```bash
jupyter notebook PII_Masking_Gemma_3_270M.ipynb
```

4. **Run all cells**
   - Kernel → Restart & Run All

### Option 3: Kaggle Notebooks

1. Upload the notebook to Kaggle
2. Settings → Accelerator → GPU T4 x2
3. Run all cells

## 🔧 How It Works

### Notebook Workflow

```
Install Dependencies → Load Model → Prepare Data → Pre-Eval → Train → Visualize → Post-Eval → Save
```

### 1. Model Architecture

The notebook uses **Gemma 3 270M** (instruction-tuned variant) with:
- **QLoRA**: 4-bit quantization for memory efficiency
- **LoRA Adapters**: 
  - Rank (r): 128
  - Alpha: 128
  - Target modules: All attention and MLP layers
  - Trainable parameters: ~0.5% of base model

### 2. Data Processing

The `ai4privacy/pii-masking-400k` dataset contains:
- **source_text**: Original text with PII
- **masked_text**: Text with PII replaced by tags

Example transformation:
```
Input:  "My name is Alice Smith, call me at 555-1234"
Output: "My name is [NAME], call me at [PHONE]"
```

### 3. Training Process

**Training Configuration:**
- Dataset: 50K training samples, 10K validation samples
- Batch size: 8
- Epochs: 2
- Learning rate: 5e-5
- Optimizer: AdamW 8-bit
- Warmup steps: 5
- Evaluation: Every 1000 steps
- Total training steps: ~12,500

### 4. Evaluation

The notebook provides:
- **Pre-training evaluation**: Tests base model before fine-tuning
- **Post-training evaluation**: Tests fine-tuned model on same examples
- **Loss plots**: Visual comparison of training vs evaluation loss
- **Multiple test cases**: Validation data and training data samples

## 📓 Notebook Structure

The notebook is organized into clear sections:

### Section 1: Installation and Setup
- Automatic detection of Colab environment
- Installation of Unsloth and dependencies
- Version-specific package installation

### Section 2: Load Model and Add LoRA Adapters
- Load Gemma 3 270M in 4-bit precision
- Configure LoRA parameters
- Set up Gemma 3 chat template

### Section 3: Data Preparation
- Load PII masking dataset
- Split into train/test sets
- Format data for instruction fine-tuning
- Apply Gemma 3 chat template

### Section 4: Pre-Fine-Tuning Evaluation
- Test base model capabilities
- Display original input and expected output
- Generate base model response

### Section 5: Train the Model
- Configure SFTTrainer with evaluation
- Set up response-only training
- Execute training loop
- Display memory statistics

### Section 6: Visualize Training
- Extract loss history
- Plot training and evaluation loss
- Analyze training progress

### Section 7: Post-Fine-Tuning Evaluation
- Test fine-tuned model on validation data
- Test on custom examples
- Test on training data samples
- Compare before/after results

### Section 8: Save LoRA Adapters
- Save model locally
- Instructions for pushing to Hugging Face

## ⚙️ Configuration

### Modify Training Parameters

Find and edit these cells in the notebook:

**Model Settings (Section 2)**
```python
max_seq_length = 2048  # Maximum sequence length
model_name = "unsloth/gemma-3-270m-it"

# LoRA settings
r = 128  # LoRA rank (higher = more expressive, more memory)
lora_alpha = 128  # Scaling factor
```

**Dataset Size (Section 3)**
```python
# Use more/less data for training
dataset = load_dataset("ai4privacy/pii-masking-400k", split="train[:60000]")

# For full dataset (longer training):
# dataset = load_dataset("ai4privacy/pii-masking-400k", split="train")

# For quick testing (faster training):
# dataset = load_dataset("ai4privacy/pii-masking-400k", split="train[:10000]")
```

**Training Settings (Section 5)**
```python
num_epochs = 2  # Number of training epochs
per_device_train_batch_size = 8  # Batch size
learning_rate = 5e-5  # Learning rate
eval_steps = 1000  # Evaluate every N steps
logging_steps = 100  # Log metrics every N steps
```

### Test Your Own Examples

In **Section 7**, modify this cell to test custom text:

```python
messages = [
    {"role": "user", "content": f"Mask all Personally Identifiable Information (PII) in the following text with their respective placeholder tags. Only return the masked text.\n\nTEXT:\n Your custom text here"},
]
```

## 📊 Training Results

### Expected Performance

After 2 epochs on 50K examples:
- **Training Time**: ~45-60 minutes on T4 GPU
- **Memory Usage**: ~7-8 GB VRAM
- **Final Training Loss**: ~0.3-0.5
- **Final Eval Loss**: ~0.4-0.6
- **Training Speed**: ~200-250 steps per minute

### Output Visualizations

The notebook generates:
1. **Loss Plot**: Line chart showing training and evaluation loss over time
2. **Memory Statistics**: GPU usage breakdown
3. **Before/After Comparison**: Side-by-side model outputs

### What Good Training Looks Like

✅ **Good signs:**
- Both training and eval loss decrease steadily
- Eval loss stays close to training loss
- Model outputs proper PII tags in post-evaluation

⚠️ **Warning signs:**
- Training loss decreases but eval loss increases (overfitting)
- Loss plateaus early (learning rate too low)
- Loss fluctuates wildly (learning rate too high)

## 💡 Usage Examples

### Example 1: Basic PII Masking

```
Input:  "Contact Jane Doe at jane.doe@company.com or 555-0123"
Output: "Contact [NAME] at [EMAIL] or [PHONE]"
```

### Example 2: Multiple PII Types

```
Input:  "Dr. Smith lives at 123 Main St, works at Acme Corp, SSN: 123-45-6789"
Output: "Dr. [NAME] lives at [ADDRESS], works at [ORGANIZATION], SSN: [SSN]"
```

### Example 3: Mixed Context

```
Input:  "The meeting with CEO John at headquarters (john@corp.com) is confirmed"
Output: "The meeting with CEO [NAME] at headquarters ([EMAIL]) is confirmed"
```

### Testing in the Notebook

Add a new cell after Section 7 and test your own examples:

```python
# Test custom text
custom_text = "My name is Paul Atredis, call me at +919694565231"

messages = [
    {"role": "user", "content": f"Mask all PII in: {custom_text}"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).removeprefix('<bos>')

_ = model.generate(
    **tokenizer(text, return_tensors="pt").to("cuda"),
    max_new_tokens=125,
    temperature=0.5,
    top_p=0.95,
    top_k=64,
    streamer=TextStreamer(tokenizer, skip_prompt=True),
)
```

## 🐛 Troubleshooting

### Issue: "Runtime disconnected" in Colab

**Cause**: Colab free tier has usage limits

**Solutions:**
1. Reconnect and resume: Runtime → Reconnect
2. Reduce training time: Use fewer examples or epochs
3. Save checkpoints: Add checkpoint saving every 1000 steps
4. Use Colab Pro for longer sessions

### Issue: CUDA Out of Memory

**Solution 1**: Reduce batch size (edit Section 5)
```python
per_device_train_batch_size = 4  # Instead of 8
gradient_accumulation_steps = 2  # Maintain effective batch size
```

**Solution 2**: Reduce sequence length (edit Section 2)
```python
max_seq_length = 1024  # Instead of 2048
```

**Solution 3**: Use smaller dataset (edit Section 3)
```python
dataset = load_dataset("ai4privacy/pii-masking-400k", split="train[:30000]")
```

**Solution 4**: Reduce LoRA rank (edit Section 2)
```python
r = 64  # Instead of 128
lora_alpha = 64
```

### Issue: Notebook cells fail with import errors

**Solution**: Restart runtime and run from the beginning
1. Runtime → Restart runtime
2. Runtime → Run all
3. Wait for installation to complete (~2-3 minutes)

### Issue: "Cannot find module 'unsloth'"

**Cause**: Installation cell didn't complete

**Solution**: 
1. Manually run the installation cell (Section 1)
2. Wait for it to finish (check for green checkmark)
3. Restart runtime: Runtime → Restart runtime
4. Continue from Section 2

### Issue: Dataset download is very slow

**Solution 1**: Use smaller subset initially
```python
dataset = load_dataset("ai4privacy/pii-masking-400k", split="train[:10000]")
```

**Solution 2**: Check Colab internet connection
```python
!ping -c 4 huggingface.co
```

### Issue: Training is very slow

**Solution 1**: Verify GPU is enabled
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
# Should show: Tesla T4 or similar GPU
```

**Solution 2**: Change runtime type
- Runtime → Change runtime type
- Ensure "T4 GPU" is selected

### Issue: Poor model performance after training

**Diagnosis**: Run these checks in a new cell:
```python
# Check if model was actually trained
print(f"Training completed: {trainer_stats.metrics}")
print(f"Final loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")
```

**Solutions:**
1. **Increase training**: Change `num_epochs = 3` or `4`
2. **More data**: Use full dataset instead of subset
3. **Adjust learning rate**: Try `learning_rate = 2e-5`
4. **Check evaluation**: Ensure eval_loss is decreasing

### Issue: Plots not displaying

**Solution**: Install matplotlib
```python
!pip install matplotlib seaborn
import matplotlib.pyplot as plt
plt.figure()
plt.plot([1,2,3])
plt.show()
```

### Issue: Can't save model to Google Drive

**Solution**: Mount Google Drive first
```python
from google.colab import drive
drive.mount('/content/drive')

# Save to Drive
model.save_pretrained("/content/drive/MyDrive/gemma-3-pii-masker")
tokenizer.save_pretrained("/content/drive/MyDrive/gemma-3-pii-masker")
```

## 🚀 Performance Tips

### 1. Speed Optimization

**Use larger batch size if you have more GPU memory:**
```python
per_device_train_batch_size = 16  # If using A100 or V100
```

**Reduce logging for faster training:**
```python
logging_steps = 500  # Instead of 100
eval_steps = 2000  # Instead of 1000
```

**Skip evaluation during training:**
```python
eval_strategy = "no"  # Only evaluate at the end
```

### 2. Quality Optimization

**Train longer:**
```python
num_epochs = 4  # More epochs
dataset = load_dataset("ai4privacy/pii-masking-400k", split="train")  # Full dataset
```

**Lower learning rate for stability:**
```python
learning_rate = 2e-5  # More conservative
warmup_steps = 100  # More warmup
```

**Adjust inference parameters:**
```python
temperature = 0.3  # Lower = more deterministic
top_p = 0.9  # More focused sampling
```

### 3. Memory Optimization

**Already optimized by default!**
- 4-bit quantization: `load_in_4bit=True` ✅
- Gradient checkpointing: Enabled via Unsloth ✅
- Response-only training: Reduces memory usage ✅

**If still having issues:**
```python
r = 32  # Very low rank
max_seq_length = 512  # Very short sequences
per_device_train_batch_size = 2  # Very small batch
gradient_accumulation_steps = 4  # Maintain effective batch size of 8
```

## 📤 Saving and Sharing Your Model

### Save to Google Drive

Add this cell after training:
```python
from google.colab import drive
drive.mount('/content/drive')

# Save to your Drive
model.save_pretrained("/content/drive/MyDrive/gemma-3-pii-masker")
tokenizer.save_pretrained("/content/drive/MyDrive/gemma-3-pii-masker")
print("Model saved to Google Drive!")
```

### Download Locally from Colab

```python
# Create a zip file
!zip -r gemma-3-pii-masker.zip gemma-3-pii-masker/

# Download via Colab's file browser
from google.colab import files
files.download('gemma-3-pii-masker.zip')
```

### Push to Hugging Face Hub

Add your token and run:
```python
from huggingface_hub import login
login()  # Enter your token when prompted

model.push_to_hub("your_username/gemma-3-pii-masker")
tokenizer.push_to_hub("your_username/gemma-3-pii-masker")
```

### Load Your Saved Model

Create a new notebook or cell:
```python
from unsloth import FastModel

# Load from local folder
model, tokenizer = FastModel.from_pretrained(
    model_name="gemma-3-pii-masker",  # Local folder
    max_seq_length=2048,
    load_in_4bit=True,
)

# Or load from Hugging Face
model, tokenizer = FastModel.from_pretrained(
    model_name="your_username/gemma-3-pii-masker",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

## 🎓 Learning Resources

### Understanding the Code

- **Unsloth**: [Documentation](https://github.com/unslothai/unsloth)
- **QLoRA**: [Paper](https://arxiv.org/abs/2305.14314)
- **LoRA**: [Paper](https://arxiv.org/abs/2106.09685)
- **Gemma Models**: [Google AI](https://ai.google.dev/gemma)

### Improving Your Model

- **Prompt Engineering**: Better instructions = better results
- **Hyperparameter Tuning**: Experiment with learning rates, batch sizes
- **Data Quality**: Clean, diverse training data is key
- **Evaluation**: Create test sets that match your use case

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/amazing-feature`
3. Make your changes to the notebook
4. Test thoroughly in Colab
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

**Ideas for contributions:**
- Add more evaluation metrics
- Create inference-only notebook
- Add support for other languages
- Improve visualization
- Add more test cases

## 📝 Citation

If you use this notebook in your research or project:

```bibtex
@software{pii_masking_gemma3,
  title={Fine-Tuning Gemma 3 270M for PII Masking},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/pii-masking-gemma3}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for fast fine-tuning capabilities
- [ai4privacy](https://huggingface.co/ai4privacy) for the excellent PII masking dataset
- [Google](https://ai.google.dev/gemma) for the Gemma model family
- [Hugging Face](https://huggingface.co/) for the transformers library and model hub
- [Google Colab](https://colab.research.google.com/) for free GPU access

## 📧 Support

- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Email**: [your email]

## 🌟 Star History

If you find this notebook helpful, please star the repository!

---

**Made with ❤️ for the privacy-conscious AI community**

**⭐ Star this repo | 🔀 Fork and customize | 🐛 Report issues | 💡 Suggest features**