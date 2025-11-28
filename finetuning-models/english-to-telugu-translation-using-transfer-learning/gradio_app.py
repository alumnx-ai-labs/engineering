import gradio as gr
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Global variables for model and tokenizer
model = None
tokenizer = None
device = None

def load_model():
    """Load the model once when the app starts"""
    global model, tokenizer, device
    
    model_path = "mbart-finetuned-en-te"
    print(f"Loading model from {model_path}...")
    
    try:
        tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
        model = MBartForConditionalGeneration.from_pretrained(model_path)
        
        tokenizer.src_lang = "en_XX"
        tokenizer.tgt_lang = "te_IN"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"Model loaded successfully on {device}!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def translate_text(english_text):
    """Translate English text to Telugu"""
    if not english_text or not english_text.strip():
        return "Please enter some text to translate."
    
    try:
        inputs = tokenizer(
            english_text, 
            return_tensors="pt", 
            max_length=128, 
            truncation=True, 
            padding=True
        ).to(device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id["te_IN"],
                max_length=128,
                num_beams=5,
                early_stopping=True
            )

        translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return translation
    except Exception as e:
        return f"Error during translation: {str(e)}"

def chat_interface(message, history):
    """Handle chat interface logic"""
    if message.strip():
        translation = translate_text(message)
        return translation
    return "Please enter some English text to translate."

# Load model at startup
print("Initializing model...")
if not load_model():
    print("Failed to load model. Please check the model path.")
    exit(1)

# Create Gradio interface
demo = gr.ChatInterface(
    fn=chat_interface,
    title="🇬🇧 ➡️ 🇮🇳 English to Telugu Translator",
    description="Enter English text and get Telugu translation. Powered by fine-tuned mBART model.",
    examples=[
        "What is your name?",
        "Hi",
        "How are you?",
        "Good morning",
        "Thank you very much"
    ]
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # Makes it accessible from other devices on your network
        server_port=7860,
        share=False  # Set to True if you want a public link
    )