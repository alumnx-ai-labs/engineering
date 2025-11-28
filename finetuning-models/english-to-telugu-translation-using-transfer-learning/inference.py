import argparse
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import os

def load_model(model_path):
    """
    Loads the model and tokenizer from the specified path.
    """
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
        model = MBartForConditionalGeneration.from_pretrained(model_path)
        
        # Set languages
        tokenizer.src_lang = "en_XX"
        tokenizer.tgt_lang = "te_IN"
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"Model loaded successfully on {device}!")
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

def translate(text, model, tokenizer, device, max_len=128):
    """
    Translates English text to Telugu.
    """
    if not text:
        return ""
        
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        max_length=max_len, 
        truncation=True, 
        padding=True
    ).to(device)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["te_IN"],
            max_length=max_len,
            num_beams=5,
            early_stopping=True
        )

    translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translation

def main():
    parser = argparse.ArgumentParser(description="English to Telugu Translation Inference")
    parser.add_argument("--model_dir", type=str, default="mbart-finetuned-en-te", help="Path to the fine-tuned model directory")
    parser.add_argument("--text", type=str, help="Text to translate (if not provided, runs in interactive mode)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' not found.")
        return

    model, tokenizer, device = load_model(args.model_dir)
    if model is None:
        return

    if args.text:
        # Single translation mode
        translation = translate(args.text, model, tokenizer, device)
        print(f"\nEnglish: {args.text}")
        print(f"Telugu: {translation}")
    else:
        # Interactive mode
        print("\n--- Interactive Translation Mode (Type 'q' to quit) ---")
        while True:
            text = input("\nEnter English text: ").strip()
            if text.lower() in ['q', 'quit', 'exit']:
                break
            if not text:
                continue
                
            translation = translate(text, model, tokenizer, device)
            print(f"Telugu: {translation}")

if __name__ == "__main__":
    main()
