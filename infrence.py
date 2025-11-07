import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler


from model import GPTLanguageModel
print(GPTLanguageModel)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# the used data
with open('harry.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all the unique chars in our data
chars = sorted(list(set(text)))
vocab_size = len(chars)
# chars to int
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


model = GPTLanguageModel().to(device)
model.eval()
model.load_state_dict(torch.load('model.pth', map_location=device))

def encode(input_str):
    return torch.tensor([stoi.get(c, stoi.get(' ', 0)) for c in input_str],
                        dtype=torch.long).unsqueeze(0).to(device)

def decode(tensor):
    return ''.join([itos.get(i.item() if torch.is_tensor(i) else i, ' ') 
                    for i in tensor])

def generate_novel_chapter(prompt, length=5000, temperature=0.85, top_k=50, top_p=0.92):
    """Generate a novel-length chapter"""
    print(f"\nGenerating {length} tokens (approximately {length*0.75:.0f} words)...")
    input_ids = encode(prompt)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_new_tokens=length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    generated_text = decode(output_ids[0].tolist())
    return generated_text

# interactive novel generation
print("\n" + "="*60)
print("HARRY POTTER STYLE NOVEL GENERATOR")
print("="*60)
print("\nCommands:")
print("  'chapter' - Generate a full chapter (~3750 words)")
print("  'scene' - Generate a scene (~750 words)")
print("  'page' - Generate a page (~250 words)")
print("  'custom' - Custom length generation")
print("  'exit' - Quit")
print("\nOr just type a prompt to generate a response.")

while True:
    user_input = input("\nYou: ").strip()
    
    if user_input.lower() in ['exit', 'quit']:
        break
    
    try:
        if user_input.lower() == 'chapter':
            prompt = input("Enter chapter opening (e.g., 'Chapter 12: The Hidden Chamber'): ")
            text = generate_novel_chapter(prompt, length=5000, temperature=0.85)
            print(f"\n{text}")
            
            # save to file
            save = input("\nSave to file? (y/n): ")
            if save.lower() == 'y':
                filename = input("Filename: ") + ".txt"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Saved to {filename}")
        
        elif user_input.lower() == 'scene':
            prompt = input("Enter scene opening: ")
            text = generate_novel_chapter(prompt, length=1000, temperature=0.85)
            print(f"\n{text}")
        
        elif user_input.lower() == 'page':
            prompt = input("Enter prompt: ")
            text = generate_novel_chapter(prompt, length=300, temperature=0.85)
            print(f"\n{text}")
        
        elif user_input.lower() == 'custom':
            prompt = input("Enter prompt: ")
            length = int(input("Length in tokens (500-10000): "))
            temp = float(input("Temperature (0.7-1.2, default 0.85): ") or "0.85")
            text = generate_novel_chapter(prompt, length=length, temperature=temp)
            print(f"\n{text}")
        
        else:
            #quick generation
            input_ids = encode(user_input)
            with torch.no_grad():
                response_ids = model.generate(
                    input_ids, 
                    max_new_tokens=500,
                    temperature=0.85,
                    top_k=50,
                    top_p=0.92
                )
            response_text = decode(response_ids[0].tolist())
            print(f"\nModel: {response_text}")
    
    except Exception as e:
        print(f"Error: {e}")