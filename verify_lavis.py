import os
import random
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
import matplotlib.pyplot as plt
# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# 2. Load the Model
print("⏳ Loading BLIP-2 Model...")
try:
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt",
        model_type="pretrain_opt2.7b",
        is_eval=True,
        device=device
    )
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit()

# 3. Load Test Images
images_dir = os.path.join(os.path.dirname(__file__), "RandomImages")
image_files = [
    os.path.join(images_dir, name)
    for name in sorted(os.listdir(images_dir))
    if name.lower().endswith(".png")
]
if not image_files:
    raise FileNotFoundError(f"No PNG images found in {images_dir}")

print(f"Found {len(image_files)} images to process\n")

# 4. Process Each Image
results = []
output_dir = "/data1/mariam/varad_mech/Lavis_BLIP/LAVIS"

for idx, image_path in enumerate(image_files, 1):
    print(f"[{idx}/{len(image_files)}] Processing: {os.path.basename(image_path)}")

    # Load and preprocess image
    raw_image = Image.open(image_path).convert("RGB")
    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    # Generate caption
    print("⏳ Generating Caption...")
    with torch.no_grad():
        generated_text = model.generate({"image": image})
    caption = generated_text[0]

    # Store result
    results.append({
        "image_name": os.path.basename(image_path),
        "caption": caption
    })

    print(f"✅ Caption: {caption}\n")

    # Save image with caption
    plt.figure(figsize=(10, 6))
    plt.imshow(raw_image)
    plt.title(f"{caption}", fontsize=12, color='blue', wrap=True)
    plt.axis('off')

    output_filename = f"captioned_{os.path.splitext(os.path.basename(image_path))[0]}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"💾 Saved to: {output_path}\n")

# 5. Summary
print("="*50)
print("🎉 SUCCESS! All images processed.")
print("="*50)
for result in results:
    print(f"📷 {result['image_name']}: {result['caption']}")
print("="*50)
