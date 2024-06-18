from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import matplotlib.pyplot as plt
import time
import torch

# prepare image + question
image_path = "question1.png"
# image_path = "question2.png"
# image_path = "question3.png"
image = Image.open(image_path).convert("RGB")

# Display the image
plt.imshow(image)
plt.axis('off')
plt.show()

text = "What is on top of the building?"
# text = "How many cats in this picture?"
# text = "What's this animal's color?"

processor = ViltProcessor.from_pretrained("./models")
model = ViltForQuestionAnswering.from_pretrained("./models")

# prepare inputs
encoding = processor(images=image, text=text, return_tensors="pt")

# forward pass
start_time = time.time()
outputs = model(**encoding)
end_time = time.time()
inference_time = end_time - start_time
logits = outputs.logits

# get top 5 predictions
top_k = 5
top_k_values, top_k_indices = torch.topk(logits, top_k)
top_k_indices = top_k_indices[0].tolist()

print("Top 5 predicted answers:")
for idx in top_k_indices:
    print(f"{model.config.id2label[idx]}")

print("Inference time:", inference_time)
