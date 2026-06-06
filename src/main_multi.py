import os
import warnings
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import cv2
import logging
import openai
from dotenv import load_dotenv
from transformers import BertTokenizer, BertModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import base64

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("VALIDATION_DATA_DIR", PROJECT_ROOT / "data" / "validation"))
CHECKPOINT_PATH = Path(os.getenv("MODEL_CHECKPOINT", PROJECT_ROOT / "checkpoints" / "fine_tuned_resnet18.pth"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Load models for semantic and syntactic analysis
bert_model_path = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_path)
question_model = BertModel.from_pretrained(bert_model_path).to(device)
syntax_model = SentenceTransformer('all-MiniLM-L6-v2')

# Modify the number of classes to match the fine-tuned model
num_classes = 41

# Function to map classes to indices
def get_class_to_idx(data_dir):
   classes = sorted(entry.name for entry in os.scandir(data_dir) if entry.is_dir())
   class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
   return class_to_idx

data_dir = str(DATA_DIR)
class_to_idx = get_class_to_idx(data_dir)
class_labels = list(class_to_idx.keys())[:num_classes]  # Adjust to match the fine-tuned model

class HumanActivityDataset(Dataset):
   def __init__(self, data_dir, class_to_idx, transform=None):
       self.data_dir = data_dir
       self.transform = transform
       self.class_to_idx = class_to_idx
       self.data = []
       for class_dir in os.listdir(data_dir):
           class_path = os.path.join(data_dir, class_dir)
           if os.path.isdir(class_path):
               for file_name in os.listdir(class_path):
                   if file_name.endswith(('.mp4', '.avi')):
                       self.data.append((os.path.join(class_path, file_name), self.class_to_idx[class_dir]))

   def __len__(self):
       return len(self.data)

   def __getitem__(self, idx):
       video_path, label = self.data[idx]
       return video_path, label

# Define data transformations
transform = transforms.Compose([
   transforms.Resize((224, 224)),
   transforms.ToTensor(),
   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset and dataloader
dataset = HumanActivityDataset(data_dir, class_to_idx, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Load the fine-tuned ResNet18 model
image_model = models.resnet18()
num_features = image_model.fc.in_features
image_model.fc = nn.Linear(num_features, num_classes)

# Safely load the model
image_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
image_model = image_model.to(device)
image_model.eval()

# Load OpenAI API Key from environment variables
openai.api_key = os.getenv('OPENAI_API_KEY')

def extract_keyframes(video_path, interval=30):
   """Extract keyframes from the video at specified intervals."""
   cap = cv2.VideoCapture(video_path)
   frames = []
   frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   for i in range(0, frame_count, interval):
       cap.set(cv2.CAP_PROP_POS_FRAMES, i)
       ret, frame = cap.read()
       if ret:
           frames.append(frame)
   cap.release()
   return frames

def select_snippets(frames, snippet_length=16):
   """Create snippets from extracted frames."""
   snippets = []
   total_frames = len(frames)
   for i in range(0, total_frames, snippet_length):
       snippet = frames[i:i + snippet_length]
       if len(snippet) == snippet_length:
           snippets.append(snippet)
   return snippets

def create_adversarial_example(model, image_tensor, true_label_tensor, epsilon):
   """Create adversarial examples."""
   image_tensor.requires_grad = True
   output = model(image_tensor)

   true_label_tensor = true_label_tensor.squeeze()

   # Validate true_label_tensor to ensure it's not out of bounds
   if true_label_tensor.dim() == 0:
       true_label_tensor = true_label_tensor.unsqueeze(0)

   if true_label_tensor.long().item() < 0 or true_label_tensor.long().item() >= num_classes:
       print(f"Warning: True label {true_label_tensor.item()} is out of bounds. Defaulting to class 0.")
       true_label_tensor = torch.tensor([0]).to(image_tensor.device)

   loss = F.cross_entropy(output, true_label_tensor.long())
   model.zero_grad()
   loss.backward()
   sign_data_grad = image_tensor.grad.data.sign()
   perturbed_image = image_tensor + epsilon * sign_data_grad
   perturbed_image = torch.clamp(perturbed_image, 0, 1)
   return perturbed_image

class AssistantModel:
   def __init__(self, image_model, device):
       self.image_model = image_model.to(device)
       self.device = device

   def encode_image_to_base64(self, image):
       """Encode image to base64."""
       _, buffer = cv2.imencode('.jpeg', image)
       base64_image = base64.b64encode(buffer).decode('utf-8')
       return base64_image

   def preprocess_image(self, image):
       """Preprocess the image for inference."""
       if isinstance(image, np.ndarray):
           if image.ndim != 3 or image.shape[2] != 3:
               print(f"Invalid frame shape: {image.shape}. Skipping this frame.")
               return None
           if image.dtype != np.uint8:
               image = (image * 255).astype(np.uint8)
           image = Image.fromarray(image, 'RGB')
       elif not isinstance(image, Image.Image):
           raise TypeError(f"Unexpected image type: {type(image)}, expected np.ndarray or Image.Image")

       preprocess = transforms.Compose([
           transforms.Resize((224, 224)),
           transforms.ToTensor(),
           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
       ])

       try:
           image_tensor = preprocess(image).unsqueeze(0).to(self.device)
       except Exception as e:
           print(f"Error during preprocessing: {e}. Skipping this frame.")
           return None
       return image_tensor

   def process_images(self, frames, true_label, epsilon=0.1, attack=False):
       """Process images to get classification results."""
       self.image_model.eval()
       descriptions = []
       encoded_frames = []

       for frame in frames:
           image_tensor = self.preprocess_image(frame)
           if image_tensor is None:
               continue

           if attack:
               true_label_tensor = torch.tensor([true_label]).to(self.device)
               image_tensor = create_adversarial_example(self.image_model, image_tensor, true_label_tensor, epsilon)

           base64_image = self.encode_image_to_base64(frame)
           encoded_frames.append(base64_image)

           with torch.no_grad():
               try:
                   output = self.image_model(image_tensor)
                   _, pred = torch.max(output, 1)
                   predictions = int(pred.item())

                   if 0 <= predictions < len(class_labels):
                       descriptions.append(class_labels[predictions])
                   else:
                       print(f"Predicted class index out of bounds: {predictions}. Setting to 'Unknown Activity'.")
                       descriptions.append("Unknown Activity")

               except Exception as e:
                   print(f"Error during model inference: {e}. Skipping this frame.")
                   continue

       if not descriptions:
           print("No valid frames for classification.")
           return None, []  # Return empty if no valid frames

       description_votes = {desc: descriptions.count(desc) for desc in set(descriptions)}
       final_description = max(description_votes, key=description_votes.get)
       return final_description, encoded_frames  # Return description and encoded frames

def calculate_similarity(response1, response2):
   """Calculate semantic and syntactic similarities between responses."""
   inputs1 = tokenizer(response1, return_tensors="pt", padding=True, truncation=True).to(device)
   inputs2 = tokenizer(response2, return_tensors="pt", padding=True, truncation=True).to(device)

   with torch.no_grad():
       embeddings1 = question_model(**inputs1).last_hidden_state.mean(dim=1)
       embeddings2 = question_model(**inputs2).last_hidden_state.mean(dim=1)

   semantic_similarity = cosine_similarity(embeddings1.cpu().numpy(), embeddings2.cpu().numpy())[0][0]

   syntax_embeddings1 = syntax_model.encode(response1)
   syntax_embeddings2 = syntax_model.encode(response2)
   syntax_similarity = cosine_similarity([syntax_embeddings1], [syntax_embeddings2])[0][0]

   return semantic_similarity, syntax_similarity

def select_class_samples(class_index, dataset):
   """Select samples from a specific class."""
   class_samples = [(video_path, label) for video_path, label in dataset.data if label == class_index]
   return DataLoader(class_samples, batch_size=1, shuffle=True)

def main(num_videos=1, specific_class=None):
   model = ChatOpenAI(model=OPENAI_MODEL)
   video_count = 0
   attempted_videos = 0

   # If a specific class is provided, create a DataLoader for that class
   if specific_class is not None:
       if specific_class < 0 or specific_class >= len(class_labels):
           raise ValueError("Invalid class index.")
       dataloader = select_class_samples(specific_class, dataset)
   else:
       dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

   for video_path, true_label in dataloader:
       if attempted_videos >= num_videos:
           break

       video_path = video_path[0]
       attempted_videos += 1

       print(f"Processing video {video_count + 1}/{num_videos}")
       print(f"Video Path: {video_path}")

       # Extract keyframes and generate snippets
       frames = extract_keyframes(video_path, interval=5)
       snippets = select_snippets(frames, snippet_length=4)

       if not snippets:
           print("Error: No valid snippets extracted.")
           continue

       selected_frames = [frame for snippet in snippets for frame in snippet]

       # Initialize the assistant model
       assistant_model = AssistantModel(image_model, device)

       # Process all frames to get classifications for the attacked video
       print("Attacking all frames...")
       attacked_description, encoded_frames = assistant_model.process_images(selected_frames, true_label, attack=True)

       if attacked_description is None:
           print("Error: No valid frames for classification after attack.")
           continue

       # Prepare to evaluate the original video (without attack), sending the same frames for baseline
       original_description, _ = assistant_model.process_images(selected_frames, true_label, attack=False)

       # Print results for both original and attacked descriptions
       print(f"\n--- Classification Results ---")
       print(f"Original Video Description: {original_description}")
       print(f"Attacked Video Description: {attacked_description}")

       # Prepare the question for ChatGPT
       question = "In a paragraph: Suppose you are in a house. Do you think this is a suitable activity to be carried out here based on the provided video?"

       # Limit number of images to send to ChatGPT (let's say max 4 images)
       max_images_to_send = 4  
       selected_frames_for_chatgpt = encoded_frames[:max_images_to_send]  # Take up to the max number of frames

       # Prepare a single prompt with multiple images for the original video
       prompt_original = question + "\nHere are some frames from the original video:\n"
       for base64_image in selected_frames_for_chatgpt:
           prompt_original += f"data:image/jpeg;base64,{base64_image}\n"

       # Send requests for the original video
       original_response = model([HumanMessage(content=prompt_original)])  # Send question + frames as a single request
       print(f"Response from ChatGPT (Original): {original_response.content}")

       # Prepare a single prompt with multiple images for the attacked video
       prompt_attacked = question + "\nHere are some frames from the attacked video:\n"
       for base64_image in selected_frames_for_chatgpt:
           prompt_attacked += f"data:image/jpeg;base64,{base64_image}\n"

       # Send requests for the attacked video
       attacked_response = model([HumanMessage(content=prompt_attacked)])  # Send question + frames as a single request
       print(f"Response from ChatGPT (Attacked): {attacked_response.content}")

       # Calculate semantic and syntactic similarities
       semantic_sim, syntax_sim = calculate_similarity(original_response.content, attacked_response.content)
       print(f"Semantic Similarity between Original and Attacked: {semantic_sim}")
       print(f"Syntactic Similarity between Original and Attacked: {syntax_sim}")

       video_count += 1

if __name__ == "__main__":
   main(num_videos=1, specific_class=3)  # Change 0 to the index of the desired class
