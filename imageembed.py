import os
import json
import torch
import numpy as np
import pymupdf  # PyMuPDF
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModel

class ImageEmbeddingProcessor:
    def __init__(self, model_name="openai/clip-vit-large-patch14", top_k=3, output_dir="extracted_images", embeddings_file="image_embeddings.json"):
        self.model_name = model_name
        self.top_k = top_k
        self.output_dir = output_dir
        self.embeddings_file = embeddings_file
        self.processor, self.model = self.load_model()
        os.makedirs(self.output_dir, exist_ok=True)
    
    def load_model(self):
        """Load CLIP model and processor."""
        processor = AutoProcessor.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        return processor, model

    def extract_images_with_pymupdf(self, pdf_path):
        """Extract images from a PDF file using PyMuPDF."""
        doc = pymupdf.open(pdf_path)
        images_data = []

        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            img_list = page.get_images(full=True)

            for img_index, img in enumerate(img_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"image_page{page_number + 1}_{img_index + 1}.{image_ext}"
                image_path = os.path.join(self.output_dir, image_filename)

                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)

                images_data.append(image_path)

        doc.close()
        return images_data

    def get_image_embeddings(self, image_paths):
        """Generate embeddings for a list of images."""
        embeddings = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_embedding = self.model.get_image_features(**inputs)
            embeddings.append((image_path, image_embedding.squeeze().cpu().numpy().tolist()))
        return embeddings

    def save_embeddings(self, embeddings):
        """Save image embeddings to a JSON file."""
        with open(self.embeddings_file, "w") as json_file:
            json.dump(embeddings, json_file, indent=4)

    def load_embeddings(self):
        """Load image embeddings from a JSON file."""
        with open(self.embeddings_file, "r") as json_file:
            return json.load(json_file)

    def compute_similarity(self, image_embeddings, text_query):
        """Compute similarity between image and text embeddings using CLIP."""
        text_inputs = self.processor(text=text_query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_embedding = self.model.get_text_features(**text_inputs)
        text_embedding = torch.nn.functional.normalize(text_embedding, p=2, dim=1)
        
        image_embeddings = np.array([np.array(item[1]) for item in image_embeddings])
        image_embeddings = torch.tensor(image_embeddings, dtype=torch.float32)
        image_embeddings = torch.nn.functional.normalize(image_embeddings, p=2, dim=1)
        
        # Ensure embeddings are of the same size before computing similarity
        if text_embedding.shape[1] != image_embeddings.shape[1]:
            print(f"Dimension mismatch: text ({text_embedding.shape[1]}) vs image ({image_embeddings.shape[1]}). Applying projection.")
            projection_layer = torch.nn.Linear(image_embeddings.shape[1], text_embedding.shape[1])
            image_embeddings = projection_layer(image_embeddings)
        
        similarities = torch.matmul(text_embedding, image_embeddings.T)
        return similarities

    def get_top_similar_images(self, similarities, image_paths):
        """Retrieve the top-K most similar images."""
        top_indices = torch.argsort(similarities[0], descending=True)[:self.top_k]
        return [(image_paths[idx], similarities[0, idx].item()) for idx in top_indices]
    
    def display_images(self, image_paths):
        """Display images using matplotlib."""
        fig, axes = plt.subplots(1, len(image_paths), figsize=(15, 5))
        if len(image_paths) == 1:
            axes = [axes]
        for ax, img_path in zip(axes, image_paths):
            image = Image.open(img_path)
            ax.imshow(image)
            ax.axis("off")
        plt.show()

if __name__ == "__main__":
    processor = ImageEmbeddingProcessor()
    pdf_path = input("Enter the PDF file path: ")
    text_query = input("Enter the text query for image similarity search: ")
    
    extracted_images = processor.extract_images_with_pymupdf(pdf_path)
    print(f"Extracted {len(extracted_images)} images.")
    
    image_embeddings = processor.get_image_embeddings(extracted_images)
    processor.save_embeddings(image_embeddings)
    print(f"Image embeddings saved to {processor.embeddings_file}")
    
    loaded_embeddings = processor.load_embeddings()
    similarities = processor.compute_similarity(loaded_embeddings, text_query)
    
    top_images = processor.get_top_similar_images(similarities, [item[0] for item in loaded_embeddings])
    print(f"Most similar images to: '{text_query}'")
    image_paths_to_display = []
    for rank, (image_path, similarity) in enumerate(top_images):
        print(f"{rank+1}. {image_path} (Similarity: {similarity:.4f})")
        image_paths_to_display.append(image_path)
    
    processor.display_images(image_paths_to_display)
