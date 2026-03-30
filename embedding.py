from PIL import Image 
import torch
import clip
import numpy as np
from database import VectorDatabase

db = VectorDatabase()

class Embedding:

    def __init__(self, model_name="RN50"):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)

    def get_image_embedding(self, image, target_dim=1000):

        # converter numpy (opencv) → PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(image_input)

        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        embedding_np = embedding.cpu().numpy().flatten()

        if len(embedding_np) > target_dim:
            embedding_np = embedding_np[:target_dim]

        elif len(embedding_np) < target_dim:
            embedding_np = np.pad(embedding_np, (0, target_dim - len(embedding_np)))

        return embedding_np


# similaridade cosseno
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )


# encontra ID parecido
def find_similar_id(new_embedding, embeddings_ids, threshold=0.85):

    best_id = None
    best_score = -1

    for existing_id, emb in embeddings_ids.items():

        score = cosine_similarity(new_embedding, emb)

        if score > best_score:
            best_score = score
            best_id = existing_id

    if best_score >= threshold:
        return best_id, best_score

    return None, best_score