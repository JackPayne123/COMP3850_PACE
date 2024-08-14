from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def calculate_distance_metric(original_text, regenerated_text):
    embeddings = model.encode([original_text, regenerated_text])
    distance = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return distance.item()

def verify_outputs(original_outputs, watermarked_outputs):
    distances = [calculate_distance_metric(orig, regen) for orig, regen in zip(original_outputs, watermarked_outputs)]
    return distances
