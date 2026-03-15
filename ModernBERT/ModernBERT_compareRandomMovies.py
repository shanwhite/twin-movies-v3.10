import os
import random
import datetime
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

# personally using a M3 mac to run this project but kept facing errors, so switched from cpu to mps
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

torch.set_grad_enabled(False)

# load model on mps in float16 
model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModel.from_pretrained(
    model_id,
    dtype=torch.float16
).to(device)

# Warm-up pass (MPS compiles kernels here)
_ = model(**tokenizer("warmup", return_tensors="pt").to(device))

# load movie summaries from cmu dataset
def load_summaries(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            movie_id, summary = line.split("\t", 1)
            data.append((movie_id.strip(), summary.strip()))
    return data

movies = load_summaries("cmu_data/plot_summaries.txt")

# mean pooling
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1)
    return summed / counts

# compute similarities of N pairs of movies 
N = 20
similarities = []

for _ in range(N):
    # calculate similarity score
    (movie_id1, text1), (movie_id2, text2) = random.sample(movies, 2)

    # print out movie IDs and summaries of each pair
    print(f"\nMovie 1 ID: {movie_id1}")
    print(f"Movie 1 Summary: {text1}\n")

    print(f"Movie 2 ID: {movie_id2}")
    print(f"Movie 2 Summary: {text2}\n")

    # tokenize (reduced max_length for speed)
    inputs1 = tokenizer(text1, return_tensors="pt", truncation=True, max_length=256)
    inputs2 = tokenizer(text2, return_tensors="pt", truncation=True, max_length=256)

    inputs1 = {k: v.to(device) for k, v in inputs1.items()}
    inputs2 = {k: v.to(device) for k, v in inputs2.items()}

    out1 = model(**inputs1)
    out2 = model(**inputs2)

    emb1 = mean_pool(out1.last_hidden_state, inputs1["attention_mask"])
    emb2 = mean_pool(out2.last_hidden_state, inputs2["attention_mask"])

    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)

    sim = F.cosine_similarity(emb1, emb2).item()
    similarities.append(sim)

print(f"\nLast similarity score: {similarities[-1]:.4f}")

# save notch boxplot results
# kept facing errors when results displayed in a popup, so i switched to saving them to a folder instead
os.makedirs("../results/modernbert", exist_ok=True)

plt.figure(figsize=(8, 6))
plt.boxplot(similarities, notch=True, patch_artist=True,
            boxprops=dict(facecolor="lightblue"))
plt.title("Similarity Scores Between Random Movie Pairs (ModernBERT)")
plt.ylabel("Cosine Similarity")
plt.grid(axis="y", linestyle="--", alpha=0.6)

# save results with unique timestamp so they'll save separately instead of overwriting previous images
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"/Users/shanwhite/Desktop/fyp/twin-movies-v3.10/ModernBERT/results/random_similarity_scores_{timestamp}.png", dpi=300)
plt.close()
