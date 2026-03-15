import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import datetime

print("Loading model...")

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

model_id = "answerdotai/ModernBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModel.from_pretrained(model_id).to(device)
model.eval()

print("Model loaded.")

# testing chinese vs english sentences
# pairs = [
#     ("狗上了厕所。", "The dog went to the toilet."),
#     ("今天天气很好。", "The weather today is very good."),
#     ("我喜欢吃苹果。", "I like to eat apples."),
#     ("这是一本书。", "This is a book."),
#     ("他在看电视。", "He is watching television."),
#     ("她是我的朋友。", "She is my friend."),
#     ("我们住在爱尔兰。", "We live in Ireland."),
#     ("我在学习中文。", "I am learning Chinese."),
#     ("请给我一杯水。", "Please give me a glass of water."),
#     ("猫在睡觉。", "The cat is sleeping."),
# ]

# testing chinese vs irish sentences
pairs = [
    ("狗上了厕所。", "Chuaigh an madra go dtí an leithreas."),
    ("今天天气很好。", "Tá an aimsir go hálainn inniu."),
    ("我喜欢吃苹果。", "Is maith liom úlla a ithe."),
    ("这是一本书。", "Is leabhar é seo."),
    ("他在看电视。", "Tá sé ag féachaint ar an teilifís."),
    ("她是我的朋友。", "Is cara liom í."),
    ("我们住在爱尔兰。", "Táimid ina gcónaí in Éirinn."),
    ("我在学习中文。", "Táim ag foghlaim Sínise."),
    ("请给我一杯水。", "Tabhair gloine uisce dom, le do thoil."),
    ("猫在睡觉。", "Tá an cat ina chodladh."),
]

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    token_embeddings = output.last_hidden_state
    attention_mask = inputs["attention_mask"]

    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)

    return summed / counts


similarities = []

for t1, t2 in pairs:
    e1 = embed(t1)
    e2 = embed(t2)

    sim = F.cosine_similarity(e1, e2)
    similarities.append(sim.item())

    print(f"{t1}  <->  {t2} = {sim.item():.3f}")

print("Similarity scores:", similarities)

plt.boxplot([similarities], notch=True)
plt.ylabel("Cosine Similarity")
plt.title("Chinese–English Sentence Similarity (ModernBERT)")

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"/Users/shanwhite/Desktop/fyp/twin-movies-v3.10/ModernBERT/results/text_similarity_scores_{timestamp}.png", dpi=300)