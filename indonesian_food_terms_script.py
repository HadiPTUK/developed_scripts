import csv
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

# Load CSV file and read data
filename = "indonesian_definitions.csv"
word_defs = []
with open(filename, 'r') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  # Skip header
    for row in csv_reader:
        if len(row) >= 2:  # Ensure at least two columns
            word_defs.append((row[0].strip(), row[1].strip()))

# Initialize device, tokenizer, and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased", model_max_length=512)
model = AutoModel.from_pretrained("indolem/indobert-base-uncased").to(device)

# Compute the embedding for the main sentence
main_sentence = "makanan adalah segala sesuatu yang dapat dimakan (seperti penganan, lauk-pauk, kue)"
main_tokens = tokenizer(main_sentence, return_tensors="pt", padding=True, truncation=True).to(device)
with torch.no_grad():  # Disable gradient computation for inference
    main_output = model(**main_tokens)
main_vec = main_output.last_hidden_state.squeeze()[1:-1] # remove [CLS] and [EOL]
main_vec = torch.mean(main_vec, dim=0).to("cpu")  # Mean pooling

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    vec1 = vec1 / torch.norm(vec1)
    vec2 = vec2 / torch.norm(vec2)
    return torch.dot(vec1, vec2).item()

# Batch processing definitions
batch_size = 256  # Adjust batch size based on memory capacity
terms_defs_similarities = []

for i in tqdm(range(0, len(word_defs), batch_size)):
    batch = word_defs[i:i + batch_size]
    batch_texts = [f"{word} {definition}" for word, definition in batch]
    
    tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        output = model(**tokens)
    hidden_states = output.last_hidden_state

    for idx, (word, definition) in enumerate(batch):
        vec = torch.mean(hidden_states[idx].squeeze()[1:-1], dim=0).to("cpu")  # Mean pooling
        similarity = cosine_similarity(main_vec, vec)
        terms_defs_similarities.append((word, definition, similarity))

with open("idn_terms_defs_similarities.csv", mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(terms_defs_similarities)
