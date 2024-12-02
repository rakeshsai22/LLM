import torch
from transformers import AutoTokenizer, AutoModel
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def generate_embeddings_bpe(text, model, tokenizer):
    tokens = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.cpu().numpy()


from sklearn.metrics.pairwise import cosine_similarity
import csv

input_file = "/home/snakkill/sem_eval/llama/generated_ans.csv"
results = []

with open(input_file, "r") as file:
    reader = csv.reader(file)
    next(reader)
    for row in reader:
        answer_1, answer_2 = row
        embedding_1 = generate_embeddings_bpe(answer_1.strip(), model, tokenizer)
        embedding_2 = generate_embeddings_bpe(answer_2.strip(), model, tokenizer)
        
        #cosine similarity
        similarity = cosine_similarity(embedding_1, embedding_2)[0][0]
        results.append((answer_1, answer_2, similarity))

output_file = "bpe_similarity_results.csv"
with open(output_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Answer 1", "Answer 2", "Cosine Similarity"])
    writer.writerows(results)

print(f"Comparison completed. Results saved to {output_file}.")


