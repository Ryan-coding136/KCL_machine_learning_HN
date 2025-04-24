
import argparse
import torch
import numpy as np
from transformers import EsmModel, EsmTokenizer
from Bio import SeqIO
from tqdm import tqdm

def load_model(model_name="facebook/esm2_t6_8M_UR50D"):
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    return model, tokenizer, device

def extract_embeddings(fasta_path, output_path):
    model, tokenizer, device = load_model()
    embeddings = []

    for record in tqdm(SeqIO.parse(fasta_path, "fasta")):
        seq = str(record.seq).replace("U", "X").replace("*", "")
        tokenized = tokenizer(seq, return_tensors="pt", truncation=True)
        input_ids = tokenized["input_ids"].to(device)
        with torch.no_grad():
            outputs = model(input_ids)
            last_hidden = outputs.last_hidden_state[0]  # (L, D)
            mean_embed = last_hidden[1:len(seq)+1].mean(dim=0)  # Skip [CLS] and [PAD]
        embeddings.append(mean_embed.cpu().numpy())

    np.save(output_path, np.stack(embeddings))
    print(f"✅ Saved {len(embeddings)} embeddings to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output .npy file for embeddings")
    args = parser.parse_args()
    extract_embeddings(args.input, args.output)
