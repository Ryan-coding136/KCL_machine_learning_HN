import os
import glob
import pandas as pd
import numpy as np

def extract_epi_ids(fasta_folder):
    fasta_files = sorted(glob.glob(os.path.join(fasta_folder, "split_*.fasta")))
    epi_ids = []
    for fasta in fasta_files:
        with open(fasta, "r") as f:
            for line in f:
                if line.startswith(">"):
                    parts = line.strip().split("|")
                    epi_id = parts[1].strip()  # EPI_ISL_xxxxxxx
                    epi_ids.append(epi_id)
    return epi_ids

def match_labels(epi_ids, csv_path):
    df = pd.read_csv(csv_path)
    df = df.set_index("Isolated_Id")

    matched, missing = [], []
    for eid in epi_ids:
        if eid in df.index:
            score = df.loc[eid]["Binding_Score_Sum"]
            if isinstance(score, pd.Series):
                score = score.iloc[0] 
            try:
                matched.append(int(score))
            except Exception as e:
                print(f"⚠️ Failed to parse the tag：{eid} → {score} ({e})")
                missing.append(eid)
        else:
            missing.append(eid)

    print(f"✅ Successfully matched：{len(matched)} ")
    print(f"❗ Failed to match：{len(missing)} ")

    return np.array(matched), missing

def save_labels(labels, output_file="ordered_score_sum_labels.npy"):
    np.save(output_file, labels)
    print(f"💾 Label saved as {output_file}")

if __name__ == "__main__":
    fasta_folder = "split1"
    csv_path = "label_summary.csv"

    epi_ids = extract_epi_ids(fasta_folder)
    labels, missing = match_labels(epi_ids, csv_path)
    save_labels(labels)

    if missing:
        with open("missing_ids_score_sum.txt", "w") as f:
            f.write("\n".join(missing))
        print("📄 The unmatched ID is saved as missing_ids_score_sum.txt")
