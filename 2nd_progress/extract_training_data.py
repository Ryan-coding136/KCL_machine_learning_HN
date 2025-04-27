
import pandas as pd
import numpy as np

def extract_features_labels(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    # Label Column
    y = df["binding_score"].values
    np.save(f"{output_dir}/y_binding_score.npy", y)

    # Structural feature column (RSA+Dist × 4 sites)
    struct_cols = [
        "RSA_187", "Dist_187",
        "RSA_222", "Dist_222",
        "RSA_223", "Dist_223",
        "RSA_225", "Dist_225",
    ]
    X_struct = df[struct_cols].values
    np.save(f"{output_dir}/X_structural_features.npy", X_struct)

    # Host tag (0/1)
    X_host = df["Host_Label"].values
    np.save(f"{output_dir}/X_host_label.npy", X_host)

    print(f"✅ Extraction completed：{len(y)} samples")
    print(f"Save tags to: {output_dir}/y_binding_score.npy")
    print(f"Structural features are saved to: {output_dir}/X_structural_features.npy")
    print(f"Host tag is saved to: {output_dir}/X_host_label.npy")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument("--out", required=True, help="Output Directory")
    args = parser.parse_args()

    extract_features_labels(args.csv, args.out)
