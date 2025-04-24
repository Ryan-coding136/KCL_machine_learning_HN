
import pandas as pd
import numpy as np

def extract_features_labels(csv_path, output_dir):
    df = pd.read_csv(csv_path)

    # 标签列
    y = df["binding_score"].values
    np.save(f"{output_dir}/y_binding_score.npy", y)

    # 结构特征列（RSA+Dist × 4 位点）
    struct_cols = [
        "RSA_187", "Dist_187",
        "RSA_222", "Dist_222",
        "RSA_223", "Dist_223",
        "RSA_225", "Dist_225",
    ]
    X_struct = df[struct_cols].values
    np.save(f"{output_dir}/X_structural_features.npy", X_struct)

    # Host 标签（0/1）
    X_host = df["Host_Label"].values
    np.save(f"{output_dir}/X_host_label.npy", X_host)

    print(f"✅ 提取完成：{len(y)} 个样本")
    print(f"标签保存至: {output_dir}/y_binding_score.npy")
    print(f"结构特征保存至: {output_dir}/X_structural_features.npy")
    print(f"Host 标签保存至: {output_dir}/X_host_label.npy")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="输入的 CSV 路径")
    parser.add_argument("--out", required=True, help="输出目录")
    args = parser.parse_args()

    extract_features_labels(args.csv, args.out)
