# KCL_machine_learning_HN

# Step 1
# Upload label_summary.csv to generate_sum_labels.py
      1️ Extract EPI_ISL_xxx → from split1/split_*.fasta to form the embedding order
      2️ Use these IDs to match Binding_Score_Sum in label_summary.csv
      3 Generate a new ordered_score_sum_labels.npy file for model training

#Step 2
# train_dnn_sum.py

    1 Use the generated ordered_score_sum_labels.npy
    2 Memory-optimized dataset (supports on-demand loading of embedded files)
    3 Multi-layer DNN model regression output (sum of predicted binding scores)
