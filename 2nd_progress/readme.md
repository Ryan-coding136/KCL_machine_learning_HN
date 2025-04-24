#Project Workflow: Predicting Binding Affinity Using ESM-2 Embeddings and Structural Features

Stage 1: Data Preparation

1. Sequence Processing
	•	H1 HA protein sequences were split into 6 FASTA files: split_1.fasta to split_6.fasta.
	•	ESM-2 embeddings (320 dimensions) were generated for each file.
	•	Script: generate_embeddings.py
	•	Output files: split_1_embeddings.npy to split_6_embeddings.npy
	•	All embeddings were concatenated into a single file: X_esm2_embeddings.npy

2. Label and Feature Construction
	•	Input metadata: H1_mutation_table_labeled_with_host.csv
	•	Output labels:
	•	Binding scores (0/1/2): y_binding_score.npy
	•	Structural proxy features:
	•	Average RSA + Distance across 5 representative PDBs: X_structural_features.npy (8 dimensions)
	•	Host type feature:
	•	Human = 0, Others = 1: X_host_label.npy

⸻

Stage 2: Model Building and Training

1. Model Architecture
	•	Input feature dimensions: 320 (ESM-2) + 8 (structure) + 1 (host) = 329
	•	MLP architecture example: 320 → 64 → 32 → 3
	•	The training script supports:
	•	Class weighting
	•	Early stopping
	•	Plotting for loss and ROC curves

2. Training and Evaluation
	•	Multiple training sessions were conducted with different class weight settings:
	•	train_model_weighted.py (balanced weights)
	•	train_model_weighted_soft.py (soft adjustment)
	•	Each training run generates:
	•	Confusion matrix: confusion_matrix.png
	•	ROC curve: roc_curve.png
	•	Loss curve: loss_curve.png
	•	Control variables:
	•	Fixed network architecture
	•	Consistent train-validation split

⸻

Stage 3: Results Comparison and Analysis

1. Performance Metrics
	•	Validation accuracy
	•	ROC AUC scores per class
	•	Class distribution vs prediction accuracy

2. Effect of Class Weights
	•	Compared results from:
	•	No weighting
	•	Hard class weighting
	•	Soft class weighting
	•	Insight:
	•	Weighting improves minority class performance (especially Class 0/1)

3. Misclassification Analysis
	•	Reasons for poor performance:
	•	Class imbalance still influences prediction
	•	May require more diverse mutation samples in minority classes

⸻

Stage 4: Future Work
	•	Explore higher-dimensional ESM-2 embeddings (e.g., 640 dimensions)
	•	Expand model architecture with more hidden layers or regularization (e.g., Dropout)
	•	Add amino acid physicochemical features
	•	Investigate if misclassified samples have mutations outside the RBS
	•	Apply trained model to H5/H7/H9 data for validation and generalization
