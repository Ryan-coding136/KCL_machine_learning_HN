Project Workflow: Predicting Binding Affinity Using ESM-2 Embeddings and Structural Features

⸻

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
	•	Average RSA + Distance values for selected RBS residues, extracted from five representative H1 PDB structures
	•	Output: X_structural_features.npy (8 dimensions: 4 positions × 2 features each)
	•	Host type feature:
	•	From matched metadata: X_host_label.npy (Human = 0, Non-human = 1)

⸻

Stage 2: Binding Score Classification Criteria

Definition of Three Binding Classes

Each sequence was associated with four key receptor-binding residues in the HA RBS (positions 187, 222, 223, and 225), selected based on known functional importance (literature: Q226L, G228S, D225G, etc.).
	•	For each sequence, these 4 positions were analyzed for amino acid mutations compared to the dominant “avian” residue.
	•	A mutation was counted as “adaptive” if it changed to a residue associated with enhanced human binding (e.g., Q226L, D225G, S227R, etc.).
	•	The total number of such mutations among the 4 key sites determined the binding score:

Mutation Count	Binding Score	Description
0 mutations	     0	        Weak binding
1–2 mutations	     1	        Moderate binding
3–4 mutations	     2	        Strong binding

	•	These scores were assigned programmatically and saved in y_binding_score.npy.

⸻

Stage 3: Model Building and Training

1. Model Architecture
	•	Input dimensions: 320 (ESM-2) + 8 (structure) + 1 (host) = 329
	•	Example MLP: 320 → 64 → 32 → 3 (3-way classification)
	•	Training script supports:
	•	Class weighting
	•	EarlyStopping
	•	Loss and ROC curve plotting

2. Training and Evaluation
	•	Training experiments with different class weighting strategies:
	•	train_model_weighted.py: class-balanced weights
	•	train_model_weighted_soft.py: softly adjusted weights
	•	Evaluation metrics:
	•	Accuracy, ROC AUC, confusion matrix

⸻

Stage 4: Results Analysis

1. Key Metrics
	•	Evaluation curves:
	•	Confusion matrix
	•	ROC curves per class (One-vs-Rest)
	•	Training/validation loss curves

2. Class Weight Impact
	•	First-round models had poor minority-class recall
	•	Second-round training with adjusted weights significantly improved Class 0/1 performance

3. Class Distribution (Example)
	•	Label distribution: Class 2 (strong binders) dominates due to high prevalence of zoonotic mutations in avian sequences

4. Misclassification Insight
	•	Sequences misclassified as “strong” may carry non-RBS adaptive mutations, suggesting additional signals beyond the four key residues

⸻

Stage 5: Future Work
	•	Increase ESM-2 embedding dimensionality (e.g., 640)
	•	Add dropout, batch norm, or deeper layers
	•	Integrate full-length HA domain-aware features (e.g., glycosylation)
	•	Train on H1/H3, test on H5/H7/H9
	•	Apply explainability (e.g., SHAP) to identify key residues
	•	Publish a scoring interface for outbreak monitoring
