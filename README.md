# Smart-Logic-AI-Powered-Depth-Prediction-in-Digital-Circuits-


Problem Statement
Predicting logical depth in ALU circuits is essential for optimizing circuit design and improving performance. However, real-time synthesis datasets are limited, making it challenging to develop accurate models. This project aims to solve this problem by leveraging machine learning techniques to predict logical depth efficiently, benefiting researchers and engineers working in digital circuit design.

Overview
This project focuses on predicting the logical depth of ALU circuits using Gradient Boosting and Graph Neural Networks (GNNs). Since real-time synthesis datasets were limited, synthetic samples were generated to maintain consistency with real ALU patterns.

Features
•	Gradient Boosting Model: A traditional ML approach using engineered features.
•	Graph Neural Network (GNN): A deep learning model leveraging circuit graph structures.
•	Hybrid Approach (GNN + Gradient Boosting): Combining both models for improved prediction accuracy.

Dataset
•	The dataset includes circuit parameters such as num_inputs, num_outputs, fan_in, fan_out, gate_density, complexity_score, critical_path_length, and logic depth.
•	It was initially small and augmented using random synthetic samples that match real ALU circuit patterns.

Solution
To accurately predict logical depth in ALU circuits, we evaluated multiple machine learning models. Gradient Boosting emerged as the best-performing model with an R² score of ~0.91, making it the most reliable method for logical depth estimation. We also compared it with Graph Neural Networks (GNNs), which captured circuit topology but had lower accuracy (R² ≈ 0.74). Furthermore, an ensemble approach combining GNN and Gradient Boosting achieved R² ≈ 0.88, showing that ensembling enhances prediction performance but does not surpass the Gradient Boosting model alone.

Implementation
1️.Gradient Boosting Model
•	Preprocessing: Data normalization and feature selection.
•	Training: Implemented with GradientBoostingRegressor from Scikit-learn.
•	Evaluation: 
  o	Mean Squared Error (MSE)
  o	R² Score
2️.Graph Neural Network (GNN) Model
•	Graph Representation: Circuit data structured using k-NN graphs.
•	Model Architecture: 
  o	Multi-layer GATConv-based GNN.
  o	Dropout and BatchNorm for regularization.
  o	Trained using Huber Loss.
•	Evaluation: 
  o	Mean Squared Error (MSE)
  o	R² Score
3️.Hybrid Model (GNN + Gradient Boosting)
•	Predictions from Gradient Boosting and GNN were averaged.
•	Performance Comparison: 
  o	Gradient Boosting alone: R² ≈ 0.91
  o	GNN alone: R² ≈ 0.74
  o	Hybrid Model: R² ≈ 0.88
  
Results
•	Gradient Boosting provided the best performance for logical depth prediction.
•	The Hybrid Model (GNN + Gradient Boosting) improved prediction accuracy over GNN alone but did not outperform Gradient Boosting.
•	Ensembling techniques can enhance performance, but tabular data-driven models remain the most effective for this task.

Installation & Setup
1. Clone the repository:
git clone https://github.com/duhitslik/Smart-Logic-AI-Powered-Depth-Prediction-in-Digital-Circuits
cd Smart-Logic-AI-Powered-Depth-Prediction-in-Digital-Circuits
2. Set up the environment
It is recommended to use a virtual environment to manage dependencies:
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Dataset : logical_depth_dataset.scv
5. Run the models
python gradient_boosting.py  # Train Gradient Boosting model
python Comparision_models.py  # Train GNN model+Hybrid

Future Improvements
•	Enhancing GNN structure by incorporating more circuit topology insights.
•	Exploring ensemble techniques beyond simple averaging.
•	Expanding real-world dataset collection for better validation.
________________________________________
Contributors
•	Likitha S
For questions, open an issue or reach out! 📩
pes1202203918@pesu.pes.edu

