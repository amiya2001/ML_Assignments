1. Problem Statement

Network intrusion detection is a critical task in cybersecurity, aimed at identifying malicious activities within network traffic. Traditional rule-based systems often fail to detect sophisticated and evolving attacks.
This project focuses on building a machine learning–based intrusion detection system to classify network traffic as normal or malicious using the UNSW-NB15 dataset.

⸻

2. Dataset Description

The UNSW-NB15 dataset is a publicly available network intrusion detection dataset containing both normal and attack traffic.
It consists of multiple numerical and categorical features extracted from real network flows.

Target Variable:
	•	label
	•	0 → Normal traffic
	•	1 → Malicious traffic

Dataset Preprocessing:
	•	A stratified sample of 20,000 records was selected to preserve class distribution.
	•	Non-informative columns such as id and multi-class attribute attack_cat were removed.

⸻

3. Feature Selection

A total of 22 features were selected:
	•	19 Numerical Features (traffic volume, timing, packet statistics)
	•	3 Categorical Features (proto, service, state)

This selection balances model performance and computational efficiency while avoiding redundant attributes.

⸻

4. Models Implemented

The following machine learning models were trained and evaluated:
	1.	Logistic Regression
	2.	Decision Tree
	3.	Random Forest
	4.	K-Nearest Neighbors (KNN)
	5.	Naive Bayes
	6.	XGBoost

All models were trained using a unified preprocessing pipeline to ensure fair comparison.

⸻

5. Evaluation Metrics

Each model was evaluated using the following metrics:
	•	Accuracy
	•	Precision
	•	Recall
	•	F1-score
	•	AUC (Area Under ROC Curve)
	•	MCC (Matthews Correlation Coefficient)

⸻

6. Model Comparision

| Model | Accuracy | Precision | Recall | F1-score | AUC | MCC |
|------|----------|-----------|--------|----------|-----|-----|
| Logistic Regression | 0.9165 | 0.905571 | 0.979427 | 0.941052 | 0.956189 | 0.805947 |
| Decision Tree | 0.92950 | 0.922438 | 0.978692 | 0.949733 | 0.973746 | 0.836181 |
| Random Forest | 0.93575 | 0.919075 | 0.993020 | 0.954618 | 0.986844 | 0.852659 |
| SVM | 0.92475 | 0.908263 | 0.989346 | 0.947072 | 0.914177 | 0.826753 |
| KNN | 0.92275 | 0.928902 | 0.959956 | 0.944173 | 0.962572 | 0.820095 |
| Naive Bayes | 0.48425 | 1.000000 | 0.242101 | 0.389825 | 0.631283 | 0.304317 |
| XGBoost | 0.93975 | 0.941302 | 0.972079 | 0.956443 | 0.987867 | 0.860005 |

⸻

7. Observations
	•	XGBoost and Random Forest achieved the highest overall performance across most metrics.
	•	Ensemble-based models demonstrated better generalization compared to linear and probabilistic models.
	•	Naive Bayes showed comparatively lower performance due to its strong independence assumptions.
	•	KNN performance was sensitive to data scaling but provided reasonable results after preprocessing.

⸻

8. Conclusion

This project demonstrates the effectiveness of machine learning techniques for network intrusion detection.
Among all evaluated models, XGBoost delivered the best overall performance, making it a strong candidate for real-world intrusion detection systems.

⸻

9. Technologies Used
	•	Python
	•	Pandas, NumPy
	•	Scikit-learn
	•	XGBoost
	•	Streamlit (for deployment)

⸻

10. Streamlit Application

A Streamlit-based web application was developed to allow users to upload test data, select models, and view prediction results and evaluation metrics.

(https://mlassignments-n2qvq4d3jfr6kncrkypx3r.streamlit.app)
