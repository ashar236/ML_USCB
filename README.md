# ML_USCB
**Project Overview**

This repository includes hands-on take-home project that builds classification and segmentation models on the U.S. Census dataset. The project notebook is available at [ML_USCB.ipynb](ML_USCB.ipynb).

**About The Project**

- **Goal:** Build and evaluate (1) a classification pipeline to predict whether an individual's income exceeds $50K, and (2) a segmentation pipeline to discover customer segments useful for targeted marketing.
- **Data:** Census dataset with 42 features (demographics, employment, income label). The notebook documents data ingestion, cleaning, and feature engineering.

**Data Exploration**

- **Missingness:** Several migration-related columns contain heavy missing values and are dropped for modeling. Some categorical entries include special tokens ("Not in universe"); these are mapped during preprocessing.
- **Distributions:** Numeric features (age, wage, capital gains/losses, weeks worked) show skew—log transforms and flags (e.g., has_capital_gains) are engineered for modeling.
- **Key Correlations:** Education level, occupation, and class of worker strongly correlate with high-income labels; observed gender imbalance in high-income rates.

**Classification: Training & Evaluation**

- **Preprocessing:** numeric median imputation + scaling; categorical most-frequent imputation + one-hot encoding.
- **Models tried:** Logistic Regression (balanced and unbalanced), Random Forest (balanced/unbalanced), XGBoost.
- **Evaluation metrics:** precision, recall, F1, confusion matrices, ROC AUC, Precision–Recall AUC. The notebook reports a PR AUC ≈ 0.61 for the classification pipeline.
- **Operational insight:** Using `class_weight='balanced'` raises recall for the rare high-income class (e.g., minority recall improved from ~0.38 to ~0.90 in the notebook) at the cost of lower precision—useful when coverage of high-value customers matters.

**Segmentation: Training & Profiling**

- **Feature design:** numeric features plus log1p-transforms, binary flags for capital gains/losses/dividends, and collapsed rare categories for high-cardinality categorical fields.
- **Preprocessing:** median imputation, scaling for numerics; most-frequent imputation + one-hot encoding for categoricals.
- **Clustering:** KMeans evaluated across k=2..10 using inertia and silhouette; k=4 and k=8 are inspected, with k=8 used for detailed profiling in the notebook.
- **Profiling:** PCA visualizations, cluster-size plots and mean feature heatmaps show clusters with distinct average wage, capital gains, and high-income rates—useful for targeted campaigns.

**Results & Takeaways**

- Classification: Balanced training improves recall for the high-income class, which is beneficial when missing premium customers is costly. PR AUC is ~0.61 (noted in the notebook) indicating moderate separation for the positive class.
- Segmentation: Clusters exhibit meaningful differences in income rates and demographic/occupation mixes; several clusters are enriched for higher education and wages and thus are prime targets.
- Practical advice: Choose a model and operating threshold based on campaign priorities—maximize recall for broad reach, or maximize precision when acquisition costs are high.

**How to run**

- Create and activate a suitable virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

- Open and run the analysis notebook: [ML_USCB.ipynb](ML_USCB.ipynb)

**Conclusion**

This project demonstrates a compact, reproducible workflow for both supervised classification and unsupervised segmentation on census data. The notebook contains code, plots, and observations that support model selection and business-facing recommendations for targeting higher-income segments.
