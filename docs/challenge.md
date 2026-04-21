## Model Selection and Final Decision

After reviewing and analyzing the data scientist’s exploratory notebook, several key observations were identified:

- There is no significant difference in overall performance between Logistic Regression and XGBoost when evaluated using standard metrics.

- Handling class imbalance (through techniques such as scale_pos_weight) significantly improves recall for the minority class (delayed flights), which is critical for the business use case.

Given that the problem is inherently imbalanced and the primary objective is to correctly identify delayed flights, recall for class 1 becomes a key metric.

### Final Model Choice

The selected model for production is an XGBoost classifier trained on the top 10 most relevant features with class imbalance handling.

This decision is justified by:

- XGBoost with class balancing achieves higher recall for delayed flights compared to unbalanced models.

- The model maintains performance even when using only the top 10 features, which simplifies deployment and reduces risk of feature drift.

- XGBoost provides consistent performance, scalability, and deterministic behavior, making it suitable for deployment in an API-based inference system.

### Engineering Considerations

During the transition from notebook to production:

- Only features available at prediction time (OPERA, TIPOVUELO, MES) will be retained, avoiding data leakage from post-event variables such as Fecha-O.

- Feature encoding will be standardized to ensure consistency between training and inference.

- The model pipeline will be redesigned to be deterministic and reproducible, removing dependencies on notebook-specific transformations.

