# GNN Forecast Project Documentation

## Overview
This project is focused on forecasting tasks using Graph Neural Networks (GNNs). The aim is to leverage the properties of graph data structures to enhance predictive performance in various domains.

## Data Pipeline
1. **Data Collection**: Gather data from reliable sources, ensuring it's clean and well-structured for analysis.
2. **Data Preprocessing**: 
   - Normalize the data to ensure a uniform scale.
   - Implement techniques to handle missing values and outliers.
   - Convert data into appropriate graph formats for GNN input.
3. **Feature Engineering**: Identify and extract relevant features that can enhance the model's performance.

## GNN Analyses
- **Model Selection**: Choose appropriate GNN models based on data characteristics and forecasting tasks (e.g., GCN, GAT).
- **Training**: Implement training pipelines that include:
  - Split data into training, validation, and test sets.
  - Use techniques like dropout and regularization to prevent overfitting.
- **Evaluation**: Assess model performance using metrics like MAE, RMSE, and R².

## Edge Knockout Simulation
This section details how to conduct edge knockout simulations to understand the influence of specific edges on the overall graph behavior.  
1. **Simulation Setup**: 
   - Define the edges to be knocked out based on hypotheses or analyses.
   - Run the simulation to observe changes in model predictions.
2. **Analysis**:  
   - Compare the performance of the model before and after edge knockout.  
   - Visualize the impact of edge removal on the forecast accuracy.

## Conclusion
This documentation provides a comprehensive overview of the project focusing on data pipelines, GNN analyses, and edge knockout simulations. By adhering to this structure, we aim to improve transparency and maintainability in our forecasting model development.