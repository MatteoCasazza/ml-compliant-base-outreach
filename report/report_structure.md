# Report Structure

## 1. Problem Statement

## 2. Physics Model
### 2.1 Two-Mass-Spring-Damper System
### 2.2 Robot Workspace Constraint
### 2.3 Chirp Excitation Input
### 2.4 Simulation Outputs

## 3. Dataset Generation
### 3.1 Uniform Dataset
### 3.2 Targeted Augmented Dataset
### 3.3 Dataset Distribution Before and After Augmentation
### 3.4 Dataset Statistics
### 3.5 High-Outreach Feasible Region Analysis

## 4. Surrogate Modeling and Comparison
### 4.1 Surrogate Inputs and Outputs
### 4.2 Gaussian Process Surrogate
### 4.3 Neural Network Surrogate
### 4.4 Learning Curve Analysis
### 4.5 Surrogate Model Comparison
### 4.6 ARD Feature Importance and Physical Interpretation
### 4.7 Final Surrogate Selection

## 5. Constraint-Aware Inverse Optimization
### 5.1 Target-Reaching Problem Formulation
### 5.2 Common Physics-Based Objective Function
### 5.3 Differential Evolution on the Selected Surrogate
### 5.4 Neural Surrogate with Gradient-Based Optimization
### 5.5 Bayesian Optimization with the True Simulator
### 5.6 Optimization Strategy Comparison

## 6. Physical Validation
### 6.1 Validation of DE Solutions
### 6.2 Validation of NN + Gradient Descent Solutions
### 6.3 Validation of BO Solutions
### 6.4 Final Comparison on the True Simulator

## 7. Sim-to-Real Robustness Considerations
### 7.1 Model Mismatch and Parameter Uncertainty
### 7.2 Link Between Validation and Robustness

## 8. Sensitivity Analysis
### 8.1 One-at-a-Time Sensitivity
### 8.2 Physical Interpretation of Parameter Effects

## 9. Monte Carlo Robustness Analysis
### 9.1 Parameter Uncertainty Setup
### 9.2 Robustness of the Optimal Solution
### 9.3 Feasibility Probability
### 9.4 Success Rate Under Uncertainty

## 10. Conclusions
### 10.1 Main Findings
### 10.2 Limitations
### 10.3 Future Work

############################################################################
Presentation Structure  NON AGGIORNATO

A concise slide deck could follow this structure:
Motivation and objective
Physical system and dynamic model
Dataset generation
Dataset augmentation
GP surrogate model
Cross-validation and model choice
Inverse optimization pipeline
Final feasible optimized results
Sensitivity and robustness analysis
Conclusions and future work


A slightly more detailed presentation could add:
- one slide for the GIF/animation;
- one slide for Monte Carlo robustness;
- one slide for future improvements.