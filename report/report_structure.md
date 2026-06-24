# Report Structure

## 1. Problem Statement

## 2. Physics Model
### 2.1 Dynamic System and Governing Equations
### 2.2 Chirp Excitation
### 2.3 Robot Workspace Constraint

## 3. Dataset Generation
### 3.1 Uniform Dataset Design
### 3.2 Targeted Augmented Dataset
### 3.3 Augmentation effect and High-Outreach Feasible Region 

## 4. Surrogate Modeling and Comparison
### 4.1 Gaussian Process Surrogate
### 4.2 Neural Network Surrogate
### 4.3 Learning Curve Analysisand Surrogate Model Comparison
### 4.4 ARD Feature Importance and Physical Interpretation

## 5. Constraint-Aware Inverse Optimization
### 5.1 GP-Based Differential Evolution
### 5.2 Neural Surrogate Gradient-Based Optimization
### 5.3 Bayesian Optimization with the True Simulator
### 5.4 Final Optimization Comparison and Simulator Validation

## 6. Physical Validation of Optimized Solutions
### 6.1 Validation Protocol and Selected Targets
### 6.2 Reachable Target Time-Response Validation
### 6.3 Saturated Target Time-Response Validation
### 6.4 Physical Interpretation of Constraint Saturation

## 7. Sensitivity and Robustness Analysis
### 7.1 One-at-a-Time Sensitivity Setup
### 7.2 Reachable Target Sensitivity
### 7.3 Saturated Target Sensitivity
### 7.4 Monte Carlo Parameter-Uncertainty Setup
### 7.5 Monte Carlo Robustness Results
### 7.6 Robustness and Sim-to-Real Considerations

## 8. Conclusions
### 8.1 Main Findings
### 8.2 Limitations
### 8.3 Future Work

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