# Report Structure

## 1. Problem Statement

## 2. Dynamic Simulator
### 2.1 Physical Model
### 2.2 Chirp Excitation
### 2.3 Output Metrics

## 3. Dataset Generation and Targeted Augmentation
### 3.1 Initial Dataset Generation
### 3.2 Targeted Augmentation
### 3.3 Dataset Statistics

## 4. Learning Curve Analysis

## 5. Surrogate Modeling
### 5.1 Main GP Surrogate for `peak_y`
### 5.2 Auxiliary GP Surrogate for `max_xr`
### 5.3 Constraint Violation Prediction: Derived Metric or Optional Surrogate
### 5.4 Cross-Validation and Kernel Selection
### 5.5 Random Forest Comparison
### 5.6 Final Surrogate Framework

## 6. Constraint-Aware Inverse Optimization
### 6.1 Problem Formulation
### 6.2 Baseline Objective Using Only `peak_y`
### 6.3 Constraint-Aware Objective Using `max_xr` / `constraint_violation`
### 6.4 Comparison of Optimization Algorithms
#### 6.4.1 Random Search
#### 6.4.2 Powell Multi-Start
#### 6.4.3 Differential Evolution
### 6.5 Final Inverse Optimization Setup

## 7. Physical Validation of the Final Optimized Solutions

## 8. Sensitivity and Robustness Analysis
### 8.1 One-at-a-Time Sensitivity Analysis
### 8.2 Monte Carlo Robustness Analysis
### 8.3 Feasibility Under Uncertainty

## 9. Conclusions and Future Work

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