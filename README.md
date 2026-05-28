# ML-Based Excitation Planning for Passive-Base Extra-Reach

This project develops a machine-learning-based inverse design pipeline for planning excitation parameters in a passive-base robotic system.

The objective is to exploit the compliant dynamics of the passive base in order to increase the total reachable distance of the robot, while satisfying the physical stroke constraint of the robot actuator.

The total absolute outreach of the system is defined as:

\[
y = x_b + x_r
\]

where:

- \(x_b\) is the displacement of the passive base,
- \(x_r\) is the robot displacement relative to the passive base,
- \(y\) is the total absolute outreach.

The main physical constraint is imposed on the relative robot displacement:

\[
x_r \leq 0.5 \ \text{m}
\]

The project combines:

- a physics-based dynamic simulator,
- dataset generation through Latin Hypercube Sampling,
- targeted dataset augmentation,
- Gaussian Process surrogate modeling,
- constraint-aware inverse optimization,
- final validation using the true dynamic simulator,
- sensitivity and robustness analysis.

---

## 1. Project Overview

The complete workflow is:

```text
Dynamic simulator
        ↓
Dataset generation
        ↓
Targeted augmentation
        ↓
Learning curve analysis
        ↓
Gaussian Process surrogate modeling
        ↓
Constraint surrogate modeling
        ↓
Cross-validation and model comparison
        ↓
Constraint-aware inverse optimization
        ↓
Optimizer comparison
        ↓
Final simulation validation
        ↓
Sensitivity and robustness analysis
```

The final pipeline identifies excitation parameters that allow the system to achieve extra reach beyond the nominal robot stroke. This additional reach is obtained by exploiting passive base motion, not by violating the robot displacement constraint.

The final validated solution is feasible and achieves a total outreach close to the desired target while keeping:

\[
x_r < 0.5 \ \text{m}
\]

---

## 2. Physical System

The simulated system consists of:

- a passive compliant base,
- an impedance-controlled robot,
- a chirp excitation command applied to the robot motion.

The robot desired relative motion is defined as:

\[
x_{rd}(t) = x_{r,start} + A \sin(\phi(t))
\]

where:

- \(x_{r,start}\) is the initial robot relative position,
- \(A\) is the excitation amplitude,
- \(\phi(t)\) is the chirp phase.

The excitation frequency increases from \(f_0\) to \(f_1\) during the simulation. This allows the robot to excite the passive base over a range of frequencies and exploit its dynamic response.

The main simulated metrics are:

| Metric | Meaning |
|---|---|
| `peak_y` | Maximum total absolute outreach |
| `max_xr` | Maximum robot displacement relative to the passive base |
| `max_xb` | Maximum passive base displacement |
| `extra_reach` | Additional outreach beyond the nominal 0.5 m robot stroke |
| `constraint_violation` | Violation of the robot stroke constraint |

The most important output is `peak_y`, because it represents the maximum total reach achieved by the combined robot-base system.

The most important constraint-related output is `max_xr`, because it verifies whether the robot remains within its allowed relative stroke:

\[
max(x_r) \leq 0.5 \ \text{m}
\]

---

## 3. Main Machine Learning Task

The main machine learning task is to build surrogate models that approximate the behavior of the dynamic simulator.

Instead of running the full simulator inside the optimization loop, the project trains Gaussian Process models to predict the most relevant system outputs from the input parameters.

The input vector is:

```text
[Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]
```

where the parameters include both physical system properties and excitation design variables.

The main surrogate model predicts the total outreach:

```text
[Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start] → peak_y
```

A secondary surrogate model predicts the robot stroke constraint:

```text
[Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start] → max_xr
```

The `peak_y` surrogate is used to estimate how close a candidate solution is to the desired target outreach.

The `max_xr` surrogate is used to guide the optimizer toward feasible solutions that respect the robot stroke limit:

\[
max(x_r) \leq 0.5 \ \text{m}
\]

Therefore, the final inverse design problem is not only based on target tracking, but also on physical feasibility.

In summary:

| Model | Predicted output | Role |
|---|---|---|
| Main GP surrogate | `peak_y` | Predicts total outreach |
| Constraint GP surrogate | `max_xr` | Predicts robot stroke feasibility |

---

## 4. Repository Structure

The project is organized as follows:

```text
Proj2/
│
├── data/
│   ├── dataset_outreach.csv
│   └── dataset_augmented.csv
│
├── src/
│   ├── dynamics.py
│   ├── dataset.py
│   ├── model_peak_y.py
│   ├── model_max_xr.py
│   ├── learning_curve.py
│   ├── cross_validation_gp.py
│   ├── model_comparison.py
│   ├── optimization.py
│   ├── optimization_constraint_aware.py
│   ├── optimizer_comparison.py
│   └── visualization.py
│
├── results/
│   ├── augmentation/
│   ├── learning_curve/
│   ├── gp/
│   ├── gp_constraints/
│   ├── cross_validation/
│   ├── model_comparison/
│   ├── optimizer_comparison/
│   ├── constraint_aware/
│   ├── sensitivity/
│   └── robustness/
│
├── figures/
│   ├── augmentation/
│   ├── learning_curve/
│   ├── gp/
│   ├── gp_constraints/
│   ├── cross_validation/
│   ├── model_comparison/
│   ├── optimizer_comparison/
│   ├── constraint_aware/
│   ├── sensitivity/
│   └── robustness/
│
└── README.md
```

The `data/` folder contains the original and augmented datasets used for training and evaluation.

The `src/` folder contains the complete source code of the project, including the simulator, dataset generation, surrogate modeling, optimization, and visualization scripts.

The `results/` folder stores numerical outputs such as metrics, summaries, optimization results, and validation reports.

The `figures/` folder stores the plots generated during the analysis.

If present, the folders `bo/` and `mpc/` contain exploratory experiments and are not part of the final main pipeline.

---

## 5. Main Scripts

This section describes the role of the main Python scripts used in the project.

---

### `src/dynamics.py`

This script contains the dynamic simulator of the passive-base robotic system.

It receives the physical parameters and the excitation parameters as input, simulates the system response, and returns the main output metrics.

Main role:

```text
physical parameters + excitation parameters → simulated trajectory and metrics
```

The most important simulated outputs are:

```text
peak_y
max_xr
max_xb
extra_reach
constraint_violation
```

This simulator represents the ground-truth model used to generate the dataset and to validate the final optimized solutions.

---

### `src/dataset.py`

This script generates the simulation dataset used to train the surrogate models.

The initial dataset is generated using Latin Hypercube Sampling, in order to explore the parameter space efficiently.

Then, targeted augmentation is applied to increase the number of informative samples, especially feasible high-outreach cases.

Main outputs:

```text
data/dataset_outreach.csv
data/dataset_augmented.csv
results/augmentation/dataset_augmentation_summary.csv
```

The augmented dataset is used for the final surrogate modeling pipeline.

---

### `src/model_peak_y.py`

This script trains the main Gaussian Process surrogate model for the prediction of `peak_y`.

The model learns the mapping:

```text
inputs → peak_y
```

where the input vector is:

```text
[Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]
```

Main outputs:

```text
results/gp/gp_model.pkl
results/gp/scaler_X.pkl
results/gp/scaler_y.pkl
results/gp/metrics.csv
results/gp/model_info.txt
figures/gp/gp_parity_plot.png
figures/gp/gp_ard_relevance.png
```

Final test performance:

```text
Test RMSE: 16.31 mm
Test MAE: 7.36 mm
Test R²: 0.976
High-outreach RMSE: 15.47 mm
```

The model shows high prediction accuracy and is used as the main objective surrogate during inverse optimization.

---

### `src/model_max_xr.py`

This script trains the auxiliary Gaussian Process surrogate model for the robot displacement constraint.

The model learns the mapping:

```text
inputs → max_xr
```

This surrogate is used to estimate whether a candidate excitation strategy is feasible with respect to the robot stroke limit.

Main outputs:

```text
results/gp_constraints/gp_max_xr_model.pkl
results/gp_constraints/scaler_X_max_xr.pkl
results/gp_constraints/scaler_y_max_xr.pkl
results/gp_constraints/metrics_max_xr.csv
figures/gp_constraints/max_xr_parity_plot.png
figures/gp_constraints/max_xr_constraint_classification.png
```

Final test performance:

```text
Test RMSE: 6.82 mm
Test MAE: 3.33 mm
Test R²: 0.988
Constraint classification accuracy: 98.44%
False feasible rate: 0.62%
Near-boundary RMSE: 4.42 mm
```

The low false feasible rate is particularly important because it reduces the probability of selecting unsafe solutions during optimization.

---

### `src/learning_curve.py`

This script evaluates how the accuracy of the Gaussian Process surrogate changes as the number of training samples increases.

The goal is to verify whether the augmented dataset improves the predictive performance of the model, especially in the high-outreach region.

Main outputs:

```text
results/learning_curve/learning_curve_summary.csv
figures/learning_curve/learning_curve_rmse.png
figures/learning_curve/learning_curve_r2.png
figures/learning_curve/learning_curve_combined.png
```

Main result:

```text
Global RMSE decreased from 29.59 mm to 16.31 mm.
High-outreach RMSE decreased from 33.11 mm to 15.47 mm.
```

This confirms that the targeted augmentation improves the surrogate accuracy in the most relevant region of the design space.

---

### `src/cross_validation_gp.py`

This script performs 5-fold cross-validation to select the most suitable Gaussian Process configuration.

The tested kernels are:

```text
RBF
Matern 3/2
Matern 5/2
Rational Quadratic
```

The tested noise regularization values are:

```text
alpha = 1e-10
alpha = 1e-6
```

Main outputs:

```text
results/cross_validation/cross_validation_report_table.csv
results/cross_validation/best_kernel_summary.txt
figures/cross_validation/cv_rmse_by_kernel_alpha.png
figures/cross_validation/cv_high_rmse_by_kernel_alpha.png
figures/cross_validation/cv_warnings_by_kernel_alpha.png
figures/cross_validation/cv_summary.png
```

Final selected configuration:

```text
Kernel: Matern 5/2
Alpha: 1e-6
```

The Matern 5/2 kernel achieved the best cross-validation performance and produced no convergence warnings. Although `alpha = 1e-10` and `alpha = 1e-6` gave nearly identical accuracy, `alpha = 1e-6` was selected for better numerical robustness.

---

### `src/model_comparison.py`

This script compares the final Gaussian Process surrogate against a Random Forest baseline.

The comparison is useful to justify the choice of Gaussian Processes as the final surrogate modeling approach.

Main outputs:

```text
results/model_comparison/gp_vs_rf_metrics.csv
results/model_comparison/model_comparison_summary.txt
figures/model_comparison/gp_vs_rf_metrics.png
figures/model_comparison/gp_vs_rf_parity.png
figures/model_comparison/rf_feature_importance.png
```

Main result:

```text
GP test RMSE: 16.31 mm
RF test RMSE: 28.51 mm

GP high-outreach RMSE: 15.47 mm
RF high-outreach RMSE: 35.38 mm
```

The Gaussian Process model was selected because it achieved better accuracy than the Random Forest baseline, especially in the high-outreach region. It also provides predictive uncertainty and ARD-based feature relevance, which are useful for optimization and interpretation.

---

### `src/optimization.py`

This script implements the baseline inverse optimization approach.

It uses only the `peak_y` surrogate model to search for excitation parameters that match a desired target outreach.

This baseline is useful for comparison, but it is not the final optimization strategy because it does not explicitly include the robot stroke constraint during the search.

In other words, it can find solutions that are close to the desired target, but not necessarily physically feasible.

---

### `src/optimization_constraint_aware.py`

This script implements the main inverse optimization pipeline of the project.

It uses two Gaussian Process surrogate models:

```text
GP_peak_y
GP_max_xr
```

The first surrogate predicts the expected total outreach, while the second surrogate predicts the maximum robot relative displacement.

The optimization objective includes:

- target tracking error,
- uncertainty of the `peak_y` prediction,
- predicted violation of the robot stroke constraint,
- uncertainty of the `max_xr` prediction.

The general objective can be written as:

\[
J =
(y_{pred} - y_{target})^2
+ \lambda_u \sigma_y^2
+ \lambda_c \max(0, \hat{x}_{r,max} - x_{r,opt})^2
+ \lambda_{cu} \sigma_{x_r}^2
\]

where:

```text
x_r,true = 0.500 m
x_r,opt  = 0.495 m
```

The value \(x_{r,opt} = 0.495\) m is used as a conservative optimization limit. This safety margin reduces the probability of violating the true physical constraint during final simulator validation.

Main outputs:

```text
results/constraint_aware/baseline_vs_constraint_results.csv
results/constraint_aware/report_summary_baseline_vs_constraint.csv
results/constraint_aware/final_solution_target064.csv

figures/constraint_aware/target_tracking_baseline_vs_constraint.png
figures/constraint_aware/constraint_violation_baseline_vs_constraint.png
figures/constraint_aware/max_xr_baseline_vs_constraint.png
figures/constraint_aware/final_trajectory_target064.png
```

Final selected solution:

```text
Target: 0.640 m
Simulated peak_y: 0.634974 m
Simulation error: 5.03 mm
Extra reach: 0.134974 m
max_xr: 0.494545 m
Constraint violation: 0.00 mm
Feasible: True
```

This result shows that the additional reach is obtained safely, because the robot relative displacement remains below the physical limit.

---

### `src/optimizer_comparison.py`

This script compares three optimization algorithms on the same constraint-aware inverse design problem:

```text
Random Search
Powell Multi-Start
Differential Evolution
```

The comparison evaluates accuracy, feasibility, constraint violation, computational time, and validated success after simulation.

Main outputs:

```text
results/optimizer_comparison/optimizer_comparison_results.csv
results/optimizer_comparison/optimizer_comparison_summary.csv
results/optimizer_comparison/best_optimizer_summary.txt

figures/optimizer_comparison/optimizer_comparison_summary.png
figures/optimizer_comparison/optimizer_comparison_error.png
figures/optimizer_comparison/optimizer_comparison_violation.png
figures/optimizer_comparison/optimizer_comparison_feasibility.png
```

Main result:

```text
Powell Multi-Start:
  mean error = 4.47 mm
  mean time = 94.83 s
  validated success = 100%

Differential Evolution:
  mean error = 4.73 mm
  mean time = 7.36 s
  validated success = 100%

Random Search:
  mean error = 7.20 mm
  mean time = 1.78 s
  validated success = 66.7%
```

Differential Evolution was selected as the final optimizer because it achieved almost the same accuracy as Powell Multi-Start while being much faster and requiring fewer objective evaluations.

---

### `src/visualization.py`

This script generates the final plots used to analyze and present the optimized solution.

It includes:

- final trajectory visualization,
- animation of the optimized motion,
- sensitivity analysis,
- Monte Carlo robustness analysis.

Main outputs:

```text
figures/constraint_aware/final_trajectory_target064.png
figures/animation/constraint_aware_target064.gif
figures/sensitivity/sensitivity_constraint_aware_target064.png
figures/robustness/monte_carlo_constraint_aware_target064_noise2.png
figures/robustness/monte_carlo_constraint_aware_target064.png
```

These outputs are used to verify the final behavior of the system and to evaluate how robust the selected solution is under parameter perturbations.

---

## 6. How to Run the Pipeline

The recommended execution order is shown below.

### 1. Generate or update the dataset

```bash
python src/dataset.py
```

This generates the initial dataset and the augmented dataset used for surrogate training.

---

### 2. Train the main GP surrogate

```bash
python src/model_peak_y.py
```

This trains the Gaussian Process model used to predict `peak_y`.

---

### 3. Train the constraint GP surrogate

```bash
python src/model_max_xr.py
```

This trains the Gaussian Process model used to predict `max_xr`.

---

### 4. Run the learning curve analysis

```bash
python src/learning_curve.py
```

This evaluates the effect of the dataset size on the prediction accuracy.

---

### 5. Run GP cross-validation

```bash
python src/cross_validation_gp.py
```

This compares different kernel and regularization configurations.

---

### 6. Compare GP against Random Forest

```bash
python src/model_comparison.py
```

This compares the selected Gaussian Process surrogate with a Random Forest baseline.

---

### 7. Run constraint-aware inverse optimization

```bash
python src/optimization_constraint_aware.py
```

This runs the main inverse optimization pipeline using both the outreach surrogate and the constraint surrogate.

---

### 8. Compare optimizers

```bash
python src/optimizer_comparison.py
```

This compares Random Search, Powell Multi-Start, and Differential Evolution on the same inverse design task.

---

### 9. Generate final visualizations and robustness analysis

```bash
python src/visualization.py
```

This generates the final trajectory plots, sensitivity analysis, robustness analysis, and animation.

---

## 7. Final Main Results

This section summarizes the main quantitative results obtained by the final project pipeline.

---

### Dataset

The final dataset was obtained by combining the initial Latin Hypercube Sampling dataset with targeted augmentation.

```text
Final augmented dataset size: 1600 samples
Feasible samples: 803
Feasible extra-reach cases: 352
Max feasible peak_y: 0.840586 m
Max feasible extra reach: 0.340586 m
```

The targeted augmentation increased the number of feasible high-outreach samples, making the dataset more informative for surrogate modeling and inverse optimization.

---

### Main GP surrogate for `peak_y`

The main Gaussian Process surrogate predicts the maximum total outreach of the system.

```text
Test RMSE: 16.31 mm
Test MAE: 7.36 mm
Test R²: 0.976
High-outreach RMSE: 15.47 mm
```

These results show that the surrogate can accurately approximate the dynamic simulator, including in the high-outreach region of the design space.

---

### Constraint GP surrogate for `max_xr`

The auxiliary Gaussian Process surrogate predicts the maximum robot relative displacement.

```text
Test RMSE: 6.82 mm
Test MAE: 3.33 mm
Test R²: 0.988
Constraint accuracy: 98.44%
False feasible rate: 0.62%
```

The low false feasible rate is especially important because unsafe solutions incorrectly classified as feasible could lead to violations of the robot stroke constraint.

---

### Constraint-aware optimization

The final inverse optimization was performed using both the `peak_y` surrogate and the `max_xr` constraint surrogate.

```text
Target: 0.640 m
Achieved peak_y: 0.634974 m
Error: 5.03 mm
Extra reach: 0.134974 m
max_xr: 0.494545 m
Constraint violation: 0.00 mm
Feasible: True
```

The optimized solution reaches the desired target with a small error while remaining below the robot stroke limit.

---

### Optimizer comparison

Differential Evolution was selected as the final optimizer.

```text
Reason:
- 100% feasibility
- 0 mm constraint violation
- 100% validated physical success
- similar accuracy to Powell Multi-Start
- approximately 13 times faster than Powell Multi-Start
```

This makes Differential Evolution the best trade-off between accuracy, feasibility, robustness, and computational cost.

---

### Monte Carlo robustness

The final solution was tested under perturbations of the fixed passive-base parameters.

Under 2% perturbations:

```text
Mean achieved outreach: 0.635014 m
Mean absolute error: 5.62 mm
Success rate within 10 mm: 84.0%
Feasibility rate: 100.0%
Max violation: 0.00 mm
```

Under 5% perturbations:

```text
Mean achieved outreach: 0.635952 m
Mean absolute error: 10.24 mm
Feasibility rate: 100.0%
Max violation: 0.00 mm
```

The robustness analysis shows that the final solution remains safe and feasible under parameter uncertainty. However, target tracking accuracy is sensitive to variations in the passive-base dynamics.

## 8. Final Interpretation

This project demonstrates that machine learning can be used to plan excitation parameters for a passive-base robotic system.

The Gaussian Process surrogate accurately approximates the dynamic simulator and predicts the achievable total outreach. The auxiliary constraint surrogate estimates the maximum robot relative displacement and helps the optimizer avoid unsafe solutions.

The final constraint-aware inverse optimization achieves extra reach beyond the nominal robot stroke while respecting the physical robot constraint.

The final validated solution reaches:

```text
peak_y = 0.634974 m
extra reach = 0.134974 m
max_xr = 0.494545 m
constraint violation = 0.00 mm
```

This confirms that the additional outreach is obtained through passive base motion, rather than by exceeding the allowed robot stroke.

Overall, the project shows a complete machine learning workflow for physics-based inverse design:

```text
simulation → dataset → surrogate modeling → constrained optimization → validation