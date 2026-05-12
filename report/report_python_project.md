# Machine Learning for Outreach Prediction and Inverse Design in Compliant Base Robotics

**A Gaussian Process-Based Surrogate Modeling and Inverse Optimization Framework**

---

## Abstract

This work presents a machine learning framework for predicting and controlling the peak outreach of a robotic system mounted on a compliant base. The system is modeled as two coupled mass-spring-damper subsystems: an impedance-controlled robot and a passive compliant base. Instead of suppressing the base motion, the proposed framework exploits the passive dynamics of the compliant base to achieve a desired total outreach.

A physics-based simulator is first implemented to generate data. A dataset of 500 simulations is then created using Latin Hypercube Sampling over physical and excitation parameters. A Gaussian Process (GP) surrogate model is trained to approximate the mapping from system parameters to peak outreach, achieving a test R² of 0.9676, RMSE of 21.97 mm, and MAE of 11.08 mm. The GP is then used inside an inverse optimization framework based on differential evolution to find controllable parameters that achieve specified target outreach values.

The optimized solutions are validated using the original dynamic simulator, obtaining a mean simulation error of 1.465 mm and a maximum error of 3.297 mm over five target values. Automatic Relevance Determination (ARD) identifies the robot damping ratio, base mass, and robot stiffness as the most influential parameters. Additional analyses include learning curves, kernel comparison, Random Forest comparison, sensitivity analysis, and Monte Carlo robustness validation. The final framework demonstrates that Gaussian Process surrogate modeling can support accurate prediction, interpretable parameter analysis, and reliable inverse design for compliant-base robotic systems.

---

## 1. Introduction

### 1.1 Problem Statement

Modern robotic systems are often mounted on mobile, flexible, or compliant structures. In many applications, this compliance is treated as an unwanted disturbance that must be compensated. However, in some cases, the passive motion of the supporting structure can be exploited to improve task performance.

This project investigates a robotic system mounted on a compliant base. The system is modeled as two coupled mass-spring-damper subsystems connected in series:

- a controllable robot subsystem;
- a passive compliant base subsystem.

The main goal is to predict and control the total end-effector outreach:

```text
y = x_b + x_r
```

where `x_b` is the base displacement and `x_r` is the robot displacement relative to the base.

Rather than simply maximizing the robot displacement alone, the framework aims to exploit the coupled dynamics of the robot and base in order to reach desired outreach targets.

### 1.2 Project Objective

The objective of the project is to build a complete machine learning pipeline able to:

1. simulate the physical dynamics of the robot-base system;
2. generate a dataset by varying physical and excitation parameters;
3. train a surrogate model for peak outreach prediction;
4. interpret the influence of the different parameters;
5. solve an inverse problem: given a desired outreach target, find parameters that achieve it;
6. validate the optimized parameters with the original dynamic simulator.

The final approach is an offline, open-loop inverse-design strategy. The parameters are optimized before execution and then validated in simulation. No feedback correction is applied during execution.

### 1.3 Project Requirements and Implemented Solution

| Project requirement | Implemented solution |
|---|---|
| Dynamic model of a compliant robotic system | 2-DOF mass-spring-damper simulator in `src/dynamics.py` |
| Excitation of the passive dynamics | Chirp command with parameters `f0`, `f1`, `A`, `x_r_start` |
| Dataset generation | 500 Latin Hypercube Sampling simulations |
| ML-based performance model | Gaussian Process surrogate model |
| Input/output selection | 9 parameters → peak outreach |
| Inverse design / optimization | Differential evolution using the GP surrogate |
| Validation | Optimized parameters tested with the true dynamic simulator |
| Advanced analysis | Learning curve, ARD, kernel comparison, Random Forest comparison, sensitivity, Monte Carlo robustness |

### 1.4 Contributions

The main contributions of this project are:

- implementation of a complete physics-to-ML pipeline in Python;
- generation of a 500-sample dataset using Latin Hypercube Sampling;
- training of a Gaussian Process surrogate model with high predictive accuracy;
- ARD-based interpretation of parameter relevance;
- kernel comparison and task-aware selection of the final GP kernel;
- comparison between Gaussian Process and Random Forest regression;
- inverse optimization with simulator-validated millimeter-level accuracy;
- sensitivity analysis and Monte Carlo robustness validation.

---

## 2. Dynamic System Modeling

### 2.1 System Description

The system consists of a robot mounted on a compliant base. The base is passive, while the robot is driven through a commanded trajectory.

The main variables are:

```text
x_b = displacement of the compliant base
x_r = robot displacement relative to the base
y   = x_b + x_r = total outreach
```

The state vector is:

```text
[dxb, dxr, xb, xr]
```

where `dxb` and `dxr` are the corresponding velocities.

The output of interest is the maximum total outreach reached during a simulation.

**[Insert Fig. 1: `figures/test_simulation.png`]**

### 2.2 Equations of Motion

The implemented dynamic equations are:

```text
ddxb = (Dr*dxr + Kr*(xr - xrd) - Db*dxb - Kb*xb) / Mb

ddxr = (-Dr*dxr - Kr*(xr - xrd)) / Mr - ddxb
```

where:

- `Kb` is the base stiffness;
- `Kr` is the robot stiffness;
- `Mb` is the base mass;
- `Mr` is the robot mass;
- `Db` is the base damping coefficient;
- `Dr` is the robot damping coefficient;
- `xrd(t)` is the commanded robot trajectory.

The damping coefficients are computed from the damping ratios:

```text
Db = 2 * hb * sqrt(Kb * Mb)

Dr = 2 * hr * sqrt(Kr * Mr)
```

where `hb` and `hr` are the base and robot damping ratios.

### 2.3 Excitation Signal

The robot is excited using a chirp command:

```text
xrd(t) = x_r_start + A * cos(2π(f0*t + 0.5*k*t²))
```

with:

```text
k = (f1 - f0) / T
```

where:

- `f0` is the initial chirp frequency;
- `f1` is the final chirp frequency;
- `A` is the chirp amplitude;
- `x_r_start` is the initial robot relative position;
- `T` is the simulation duration.

A chirp signal is useful because it excites the system over a range of frequencies, allowing the coupled robot-base dynamics to be explored.

### 2.4 Performance Metric

The simulation output is:

```text
peak_y = max_t (x_b(t) + x_r(t))
```

This quantity represents the maximum outreach achieved during the excitation.

The first test simulation produced:

```text
Peak outreach ≈ 0.588 m
```

### 2.5 Implementation Notes

The simulator is implemented in:

```text
src/dynamics.py
```

using `scipy.integrate.solve_ivp`.

A relevant detail is that the chirp starts from:

```text
xrd(0) = x_r_start + A
```

whereas the initial state is:

```text
x_r(0) = x_r_start
```

Therefore, the command contains an initial jump equal to `A`. This generates an initial transient, which is consistent with the reference MATLAB formulation used for the project.

---

## 3. Dataset Generation

### 3.1 Input and Output Variables

The dataset maps physical and excitation parameters to the peak outreach.

The input vector is:

```text
[Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start]
```

The output is:

```text
peak_y
```

### 3.2 Parameter Space

| Parameter | Range | Meaning |
|---|---:|---|
| `Kb` | 100–5000 N/m | Base stiffness |
| `Kr` | 100–5000 N/m | Robot stiffness |
| `Mb` | 10–100 kg | Base mass |
| `hb` | 0.1–0.3 | Base damping ratio |
| `hr` | 0.1–1.1 | Robot damping ratio |
| `f0` | 0.0001–0.01 Hz | Initial chirp frequency |
| `f1` | 3–10 Hz | Final chirp frequency |
| `A` | 0.1–0.2 m | Chirp amplitude |
| `x_r_start` | 0.3–0.5 m | Initial robot position |

The robot mass is fixed:

```text
Mr = 10 kg
```

### 3.3 Sampling Strategy

The dataset was generated using:

```text
Latin Hypercube Sampling (LHS)
```

LHS was chosen because it provides good coverage of a high-dimensional parameter space with fewer samples than a dense grid. Since the problem has 9 input dimensions, a full factorial grid would be computationally inefficient.

### 3.4 Dataset Size

The initial dataset contained:

```text
220 simulations
```

This was later extended to:

```text
500 simulations
```

to improve the robustness and generalization of the GP surrogate model.

The final dataset is saved in:

```text
data/dataset_outreach.csv
```

### 3.5 Dataset Statistics

The final dataset has the following approximate statistics:

```text
Number of samples: 500
Minimum peak_y:    0.398 m
Maximum peak_y:    1.054 m
Mean peak_y:       0.607 m
```

Figures generated for dataset analysis:

```text
figures/dataset_distribution.png
figures/dataset_correlations.png
figures/dataset_scatter.png
```

**[Insert Fig. 2: `figures/dataset_distribution.png`]**

---

## 4. Learning Curve and Dataset Size Selection

A learning curve was implemented in:

```text
src/learning_curve.py
```

to evaluate how the number of training samples affects GP performance.

### 4.1 Initial Dataset: 220 Samples

With the initial dataset, the model showed strong improvement as the number of training samples increased.

| Training samples | Test R² | Test RMSE |
|---:|---:|---:|
| 20 | 0.7148 | 49.54 mm |
| 80 | 0.9313 | 24.31 mm |
| 150 | 0.9446 | 21.84 mm |
| 176 | 0.9394 | 22.93 mm |

The results indicated that 220 samples were already sufficient to obtain a good surrogate model, but additional samples could improve stability.

### 4.2 Final Dataset: 500 Samples

The final 500-sample dataset led to the following learning curve:

| Training samples | Test R² | Test RMSE |
|---:|---:|---:|
| 20 | 0.6675 | 69.92 mm |
| 100 | 0.9087 | 36.83 mm |
| 200 | 0.9424 | 28.92 mm |
| 350 | 0.9668 | 22.19 mm |
| 400 | 0.9676 | 21.97 mm |

The curve shows that performance improves significantly up to about 300–350 training samples and then starts to stabilize. Therefore, the final dataset of 500 simulations was selected as a good compromise between accuracy and computational cost.

Output files:

```text
results/learning_curve_results.csv
figures/learning_curve.png
```

**[Insert Fig. 3: `figures/learning_curve.png`]**

---

## 5. Gaussian Process Surrogate Model

### 5.1 Motivation

Running the full dynamic simulation repeatedly inside an optimization loop can be computationally expensive. Therefore, a surrogate model was trained to approximate the mapping:

```text
[Kb, Kr, Mb, hb, hr, f0, f1, A, x_r_start] → peak_y
```

A Gaussian Process was selected because it provides:

- high accuracy on moderate-size datasets;
- uncertainty quantification;
- smooth predictions suitable for optimization;
- interpretability through Automatic Relevance Determination.

### 5.2 Model Formulation

A Gaussian Process defines a distribution over functions:

```text
f(x) ~ GP(m(x), k(x, x'))
```

The final kernel used in the official model is:

```text
ConstantKernel * Matern(ν = 5/2) + WhiteKernel
```

The Matern 5/2 kernel was selected as the final kernel because it provided the best downstream performance in inverse optimization, even though Matern 3/2 had slightly better test-set metrics.

### 5.3 Preprocessing and Training

The training pipeline is:

```text
1. Load dataset
2. Split into train/test sets
3. Standardize X using StandardScaler
4. Standardize y using StandardScaler
5. Train Gaussian Process
6. Evaluate on test set
7. Save model and scalers
```

Train/test split:

```text
80% training = 400 samples
20% testing  = 100 samples
```

Both input and output standardization were applied to improve numerical conditioning.

### 5.4 Final GP Results

The official final model uses Matern 5/2.

| Set | R² | RMSE | MAE |
|---|---:|---:|---:|
| Training | 0.999968 | 0.545 mm | 0.342 mm |
| Test | 0.9676 | 21.97 mm | 11.08 mm |

The test R² of 0.9676 means that the model explains about 96.76% of the variance in peak outreach.

The GP also provides uncertainty estimates:

```text
Mean predictive std = 12.32 mm
Max predictive std  = 43.99 mm
```

Output files:

```text
results/gp_model.pkl
results/scaler_X.pkl
results/scaler_y.pkl
results/metrics.csv
```

Figures:

```text
figures/gp_parity_plot.png
figures/gp_residuals.png
figures/gp_uncertainty.png
```

**[Insert Fig. 4: `figures/gp_parity_plot.png`]**

**[Insert Fig. 5: `figures/gp_uncertainty.png`]**

---

## 6. Automatic Relevance Determination

### 6.1 Concept

Automatic Relevance Determination (ARD) was used to estimate the relative influence of each input parameter.

In the GP kernel, each input dimension has its own length-scale. A shorter length-scale means that the output changes strongly when that parameter changes, so the parameter is more relevant.

The normalized relevance is computed as:

```text
r_i = (1 / l_i) / sum_j (1 / l_j)
```

where `l_i` is the length-scale of the i-th parameter.

### 6.2 ARD Results

The final ARD ranking using Matern 5/2 is:

| Rank | Parameter | Relevance |
|---:|---|---:|
| 1 | `hr` | 0.378 |
| 2 | `Mb` | 0.226 |
| 3 | `Kr` | 0.189 |
| 4 | `Kb` | 0.101 |
| 5 | `A` | 0.041 |
| 6 | `hb` | 0.040 |
| 7 | `x_r_start` | 0.021 |
| 8 | `f1` | 0.002 |
| 9 | `f0` | 0.001 |

**[Insert Fig. 6: `figures/gp_ard_relevance.png`]**

### 6.3 Interpretation

The most relevant parameter is `hr`, the robot damping ratio. This is physically reasonable because damping strongly affects how energy is dissipated during the excitation.

The second and third most relevant parameters are `Mb` and `Kr`, indicating that the base inertia and robot stiffness strongly affect the coupled response.

Within the sampled frequency range, `f0` and `f1` showed low relevance compared with the main mechanical parameters. This suggests that the considered output, `peak_y`, is more sensitive to damping, mass, and stiffness than to the exact chirp frequency bounds in the tested range.

---

## 7. Kernel and Model Comparison

### 7.1 Kernel Comparison

A kernel comparison was implemented in:

```text
src/kernel_comparison.py
```

The tested kernels were:

```text
RBF
Matern 1/2
Matern 3/2
Matern 5/2
Rational Quadratic
```

Results:

| Kernel | Test R² | RMSE | MAE |
|---|---:|---:|---:|
| Matern 3/2 | 0.9716 | 20.57 mm | 10.72 mm |
| Matern 5/2 | 0.9676 | 21.97 mm | 11.08 mm |
| RBF | 0.9660 | 22.49 mm | 12.26 mm |
| Matern 1/2 | 0.9390 | 30.12 mm | 14.82 mm |
| Rational Quadratic | 0.8751 | 43.12 mm | 25.94 mm |

**[Insert Fig. 7: `figures/kernel_comparison.png`]**

### 7.2 Task-Aware Kernel Selection

The Matern 3/2 kernel obtained the best test-set metrics. However, the final project objective is not only prediction, but inverse optimization validated with the true dynamic simulator.

For this reason, Matern 3/2 and Matern 5/2 were also compared in the inverse optimization task.

| Kernel | Mean simulator error | Max simulator error |
|---|---:|---:|
| Matern 3/2 | 6.555 mm | 14.626 mm |
| Matern 5/2 | 1.465 mm | 3.297 mm |

Although Matern 3/2 achieved slightly better predictive metrics, it produced less accurate inverse-design solutions. Matern 5/2 was therefore selected as the final kernel because it provided a smoother and more stable optimization landscape for the downstream task.

This is an important point: the surrogate model was selected based on the final end-to-end objective, not only on test-set prediction accuracy.

### 7.3 Random Forest Comparison

A Random Forest Regressor was also trained as a baseline model in:

```text
src/model_comparison.py
```

Results:

| Model | Test R² | RMSE | MAE |
|---|---:|---:|---:|
| Gaussian Process | 0.9676 | 21.97 mm | 11.08 mm |
| Random Forest | 0.8359 | 49.41 mm | 27.38 mm |

**[Insert Fig. 8: `figures/model_comparison.png`]**

The GP clearly outperformed the Random Forest on the test set.

Moreover, the GP provides:

- predictive uncertainty;
- ARD-based interpretability;
- smoother behavior for inverse optimization.

These advantages make it more suitable for this project than the Random Forest baseline.

---

## 8. Inverse Optimization Framework

### 8.1 Problem Formulation

The inverse problem is:

```text
Given a target y_target,
find controllable parameters that produce y_sim ≈ y_target.
```

The fixed parameters are:

```text
Kb, Mb, hb, Mr
```

The optimized controllable parameters are:

```text
Kr, hr, f0, f1, A, x_r_start
```

### 8.2 Optimization Method

The optimizer used is:

```text
scipy.optimize.differential_evolution
```

The objective function is evaluated using the GP surrogate:

```text
J(theta) = (y_GP(theta) - y_target)^2
```

The final configuration used was:

```text
maxiter = 100
popsize = 10
workers = 1
updating = immediate
```

A larger configuration with `maxiter=1000` and `popsize=15` was also tested, but it produced worse simulator-validated solutions because the optimizer exploited surrogate inaccuracies more strongly. Therefore, the smaller and more stable configuration was retained.

### 8.3 Validation Protocol

The GP is only used to search for promising parameter values. The final optimized parameters are always validated using the original dynamic simulator.

Therefore, the reported inverse optimization error is not just a surrogate error. It is a simulator-validated error.

For each target, the following quantities are compared:

```text
target outreach
GP-predicted outreach
simulated outreach
absolute simulation error
```

### 8.4 Target Values

Five target values were selected from dataset percentiles:

| Percentile | Target |
|---:|---:|
| 20% | 0.520767 m |
| 35% | 0.557278 m |
| 50% | 0.592779 m |
| 65% | 0.628967 m |
| 80% | 0.681966 m |

### 8.5 Results

| Target | GP prediction | Simulation | Error |
|---:|---:|---:|---:|
| 0.520767 m | 0.520767 m | 0.519726 m | 1.041 mm |
| 0.557278 m | 0.557278 m | 0.560575 m | 3.297 mm |
| 0.592779 m | 0.592779 m | 0.593118 m | 0.339 mm |
| 0.628967 m | 0.628967 m | 0.631119 m | 2.152 mm |
| 0.681966 m | 0.681968 m | 0.681470 m | 0.496 mm |

Summary:

```text
Mean simulator error = 1.465 mm
Maximum error        = 3.297 mm
Success rate         = 100%
```

These results show that the GP surrogate can guide the optimizer toward parameter values that remain accurate when tested on the true simulator.

Output files:

```text
results/inverse_results.csv
figures/inverse_targets.png
figures/inverse_trajectory_target1.png
figures/inverse_trajectory_target3.png
figures/inverse_trajectory_target5.png
```

**[Insert Fig. 9: `figures/inverse_targets.png`]**

**[Optional Fig. 10: `figures/inverse_trajectory_target3.png`]**

---

## 9. Visualization, Sensitivity, and Robustness Analysis

Advanced visualizations were implemented in:

```text
src/visualization.py
```

### 9.1 Animation

A GIF animation was generated to show the physical behavior of the system.

The animation includes:

```text
wall
passive base
base spring
robot spring
robot mass
target outreach
current outreach
time evolution of y(t)
```

File:

```text
figures/animation_target3.gif
```

For the central target:

```text
y_target = 0.592779 m
y_sim    = 0.593118 m
error    = 0.339 mm
```

### 9.2 GP Response Surface

A 2D response surface was generated over the plane:

```text
Kr vs hr
```

with the other parameters fixed to the optimized central-target solution.

File:

```text
figures/gp_surface_Kr_hr.png
```

This visualization shows that the GP learned a smooth response surface in the space of two highly relevant parameters.

**[Optional Fig. 11: `figures/gp_surface_Kr_hr.png`]**

### 9.3 Sensitivity Analysis

A one-at-a-time sensitivity analysis was performed on:

```text
hr, Mb, Kr, A, Kb
```

For each parameter:

1. the parameter is varied over its range;
2. the GP predicts the corresponding peak outreach;
3. selected points are validated using the true simulator.

Output:

```text
figures/sensitivity_analysis.png
results/sensitivity_analysis.csv
```

Qualitative results:

- `hr` has a strong effect on outreach;
- `A` increases outreach almost linearly;
- `Mb` clearly affects the dynamic response;
- `Kr` and `Kb` show more nonlinear effects.

**[Insert Fig. 10: `figures/sensitivity_analysis.png`]**

### 9.4 Monte Carlo Robustness

A Monte Carlo robustness test was performed by perturbing the fixed base parameters:

```text
Kb, Mb, hb
```

with uncertainty of approximately ±5%.

Number of samples:

```text
100
```

Final results:

```text
Target               = 0.592779 m
Mean achieved        = 0.595520 m
Std achieved         = 0.002108 m
Range achieved       = [0.590621, 0.603281] m
5th–95th percentile  = [0.592070, 0.598473] m
Mean error           = 2.931 mm
Max error            = 10.502 mm
Success rate         = 99%
```

The optimized solution remains close to the target even when the base parameters are perturbed. Only one case slightly exceeds the 10 mm error threshold, leading to a 99% success rate.

Output files:

```text
figures/monte_carlo_robustness.png
results/monte_carlo_robustness.csv
results/monte_carlo_robustness_stats.csv
```

**[Insert Fig. 11: `figures/monte_carlo_robustness.png`]**

---

## 10. Discussion

### 10.1 Effectiveness of the Framework

The complete pipeline successfully addresses the original problem:

```text
dynamic model
→ dataset generation
→ GP surrogate
→ inverse optimization
→ simulator validation
```

The GP surrogate achieves high predictive accuracy, and the inverse optimization reaches target values with millimeter-level simulation error.

The most important result is that the optimizer does not merely fit the GP prediction: the solutions are validated with the true dynamic simulator and remain accurate.

### 10.2 Why Gaussian Process?

The GP was selected because it provides several advantages for this problem:

- strong performance with a moderate-size dataset;
- smooth prediction surface;
- uncertainty quantification;
- ARD-based interpretability;
- good integration with optimization.

A Random Forest baseline was also tested, but it produced significantly worse test performance and does not naturally provide predictive uncertainty.

### 10.3 Why Matern 5/2?

Although Matern 3/2 achieved slightly better test-set prediction metrics, Matern 5/2 produced much better inverse optimization performance.

This shows that the final model should be chosen based on the downstream task, not only on standard prediction metrics.

For this project, the most important task is:

```text
target outreach → optimized parameters → simulator validation
```

For that task, Matern 5/2 was the best option.

### 10.4 Open-Loop Nature of the Method

The proposed method is an offline open-loop inverse-design strategy. The optimized excitation parameters are computed before execution and then applied during the simulation.

No feedback correction is applied during the trajectory.

A natural extension would be closed-loop Model Predictive Control, where the system state is measured at each time step and the command is recomputed over a finite prediction horizon.

### 10.5 Limitations

The main limitations of the current work are:

1. the results are simulation-based and not experimentally validated on hardware;
2. the physical model assumes linear spring-damper dynamics;
3. the method is open-loop and does not correct errors during execution;
4. the robot mass is fixed;
5. the optimization objective focuses only on peak outreach;
6. the GP predicts only peak outreach, not the full trajectory or next state.

---

## 11. Conclusions and Future Work

### 11.1 Conclusions

This project developed a complete machine learning framework for outreach prediction and inverse design in a compliant-base robotic system.

The final pipeline includes:

- a physics-based 2-DOF simulator;
- a 500-sample dataset generated with Latin Hypercube Sampling;
- a Gaussian Process surrogate model;
- ARD-based parameter relevance analysis;
- kernel and model comparison;
- inverse optimization using differential evolution;
- validation with the true dynamic simulator;
- sensitivity and robustness analysis.

Final numerical results:

```text
Dataset:
500 simulations
peak_y range ≈ 0.398–1.054 m
mean peak_y ≈ 0.607 m

Gaussian Process:
kernel = Matern 5/2 + WhiteKernel
Test R²   = 0.9676
Test RMSE = 21.97 mm
Test MAE  = 11.08 mm

ARD top parameters:
1. hr
2. Mb
3. Kr

Inverse optimization:
mean simulator error = 1.465 mm
max simulator error  = 3.297 mm
success rate         = 100%

Monte Carlo robustness:
noise ≈ ±5% on Kb, Mb, hb
samples = 100
mean error = 2.931 mm
max error  = 10.502 mm
success rate = 99%
```

The final framework demonstrates that Gaussian Process surrogate modeling can be used not only for prediction, but also for interpretable inverse optimization of a dynamic mechanical system.

### 11.2 Future Work

Possible future extensions include:

- Bayesian Optimization using GP uncertainty;
- robust inverse optimization that accounts for uncertainty during the optimization itself;
- closed-loop Model Predictive Control;
- multi-objective optimization including energy, time, and outreach;
- prediction of the full output trajectory instead of only peak outreach;
- nonlinear dynamics including friction, saturation, or backlash;
- experimental validation on a physical platform.

In particular, MPC would require a different model structure because the current GP predicts only the final peak outreach. A closed-loop MPC controller would need a state-space predictive model of the form:

```text
state_t, action_t → state_{t+1}
```

This is a natural but more complex extension of the current offline inverse-design framework.

---

## References

1. L. Roveda et al., “An interaction controller formulation to systematically avoid force overshoots through impedance shaping method with compliant robot base,” *Mechatronics*, vol. 39, pp. 42–53, 2016.

2. C. E. Rasmussen and C. K. I. Williams, *Gaussian Processes for Machine Learning*. MIT Press, 2006.

3. R. Storn and K. Price, “Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces,” *Journal of Global Optimization*, vol. 11, no. 4, pp. 341–359, 1997.

4. E. Schulz, M. Speekenbrink, and A. Krause, “A tutorial on Gaussian process regression,” *Journal of Mathematical Psychology*, vol. 85, pp. 1–16, 2018.

---

## Appendix A: Repository Structure

```text
project_outreach/
├── src/
│   ├── dynamics.py              # 2-DOF simulator
│   ├── dataset.py               # LHS + dataset generation
│   ├── learning_curve.py        # Dataset size analysis
│   ├── models.py                # GP training + ARD
│   ├── kernel_comparison.py     # Kernel comparison
│   ├── model_comparison.py      # GP vs Random Forest
│   ├── optimization.py          # Inverse optimization
│   └── visualization.py         # Advanced plots and robustness
├── data/
│   └── dataset_outreach.csv
├── results/
│   ├── gp_model.pkl
│   ├── scaler_X.pkl
│   ├── scaler_y.pkl
│   ├── metrics.csv
│   ├── inverse_results.csv
│   └── additional result files
├── figures/
│   └── generated figures
└── report/
    └── final report files
```

---

## Appendix B: Main Output Files

```text
data/dataset_outreach.csv

results/metrics.csv
results/ard_relevance.csv
results/kernel_comparison.csv
results/model_comparison.csv
results/inverse_results.csv
results/sensitivity_analysis.csv
results/monte_carlo_robustness.csv
results/monte_carlo_robustness_stats.csv

figures/test_simulation.png
figures/dataset_distribution.png
figures/learning_curve.png
figures/gp_parity_plot.png
figures/gp_uncertainty.png
figures/gp_ard_relevance.png
figures/kernel_comparison.png
figures/model_comparison.png
figures/inverse_targets.png
figures/sensitivity_analysis.png
figures/monte_carlo_robustness.png
figures/animation_target3.gif
```
