# ML-Based Inverse Optimization for Extra-Reach Planning of a Compliant-Base Robot

This project studies an inverse-optimization problem for a robot mounted on a compliant base.  
The goal is to increase the absolute horizontal outreach

\[
y(t)=x_b(t)+x_r(t)
\]

while respecting the robot workspace constraint:

\[
\max_t |x_r(t)| \leq 0.500 \text{ m}
\]

The project combines dynamic simulation, dataset generation, surrogate modeling, inverse optimization, physical validation, and robustness analysis.

---

## Project Overview

The system is modeled as a two-mass spring-damper system.  
A chirp excitation is applied to the robot, and the compliant base motion is exploited to obtain extra reach beyond the nominal robot limit of 0.500 m.

The inverse optimization acts on the controllable parameters:

```text
Kr, hr, f0, f1, A, x_r_start
```

with fixed physical parameters:

```text
Kb = 1000
Mb = 20
hb = 0.10
Mr = 10
```

The main simulated outputs are:

```text
peak_y       = maximum absolute outreach
max_abs_xr   = maximum robot relative displacement
```

---

## Methods

The project compares four optimization strategies:

```text
GP+DE          Gaussian Process surrogate + Differential Evolution
NN+Adam        Neural Network surrogate + multi-start gradient optimization
BO             Bayesian Optimization on the true simulator
Random Search  same-budget baseline
```

The surrogate models are trained on an augmented simulation dataset, with additional samples focused on high-outreach feasible regions.

---

## Main Results

All final methods found feasible solutions for the tested targets.

The best overall compromise was obtained by **NN+Adam**, which achieved:

```text
Mean target error:       about 20.4 mm
Feasibility rate:        100%
Minimum residual margin: about 3 mm
Optimization time:       about 7 s total
```

For reachable targets up to 0.65 m, NN+Adam reached very low tracking error:

```text
Target 0.65 m:
achieved peak_y ≈ 0.650 m
target error ≈ 0.3 mm
max_abs_xr ≈ 0.490 m
```

For the saturated high target 0.75 m, the robot constraint limited the achievable outreach:

```text
Target 0.75 m:
achieved peak_y ≈ 0.674 m
target error ≈ 75.7 mm
max_abs_xr ≈ 0.497 m
```

This shows that the system can generate significant extra reach, but the 0.500 m robot workspace constraint becomes active for very high targets.

---

## Project Structure

```text
ML_extra_reach_project/
├── src/
│   ├── dynamics.py
│   ├── dataset.py
│   ├── augment_high_outreach.py
│   ├── model_peak_y.py
│   ├── model_max_xr.py
│   ├── model_nn.py
│   ├── model_comparison.py
│   ├── optimization_gp_de.py
│   ├── optimization_nn_gradient.py
│   ├── optimization_bo.py
│   ├── bo_benchmark.py
│   ├── bo_kernel_benchmark.py
│   ├── bo_alpha_benchmark.py
│   ├── bo_budget_sweep.py
│   ├── bo_final_run.py
│   ├── final_optimization_comparison.py
│   ├── physical_validation.py
│   ├── sensitivity_oat.py
│   ├── monte_carlo_robustness.py
│   └── visualize_optimized_motion.py
├── data/
├── results/
├── figures/
├── report/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

For MP4 animations, `ffmpeg` is required.  
GIF animations work without additional system dependencies.

---

## Recommended Execution Order

### 1. Dataset generation

```bash
python src/dataset.py
python src/augment_high_outreach.py
```

### 2. Surrogate model training

```bash
python src/model_peak_y.py
python src/model_max_xr.py
python src/model_nn.py
python src/model_comparison.py
```

### 3. Inverse optimization

```bash
python src/optimization_gp_de.py
python src/optimization_nn_gradient.py
```

### 4. Bayesian Optimization benchmark and final run

```bash
python src/bo_benchmark.py
python src/bo_kernel_benchmark.py
python src/bo_alpha_benchmark.py
python src/bo_budget_sweep.py
python src/bo_final_run.py
```

### 5. Final comparison and validation

```bash
python src/final_optimization_comparison.py
python src/physical_validation.py
```

### 6. Sensitivity and robustness

```bash
python src/sensitivity_oat.py
python src/monte_carlo_robustness.py
```

---

## Motion Visualization

Example:

```bash
python src/visualize_optimized_motion.py --method nn --target 0.65 --format gif
```

Other supported methods:

```bash
--method gpde
--method bo
--method random
```

Animations are saved in:

```text
figures/visualization/
```

---

## Main Output Folders

```text
results/
├── gp/
├── gp_constraints/
├── nn/
├── optimization_gp_de/
├── optimization_nn_gradient/
├── optimization_bo_final/
├── final_optimization_comparison/
├── physical_validation/
├── sensitivity_oat/
└── monte_carlo_robustness/

figures/
├── gp/
├── gp_constraints/
├── nn/
├── optimization_gp_de/
├── optimization_nn_gradient/
├── optimization_bo_final/
├── final_optimization_comparison/
├── physical_validation/
├── sensitivity_oat/
├── monte_carlo_robustness/
└── visualization/
```

---

## Summary

This repository implements a complete ML-based inverse optimization workflow for extra-reach planning of a compliant-base robot.

The final results show that surrogate-based optimization can identify feasible excitation parameters that exploit passive base motion to exceed the nominal robot reach, while maintaining the robot relative displacement within the required workspace constraint.

All final scripts use cleaned folder names without `_v2`.
