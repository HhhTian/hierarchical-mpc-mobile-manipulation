# HTMPC-Minimal: A Simplified Reproduction of Hierarchical Task MPC

This project is a minimal simulation of a mobile manipulator controlled by a hierarchical MPC (HTMPC) scheme, based on the paper:

**"Hierarchical Task Model Predictive Control for Sequential Mobile Manipulation Tasks"**

## 🌟 Project Goals

- Reproduce key ideas of HTMPC using a simplified 2D mobile manipulator
- Implement a two-level task controller: 
  - T0: Keep the end-effector (EE) at a fixed point
  - T1: Track a desired base trajectory
- Compare with a single-task (ST) architecture baseline

## 🏗️ Project Structure

- controller/
  - model.py - Double-integrator robot model
  - mpc.py - CasADi-based MPC controller
  - planner.py - Reference trajectory generator
- simulation/
  - simulate.py - Main simulation loop
  - visualizer.py - Plotting EE/base paths and errors
  - compare_st_ht.py - ST vs HTMPC comparison
- results/ - Figures and animations

## 🧪 Features

- CasADi-based MPC with lexicographic task structure
- Multi-task tracking with task priority
- Comparison with single-task baseline controller
- Modular simulation framework

## 📦 Install

```bash
pip install -r requirements.txt
```

## Install the project as a local package (optional but recommended)

```bash
pip install -e .
```

## Run the simulation

```bash
python simulation/simulate.py
```

## 📽️ Demo

- Coming soon – animated comparison between HTMPC and ST control.

## 📚 Reference

**Hierarchical Task Model Predictive Control for Sequential Mobile Manipulation Tasks**  
Xintong Du, Siqi Zhou, Angela P. Schoellig (2024)
