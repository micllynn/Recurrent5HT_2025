# DRN Recurrent Network Simulations (Lynn et al 2025)

This repository contains code to simulate a network of recurrently connected 5-HT
neurons (inhibitory 5-HT1A connections), receiving long-range excitatory input.
It contains the full code required to generate figures from Lynn et al (2025).

Complete descriptions of the simulations can be found as docstrings in the
`sim()` and `sim_lhbpfc()` core simulation functions.

## Getting Started
The main plotting functions are found in `plot_figs.py`.
They are:
- `plot_Fig6()`
- `plot_Fig7()`
- `plot_ExtendedDataFig5()`
- `plot_ExtendedDataFig6()`
- `plot_ExtendedDataFig7()`

## Prerequisites
Requirements are: numpy, matplotlib, seaborn, and brian2.

