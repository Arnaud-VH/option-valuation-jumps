# Option Valuation under Jump-Diffusion Models

Finite difference solver for European call options under Merton's and Kou's jump-diffusion models, based on Almendral & Oosterlee (2005).

## Method

The solver discretizes the PIDE in log-price space using BDF2 time stepping and the composite trapezoidal rule for the jump integral. Analytical solutions are included for both models to validate convergence.

## Project structure

    models/          Jump-diffusion model definitions (Merton, Kou, Black-Scholes)
    solvers/         BDF2 finite difference solver
    analysis/        Convergence analysis and plotting
    tests/           Experiments and validation tests
    model_config.py  Parameters from the paper
    main.py          Entry point

## Usage

    pip install -r requirements.txt
    python main.py

This runs the full experiment suite for both models: option price plots, convergence studies, and spatial error profiles.

To run individual tests:

    python -m tests.test_merton
    python -m tests.error_infinite_norm

## Dependencies

- numpy
- scipy
- matplotlib

## Reference

Almendral, A. & Oosterlee, C.W. (2005). Numerical valuation of options with jumps in the underlying.