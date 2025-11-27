# Ionization chamber simulation using finite elements
Code to simulate the response of ionization chambers using the finite element method. It allows to simulate 1D and 2D geometries. A simple example on how to use the code can be found in examples/

Further updates to include new features are expected. The code will be ported to dolfinx 0.10.0.

## Install the code
To run the code, first install dolfinx (version 0.9.0). Refer to github.com/FEniCS/dolfinx.git

Then, install the code by downloading the module:
```
git clone https://github.com/JPazMartin/ICSimulation.git
```
and running:
```
pip install .
```
inside the package folder.

## References

Paz-Martín, J., Schüller, A., Bourgouin, A., González-Castaño, D. M.,Gómez-Fernández, N., Pardo-Montero, J., & Gómez, F. (2022). Numerical modeling of air-vented parallel plate ionization chambers for ultra-high dose rate applications. Physica medica, 103, 147–156. https://doi.org/10.1016/j.ejmp.2022.10.006
