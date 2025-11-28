"""
    Simulate a cylindrical IC.
"""

from ICSimFEM.beams        import squaredUniformPulsedBeam
from ICSimFEM.solvers      import standardSolver
from ICSimFEM.physics      import air3SpeciesGotzFull
from ICSimFEM.timesteppers import BDF2
from ICSimFEM.chambers     import CylindricalIC1D

import numpy               as np
import matplotlib.pylab    as plt

dpps   = np.logspace(-4, 1, 50, endpoint = True)  # Gy
r2     = 1.45E-3 # m
r1     = 0.30E-3 # m
height = 2.9E-3  # m
V      = 300     # V

chamber = CylindricalIC1D("PP3D_T31022_1D", 2.326E9, r1, r2, height)
chamber.nSteps  = 400
chamber.voltage = V         
chamber.volume  = 1.56E-08 # m^3

# Beam configuration:
squaredUniformPulsedBeam.pulseDuration = 1.9E-6 # s

# Time stepping:
timeStepper = BDF2(1.0E-2)

# Physics configuration:
air3SpeciesGotzFull.temperature        = 20.0    # degC
air3SpeciesGotzFull.pressure           = 1013.25 # hPa

## *-- Solver:
standardSolver.chamber     = chamber
standardSolver.physics     = air3SpeciesGotzFull
standardSolver.beam        = squaredUniformPulsedBeam
standardSolver.timeStepper = timeStepper
standardSolver.saveData    = False
standardSolver.printScreen = True
standardSolver.reportEach  = 1000

results = np.zeros([50, 3])

n = 0
for dpp in dpps:

    chamber.voltage = abs(V)
    squaredUniformPulsedBeam.dosePerPulse  = dpp
    CCE_pos, Q = standardSolver.run(f"SimPos_{n}")

    chamber.voltage = -abs(V)
    squaredUniformPulsedBeam.dosePerPulse  = dpp
    CCE_neg, Q = standardSolver.run(f"SimNeg_{n}")

    results[n, 0] = dpp
    results[n, 1] = CCE_pos
    results[n, 2] = CCE_neg

    n += 1

fig, ax = plt.subplots(figsize = (6, 6))

ax.plot(results[:, 0], results[:, 1], "-k", linewidth = 1.5,
         label = "Positive charge")
ax.plot(results[:, 0], results[:, 2], "-r", linewidth = 1.5,
         label = "Negative charge")

ax.set_ylabel("Charge collection efficiency")
ax.set_xlabel("Dose per pulse (Gy)")

ax.set_xscale("log")

ax.tick_params(axis = 'both', which = 'major', pad = 8)

leg = ax.legend(borderpad = 0.2, fontsize = 15)
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_boxstyle('Square')

fig.tight_layout()
fig.savefig("Example1.pdf")

plt.show()