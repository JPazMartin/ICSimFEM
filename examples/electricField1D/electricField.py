"""
    Simulate the voltage and electric field for a cylindrical, spherical and
    parallel plate geometry and compare them to its expected magnitude.
"""

import matplotlib.pylab as plt
import numpy            as np

from ICSimFEM.chambers import ParallelPlateIC1D, CylindricalIC1D, SphericalIC1D

nSteps  = 1000
voltage = 100    # V
d       = 1.0E-3 # m
r1      = 0.3E-3 # m
r2      = 1.0E-3 # m

# *-- 1D Parallel-plate IC:
# Construct a 1D parallel-plate IC with 1.0 mm distance between electrodes.
chamber_pp         = ParallelPlateIC1D("PPIC", 1.0, d, 1.0)
chamber_pp.voltage = voltage
chamber_pp.nSteps  = nSteps
V_pp, E_pp         = chamber_pp.computeElectricField()

# *-- 1D Cylindrical IC:
# Construct a 1D Cylindrical IC with internal radius of 0.3 mm and external 
# radius of 1.0 mm
chamber_cic         = CylindricalIC1D("CIC", 1.0, r1, r2, 1.0)
chamber_cic.voltage = voltage
chamber_cic.nSteps  = nSteps
V_cic, E_cic        = chamber_cic.computeElectricField()

# *-- 1D Spherical IC:
# Construct a 1D Cylindrical IC with internal radius of 0.3 mm and external 
# radius of 1.0 mm
chamber_sic         = SphericalIC1D("SIC", 1.0, r1, r2, 1.0)
chamber_sic.voltage = voltage
chamber_sic.nSteps  = nSteps
V_sic, E_sic        = chamber_sic.computeElectricField()


# Analytical expressions:
EppTh  = lambda x: -voltage / 1.0E-3 / 1E6 * np.ones(x.shape)     # kV/mm
EsicTh = lambda r: -voltage / (r**2 * (r1**-1 - r2**-1)) / 1E6    # kV/mm
EcicTh = lambda r: -voltage / (r * np.log(r2 / r1)) / 1E6         # kV/mm
VppTh  = lambda x: voltage / d * x                                # V
VcicTh = lambda r: voltage / (np.log(r2 / r1)) * np.log(r / r1)   # V
VsicTh = lambda r: voltage / (r1**-1 - r2**-1) * (1 / r1 - 1 / r) # V

fig, ax = plt.subplots(3, 3, figsize = (13, 8), 
                       gridspec_kw = {'height_ratios': [2, 2, 1]})

# --> Parallel-plate geometry
ax[0, 0].set_title("Parallel-plate geometry")

x      = E_pp._V.mesh.geometry.x[:, 1]                   # m
EppSim = E_pp.x.array.reshape(nSteps + 1, 3)[:, 1] / 1E6 # kV/mm

ax[0, 0].plot(x * 1E3, EppTh(x),  "-b", linewidth = 1.5, label = "Analytical")
ax[0, 0].plot(x * 1E3,   EppSim, "--k", linewidth = 2.5, label = "Simulation")
ax[0, 0].plot(0, 0)

VppSim = V_pp.x.array # V
ax[1, 0].plot(x * 1E3, VppTh(x),  "-r", linewidth = 1.5, label = "Analytical")
ax[1, 0].plot(x * 1E3,   VppSim, "--k", linewidth = 2.5)

residualsE = (EppTh(x) - EppSim) / EppTh(x) * 100
ax[2, 0].plot(x * 1E3, residualsE, "-b")

residualsV = (VppTh(x[1:]) - VppSim[1:]) / VppTh(x[1:]) * 100
ax[2, 0].plot(x[1:] * 1E3, residualsV, "-r")

# --> Cylindrical geometry
ax[0, 1].set_title("Cylindrical geometry")

# The derivate is associated to the middle of the cell.
r       = np.convolve(E_cic._V.mesh.geometry.x[:, 0], [0.5, 0.5],
                       mode = "valid")                            # m
EcicSim = E_cic.x.array.reshape(nSteps + 1, 3)[:-1, 0] / 1E6      # kV/mm

ax[0, 1].plot(r * 1E3,  EcicTh(r),  "-b", linewidth = 1.5)
ax[0, 1].plot(r * 1E3,    EcicSim, "--k", linewidth = 2.5)
ax[0, 1].plot(0, 0)

residualsE = (EcicTh(r) - EcicSim) / EcicTh(r) * 100
ax[2, 1].plot(r * 1E3, residualsE, "-b")

r       = V_cic._V.mesh.geometry.x[:, 0] # m
VcicSim = V_cic.x.array                  # V

ax[1, 1].plot(r * 1E3,  VcicTh(r),  "-r", linewidth = 1.5)
ax[1, 1].plot(r * 1E3,    VcicSim, "--k", linewidth = 2.5)

residualsV = (VcicTh(r[1:]) - VcicSim[1:]) / VcicTh(r[1:]) * 100
ax[2, 1].plot(r[1:] * 1E3, residualsV, "-r")

# --> Spherical geometry
ax[0, 2].set_title("Spherical geometry")

# The derivate is associated to the middle of the cell.
r       = np.convolve(E_sic._V.mesh.geometry.x[:, 0], [0.5, 0.5],
                       mode = "valid")                            # m
EsicSim = E_sic.x.array.reshape(nSteps + 1, 3)[:-1, 0] / 1E6      # kV/mm

ax[0, 2].plot(r * 1E3,  EsicTh(r),  "-b", linewidth = 1.5)
ax[0, 2].plot(r * 1E3,    EsicSim, "--k", linewidth = 2.5)
ax[0, 2].plot(0, 0)

residualsE = (EsicTh(r) - EsicSim) / EsicTh(r) * 100
ax[2, 2].plot(r * 1E3, residualsE, "-b")

r       = V_sic._V.mesh.geometry.x[:, 0]                           # m
VsicSim = V_sic.x.array                                            # V

ax[1, 2].plot(r * 1E3,  VsicTh(r),  "-r", linewidth = 1.5)
ax[1, 2].plot(r * 1E3,    VsicSim, "--k", linewidth = 2.5)

residualsV = (VsicTh(r[1:]) - VsicSim[1:]) / VsicTh(r[1:]) * 100
ax[2, 2].plot(r[1:] * 1E3, residualsV, "-r")

ax[0, 0].set_ylabel("Electric field (kV/mm)")
ax[1, 0].set_ylabel("Voltage (V)")
ax[2, 0].set_ylabel("Residuals (\%)")

[ax[2, i].set_xlabel("Distance (mm)") for i in range(3)]
[ax[0, i].set_ylim([ 0.0, -0.5]) for i in range(3)]
[ax[1, i].set_ylim([ 0.0,  110]) for i in range(3)]
[ax[j, i].set_xlim([-0.1,  1.1]) for i in range(3) for j in range(3)]

leg = ax[0, 0].legend(borderpad = 0.2, fontsize = 15)
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_boxstyle('Square')

leg = ax[1, 0].legend(borderpad = 0.2, fontsize = 15)
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_boxstyle('Square')

fig.tight_layout()
fig.savefig("electricFieldSimulation.pdf")
plt.show()
