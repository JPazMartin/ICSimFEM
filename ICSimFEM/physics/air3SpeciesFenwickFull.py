from ICSimFEM import Specie
from ICSimFEM import Physics

# ======================= air3SpeciesFenwickFull =======================
#
# *-- Characteristic:
#     
#     Full physics (without electric field perturbation) extracted from the
#     publications of Fenwick et al: 10.1088/1361-6560/ad63ed and 
#     10.1088/1361-6560/aca74e


P_ions = Specie("Positive ions", 1)
P_ions.setConstantMobilitySTP(1.87E-4)
P_ions.setNerstTowsendLDiffusion()
P_ions.fIonization = 1

N_ions = Specie("Negative ions", -1)
N_ions.setConstantMobilitySTP(2.04E-4)
N_ions.setNerstTowsendLDiffusion()
N_ions.fIonization = 0

Elec = Specie("Electrons", -1)
Elec.setConstantMobilitySTP(8.30E-2)
Elec.setNerstTowsendLDiffusion()
Elec.fIonization = 1

# *-- Physics of the simulation:
air3SpeciesFenwickFull = Physics("air3SpeciesFenwickFull")

air3SpeciesFenwickFull.relativePermitivity = 1.000589

air3SpeciesFenwickFull.addSpecie(P_ions)
air3SpeciesFenwickFull.addSpecie(N_ions)
air3SpeciesFenwickFull.addSpecie(Elec)

import numpy as np

def tau(E, E_mod):

    """
    Electron attachment rate from Fenwick et al. publications.

    Parameters
    ----------
    E : float
        Electric field vector in V/m.
    E_mod
        Electric field module in V/m

    Returns
    -------
    float
        Electron attachment rate in 1/s.
    """

    return np.piecewise(E_mod, [E_mod >= 0.327E5, E_mod < 0.327E5], 
                        [lambda x: 1.1E7 + 11.3E7 * np.exp(- 1.04E-5 * x),
                         lambda x: 7.0E7 + 657 * x])

air3SpeciesFenwickFull.setRecombination(1.30E-12, P_ions, N_ions)
air3SpeciesFenwickFull.setAttachmentFromFunction(tau, Elec, N_ions)

air3SpeciesFenwickFull.eFieldPerturbation = False
air3SpeciesFenwickFull.efieldFullCoupling = False
## =====================================================================
