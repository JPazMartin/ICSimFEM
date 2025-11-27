from ICSimFEM import Specie
from ICSimFEM import Physics

# ========================== air2SpeciesFenwick ==========================
#
# *-- Characteristic:
#     
#     2 species physics (without electric field perturbation) extracted
#     from the publications of Fenwick et al: 10.1088/1361-6560/ad63ed and 
#     10.1088/1361-6560/aca74e

P_ions = Specie("Positive ions", 1)
P_ions.setConstantMobilitySTP(1.87E-4)
# P_ions.setNerstTowsendLDiffusion()
P_ions.fIonization = 1

N_ions = Specie("Negative ions", -1)
N_ions.setConstantMobilitySTP(2.09E-4)
#N_ions.setNerstTowsendLDiffusion()
N_ions.fIonization = 1

# *-- Physics of the simulation:
air2SpeciesFenwick = Physics("air2SpeciesFenwick")

air2SpeciesFenwick.relativePermitivity = 1.000589
air2SpeciesFenwick.addSpecie(P_ions)
air2SpeciesFenwick.addSpecie(N_ions)

air2SpeciesFenwick.setRecombination(1.30E-12, P_ions, N_ions)

air2SpeciesFenwick.eFieldPerturbation = False
air2SpeciesFenwick.efieldFullCoupling = False
## =====================================================================
