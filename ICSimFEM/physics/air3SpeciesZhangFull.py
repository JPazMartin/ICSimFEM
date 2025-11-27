from ICSimFEM           import Specie
from ICSimFEM           import Physics
from .                  import dataPaths
from ICSimFEM.resources import mob_pos, mob_neg

# ======================= air3SpeciesZhangFull =======================
#
# *-- Characteristic:
#     
#     Full physics with electric field perturbation based on the 
#     Zhang ion mobilities (10.1109/TDEI.2019.008001)

P_ions = Specie("Positive ions", 1)
P_ions.setConstantMobility(mob_pos)
P_ions.setNerstTowsendLDiffusion()
P_ions.fIonization = 1

N_ions = Specie("Negative ions", -1)
N_ions.setConstantMobility(mob_neg)
N_ions.setNerstTowsendLDiffusion()
N_ions.fIonization = 0

Elec = Specie("Electrons", -1)
Elec.setDiscreteVelocityFromFile(dataPaths.airElectronVelocityMagboltzSTP)
Elec.setDiscreteLDiffusionFromFile(dataPaths.airElectronDiffusionMagboltzSTP)
Elec.fIonization = 1

## *-- Physics of the simulation:
air3SpeciesZhangFull = Physics("air3SpeciesZhangFull")

air3SpeciesZhangFull.relativePermitivity = 1.000589

air3SpeciesZhangFull.addSpecie(P_ions)
air3SpeciesZhangFull.addSpecie(N_ions)
air3SpeciesZhangFull.addSpecie(Elec)

air3SpeciesZhangFull.setRecombination(0.8E-12, P_ions, N_ions)
air3SpeciesZhangFull.setSeparateDiscreteAttachmentFromFile(
                    dataPaths.airElectronAttachmenMagboltzt2BSTP,
                    dataPaths.airElectronAttachmenMagboltzt3BSTP,
                    Elec, N_ions)
air3SpeciesZhangFull.setDiscreteMultiplicationFromFile(
                    dataPaths.airElectronIonizationRateSTP, Elec, P_ions)

air3SpeciesZhangFull.eFieldPerturbation = True
air3SpeciesZhangFull.efieldFullCoupling = False
## =====================================================================
