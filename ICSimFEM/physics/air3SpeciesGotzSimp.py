from ICSimFEM import Specie
from ICSimFEM import Physics
from .        import dataPaths

# ======================= air3SpeciesGotzFull =======================
#
# *-- Characteristic:
#     
#     Simplified physics with electric field perturbation based on the 
#     Boissonnat ion mobilities (10.48550/arXiv.1609.03740) and 
#     the volume recombination coefficients from Gotz et al. 
#     (Dosimetry of highly pulsed radiation fields, TUD)


P_ions = Specie("Positive ions", 1)
P_ions.setConstantMobilitySTP(1.87E-4)
P_ions.setNerstTowsendLDiffusion()
P_ions.fIonization = 1

N_ions = Specie("Negative ions", -1)
N_ions.setConstantMobilitySTP(2.09E-4)
N_ions.setNerstTowsendLDiffusion()
N_ions.fIonization = 0

Elec = Specie("Electrons", -1)
Elec.setDiscreteVelocityFromFile(dataPaths.airElectronVelocityMagboltzSTP)
Elec.setDiscreteLDiffusionFromFile(dataPaths.airElectronDiffusionMagboltzSTP)
Elec.fIonization = 1

# *-- Physics of the simulation:
air3SpeciesGotzSimp = Physics("air3SpeciesGotzSimp")

air3SpeciesGotzSimp.relativePermitivity = 1.000589

air3SpeciesGotzSimp.addSpecie(P_ions)
air3SpeciesGotzSimp.addSpecie(N_ions)
air3SpeciesGotzSimp.addSpecie(Elec)

air3SpeciesGotzSimp.setRecombination(1.30E-12, P_ions, N_ions)
air3SpeciesGotzSimp.setSeparateDiscreteAttachmentFromFile(
                    dataPaths.airElectronAttachmenMagboltzt2BSTP,
                    dataPaths.airElectronAttachmenMagboltzt3BSTP,
                    Elec, N_ions)

air3SpeciesGotzSimp.eFieldPerturbation = True
air3SpeciesGotzSimp.efieldFullCoupling = False
## =====================================================================
