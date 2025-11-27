from importlib.resources import files

def getPath(fileName: str) -> str:
    return files("ICSimFEM.data").joinpath(f"{fileName}")

# Air
airElectronVelocityMagboltzSTP     = getPath(
                                    "AirElectronVelocityMagboltzSTP.csv")
airElectronDiffusionMagboltzSTP    = getPath(
                                    "AirElectronLDiffusionMagboltzSTP.csv")
airElectronAttachmenMagboltztSTP   = getPath(
                                    "AirElectronAttachmentMagboltzSTP.csv")
airElectronAttachmenMagboltzt2BSTP = getPath(
                                    "AirElectronAttachmentMagboltz2BodySTP.csv")
airElectronAttachmenMagboltzt3BSTP = getPath(
                                    "AirElectronAttachmentMagboltz3BodySTP.csv")
airElectronIonizationRateSTP       = getPath(
                                    "AirElectronIonizationRateMagboltzSTP.csv")