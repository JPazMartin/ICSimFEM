from ICSimFEM import Beam

# ====================== SQUARED CONTINUOUS BEAM =========================
#
# *-- Characteristic:
#     1.- Continuous source.
#     2.- Square temporal profile.
#     3.- Homogeneous in space. 
#
continuousBeam = Beam()
continuousBeam.pulsed = False