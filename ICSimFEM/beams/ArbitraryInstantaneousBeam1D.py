from ICSimFEM import Beam
from ICSimFEM import Chamber
from ICSimFEM import Physics
from ICSimFEM import Logger
from typing   import Callable

import numpy as np
import dolfinx

# ====================== ARBITRARY INSTANTANEOUS BEAM 1D =========================
#
#  Beam that reseases an arbitrary charge profile inside the ionization chamber.
#
# *-- Characteristic:
#     1.- Pulsed source.
#     2.- Square temporal profile.
#     3.- Constant temporal profile.

class ArbitraryInstantaneousBeam1D(Beam):

    """A beam with an inhomogeneous distribution in space but with a square 
    temporal profile in time.
    """

    def __init__(self):

        super().__init__(pulsed = True)

        # This beam has a fixed pulse duration of 0 us. This can be adapted in 
        # the future if needed.
        self.__pulseDuration      = 0.0
        self.__chargeDistibutions = []
        self.__speciesName        = []

        self.__pulsed = True

    @property
    def pulseDuration(self) -> float:
        assert(self.__pulsed)
        return self.__pulseDuration
    
    @pulseDuration.setter
    def pulseDuration(self, value: float) -> None:
        raise Exception("Pulse duration must be 0.0 for this pulse beam")

    def setChargeDistribution(self, specie: str, function: Callable[[float,
                                                      float, float], float]):

        """
        Set the desired charge distribution in the simulation
        Parameters
        ----------
        specie : str
            Name of the specie to set the charge distribution.
        function : Callable
            Function of x y z (in order) with the desired charge
            distribution
        """
        self.__chargeDistibutions.append([specie, function])

    def initialize(self, chamber: Chamber, physics: Physics):

        """Reimplements the initialization of the Beam class to load the name
        of the species
        """

        self.functions = []
        self.computeReleasedCharge(chamber, physics)
        self.updateFunctions(0.0, 0.0)

        # Save the name and the other of the species
        for specie in physics._species:
            self.__speciesName.append(specie)

    def setInitialValues(self, u_n: dolfinx.fem.Function, specie: str):

        factor = lambda x: self.releaseCharge * specie.fIonization * \
              np.ones((1, x.shape[1]))

        function = factor
        for value in self.__chargeDistibutions:
            if value[0] == specie.name:
                function = lambda x: value[1](np.array(x[0]), np.array(x[1]),
                                               np.array(x[2])) * factor(x)
        u_n.interpolate(function)

        return u_n
        
    def printInfo(self, logger: Logger):

        logger.printHeader('*-- Beam parameters:')
        logger.printData("Beam type", "Arbitrary instanteous beam")
        logger.printData('Pulse duration', "0.0 s")
        logger.printData('Dose per pulse', f"{self.dosePerPulse:.3E} Gy")
 
        logger.info('\n')

arbitraryInstantaneousBeam1D = ArbitraryInstantaneousBeam1D()