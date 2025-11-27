"""
    Implements the base class to describe beams.
"""

import dolfinx
import numpy as np

from ICSimFEM       import Logger
from ICSimFEM       import Chamber
from ICSimFEM       import Physics
from ICSimFEM.utils import constants, ktp

class Beam:

    """Class for the description of the beams. It allows to describe pulsed or
    continuous beams and potentially arbitrary space-dependent and time-dependent
    beams
    """
    
    def __init__(self, pulsed: bool = True) -> None:

        """
        Parameters
        ----------
        pulsed: bool
            Defined if the beam is pulsed or not
        """
        
        # Basic configuration. If true beam is assumed to be pulsed.
        self.pulsed = True

        # Values when beam is on and when beam if off.
        self.values_beamon  = None
        self.values_beamoff = None
        
        # Basic needed functions
        self.V         = None
        self.functions = []

    # ========== Properties ==========
    @property
    def pulsed(self) -> float:
        return self.__pulsed
    
    @pulsed.setter
    def pulsed(self, value: bool) -> None:
        self.__pulsed = value
    
    @property
    def pulseDuration(self) -> float:
        assert(self.__pulsed)
        return self.__pulseDuration
    
    @pulseDuration.setter
    def pulseDuration(self, value: float) -> None:
        assert(self.__pulsed and value >= 0)
        self.__pulseDuration = value

    @property
    def dosePerPulse(self) -> float:
        assert(self.__pulsed)
        return self.__dosePerPulse
    
    @dosePerPulse.setter
    def dosePerPulse(self, value: float) -> None:
        assert(self.__pulsed and value > 0)
        self.__dosePerPulse = value

    @property
    def doseRate(self) -> float:
        assert(not self.__pulsed)
        return self.__doseRate
    
    @doseRate.setter
    def doseRate(self, value: float) -> None:
        assert(not self.__pulsed and value > 0)
        self.__doseRate = value

    def computeReleasedCharge(self, chamber: Chamber, physics: Physics):

        """Function to compute the released charge in the medium

        Parameters
        ----------
        chamber : Chamber
            Chamber object used to get the calibration coefficient and volume.
        physics : Physics
            Physics object used to get the current temperature and pressure.
        """

        norm  = (constants.electronCharge * chamber.Ndw * chamber.volume)
        norm *= ktp(physics.temperature, physics.pressure)

        if self.__pulsed:
            self.releaseCharge = self.dosePerPulse / norm # no units
        else:
            self.releaseCharge = self.doseRate / norm     # s^{-1}

    def initialize(self, chamber: Chamber, physics: Physics):

        """Initialize the beam computing the release charge in the medium and
        update the functions at time 0 s.

        Parameters
        ----------
        chamber : Chamber
            Chamber object used to compute the released charge.
        physics : Physics
            Physics object used to compute the released charge.
        """

        self.functions = []
        self.computeReleasedCharge(chamber, physics)
        self.updateFunctions(0.0, 0.0)

    def getFunction(self, n_id, c):
        
        for element in self.functions:            
            if element[0] == n_id: return element[1]
            
        n0_t = dolfinx.fem.function.Function(self.V)
        
        self.functions.append([n_id, n0_t, c])
        
        self.updateFunctions(0.0, 0.0)
        
        return n0_t
    
    def updateFunctions(self, t: float, dt: float):
        
        for function in self.functions:
            function[1].x.array[:] = self.eval(t + dt * function[2])

    def setInitialValues(self, u_n, specie):
        if self.pulsed and self.pulseDuration == 0:
            u_n.interpolate(lambda x: self.releaseCharge * specie.fIonization * \
                             np.ones((1, x.shape[1])))
        return u_n
    
    def setSpace(self, V: dolfinx.fem.functionspace) -> None:
        
        self.V = V
        self.x = V.tabulate_dof_coordinates()
        
    def eval(self, t: float):

        if not self.__pulsed: return self.releaseCharge
        
        if (t >= self.pulseDuration): return 0.0
        else:
            return self.releaseCharge / self.pulseDuration

    def printInfo(self, logger: Logger):

        """Print the information of the beam to the output file.

        Parameters
        ----------
        logger: Logger
            Object managing the output.
        """

        logger.printHeader('*-- Beam parameters:')
        logger.printData("Beam type", "Pulsed source")
        if self.pulsed:
            logger.printData('Pulse duration', f"{self.pulseDuration:.3E} s")
            logger.printData('Dose per pulse', f"{self.dosePerPulse:.3E} Gy")
        else:
            logger.printData('Dose rate', f"{self.doseRate:.3E} Gy/s")
            
        logger.info('\n')
