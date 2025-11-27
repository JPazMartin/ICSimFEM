from ICSimFEM import Beam
from ICSimFEM import Chamber
from ICSimFEM import Physics
from ICSimFEM import Logger
from ICSimFEM import constants, ktp

import numpy as np

# ====================== ARBITRARY PULSE BEAM =========================
#
# *-- Characteristic:
#     1.- Pulsed source.
#     2.- Arbitrary temporal shape profile.
#     3.- Homogeneous in space. 
#

class ArbitraryPulseShapeBeam(Beam):

    """A beam with an arbitrary temporary shape.
    """

    def __init__(self):

        super().__init__(pulsed = True)

        self.timeArray     = [] # s   . Array with the time.
        self.doseRateArray = [] # Gy/s. Array with the dose rate.

    def setDoseRate(self, x: np.array, y: np.array):

        """Set the dose-rate profile of the beam

        Parameters
        ----------
        x : np.array
            Time component of the dose-rate profile in s.
        y : np.array
            Corresponding dose-rate profile in Gy/s
        """
        
        assert len(x) == len(y), "Time and dose rate should have the same dimension"
        assert np.all(np.diff(x) > 0), "Time must be in increasing order"

        self.timeArray     = x
        self.doseRateArray = y

        # Calculate the dose per pulse
        self.dosePerPulse  = np.trapezoid(y, x)

        # The pulse duration here is assumed to be the last time present in the 
        # given array
        self.pulseDuration = x[-1]

        # Negative values are not allowed. If so,they are force to be zero.
        self.doseRateArray[self.doseRateArray < 0] = 0

    def computeReleasedCharge(self, chamber: Chamber, physics: Physics):

        """Function to compute the released charge in the medium

        Parameters
        ----------
        chamber : Chamber
            Chamber object used to get the calibration coefficient and volume.
        physics : Physics
            Physics object used to get the current temperature and pressure.
        """
        
        norm  = constants.electronCharge * chamber.Ndw * chamber.volume
        norm *= ktp(physics.temperature, physics.pressure)

        self.releaseChargeArray = self.doseRateArray / norm # no units
        self.releaseCharge      = np.trapezoid(self.releaseChargeArray, self.timeArray)

    def eval(self, t: float):

        """Gives the value of the dose rate at a certain time in s. It performs a 
        linear interpolation of the previous given values.

        Parameters
        ----------
        t: float
            Time in seconds.

        Returns
        -------
        float
            Value of the dose rate at a time t.
        """

        # Linear interpolation between values.
        return np.interp(t, self.timeArray, self.releaseChargeArray)
        
    def printInfo(self, logger: Logger):

        """Print the information of the beam to the output file.

        Parameters
        ----------
        logger: Logger
            Object managing the output.
        """

        logger.printHeader('*-- Beam parameters:')
        logger.printData("Beam type", "Pulsed source")
        logger.printData('Pulse duration', f"{self.pulseDuration:.3E} s")
        logger.printData('Dose per pulse', f"{self.dosePerPulse:.3E} Gy")
 
        logger.info('\n')
    
arbitraryPulseShapeBeam = ArbitraryPulseShapeBeam()
