"""
    Abstract class devoted to perform the time evolution of the simulation using
    an implicit time scheme.
"""


import dolfinx
import numpy

from ICSimFEM   import Chamber
from .Logger    import Logger
from ICSimFEM   import Beam
from ICSimFEM   import Physics


class TimeStepper:

    """Abstract devoted to perform the time evolution of the simulation using an
    implicit time scheme.
    """

    def __init__(self, name: str, error: float = 1E-4) -> str:

        """
        Parameters
        ----------
        name : str
            Name of the time stepping algorithm
        error : float
            Value to control de maximum allowed error during the time
            stepping.
        """

        self.__name = name
        self.error  = error
        self.nSteps = 0

        # Minimum number of steps during the beam
        self._nStepsDuringPulse = 400

        self.beam   = None

    @property
    def name(self) -> str:
        return self.__name
    
    @property
    def error(self):
        return self.__error
    
    @error.setter
    def error(self, value):
        assert isinstance(value, float) and value > 0
        self.__error = value

    @property
    def time(self) -> float:
        return numpy.copy(self.t.value)

    def initialize(self, chamber: Chamber, beam: Beam):

        """Initialize the class.

        Parameters
        ----------
        chamber : Chamber
            Needed to create the temporal constants.
        beam : Beam
            Used to obtain the total released charge. This value is used to 
            quantify the error an adapt the time step.
        """

        self.nSteps = 0

        self.t  = dolfinx.fem.Constant(chamber._meshData[0], 0.0)
        self.dt = dolfinx.fem.Constant(chamber._meshData[0], 1.0E-13)

        self.releaseCharge = beam.releaseCharge
        self.beam          = beam

    def generateVariationalFormulation(self, V: dolfinx.fem.functionspace,
                            physics: Physics, inPotential: dolfinx.fem.Function):
        
        """To be implemented by the derive class. Generate the variational
        formulation.

        Parameters
        ----------
        V : dolfinx.fem.functionspace
            Function space
        physics : Physics
            Physics class of the problem
        inPotential : dolfinx.fem.Function
            Function with the potential
        """

        return

    def updateFunctions(self) -> None:

        """Update the function after a step"""

        return
    
    def _computeError(self):

        return

    def updateTimeStep(self):

        return
    
    def _constrainPulseDuration(self):

        c1 = self.beam.pulseDuration > 0
        c2 = self.t.value < self.beam.pulseDuration
        if c1 and c2:
            self.dt.value = min(self.beam.pulseDuration / self._nStepsDuringPulse,
                                       self.dt.value)

        if self.dt.value > 1E-5: self.dt.value = 1E-5

    def printInfo(self, logger: Logger):

        """Prints the information of the beam to the output file.

        Parameters
        ----------
        logger: Logger
            Object managing the output.
        """

        logger.printHeader("*-- Time stepper:")
        logger.printData("Method", f"{self.name}")
        logger.printData("Error" , f"{self.error}")

        return
    
