"""
   Implementation of the backward differentiation scheme from:
   10.1016/j.procs.2014.05.091
"""

from ICSimFEM        import TimeStepper
from ICSimFEM        import Physics
from ICSimFEM        import Chamber
from ICSimFEM        import Beam
from ICSimFEM.Logger import Logger

import numpy as np
import dolfinx
import ufl

class BDF2(TimeStepper):

    """Implementation of the adaptative backward differentiation scheme.
    """

    def __init__(self, error):

        super().__init__("BDF2")
        self.error = error

        # Values to store past solutions
        self._dtLast = 0.0
        self._uLast  = 0.0

    def generateVariationalFormulation(self, V: dolfinx.fem.functionspace, 
                        physics: Physics, inPotential: dolfinx.fem.Function):
        
        """Generates the variational formulation using the class physics.

        Parameters
        ----------
        V : dolfinx.fem.functionspace
            Function space
        physics : Physics
            Physics class of the problem
        inPotential : dolfinx.fem.Function
            Function with the potential
        """

        super().generateVariationalFormulation(V, physics, inPotential)

        self._dtLast = 0.0
        self._uLast  = 0.0

        a, iInduced, u = physics.getVariationalForm(V, self.beam, inPotential)

        self.u    = u
        self.u_n  = dolfinx.fem.Function(V)
        self.u_1n = dolfinx.fem.Function(V)
        v         = ufl.TestFunctions(V)

        x  = self.dt / self.dt1

        c1 = (1 + x)**2 / (1 + 2 * x)
        c2 = x**2 / (1 + 2 * x)
        c3 = (1 + x) / (1 + 2 * x)

        a = c3 * a 
        for i in range(physics._nSpecies):

            a += (self.u[i] - c1 * self.u_n[i] + c2 * self.u_1n[i]) / self.dt * v[i]
        
        return a, iInduced, u
    
    def updateFunctions(self):
        super().updateFunctions()

        # Store past solutions
        self._uLast  = np.copy(self.u_1n.x.array)
        self._dtLast = np.copy(self.dt1.value)

        self.u_1n.x.array[:] = np.copy(self.u_n.x.array)
        self.u_n.x.array[:]  = np.copy(self.u.x.array)

        self.dt1.value = np.copy(self.dt.value)

        # Update the time step
        self.updateTimeStep()

        # Update the time and the corresponding functions
        self.nSteps  += 1
        self.t.value += self.dt.value

        self.beam.updateFunctions(self.t.value, 0)

    def _computeError(self):

        a = (self.u.x.array - self.u_n.x.array) / self.dt.value
        b = (1 + self.dt.value / self.dt1.value) * (self.u_n.x.array - self.u_1n.x.array) / self.dt1.value
        c = self.dt.value / (self.dt1.value * self._dtLast) * (self.u_1n.x.array - self._uLast)

        error = (self.dt.value + self.dt1.value) / 6 * (a - b + c)

        return self.releaseCharge * self.error / np.max(abs(error))

    def updateTimeStep(self):

        super().updateTimeStep()
        if self.nSteps > 3:
            LTE = self._computeError()            
            self.dt.value = self.dt.value * min(LTE**0.2, 1.50)

        self._constrainPulseDuration()

    def initialize(self, chamber: Chamber, beam: Beam):
        super().initialize(chamber, beam)
        self.dt1 = dolfinx.fem.Constant(chamber._meshData[0], 1E100)    

    def printInfo(self, logging: Logger):

        super().printInfo(logging)
        logging.info("\n")

