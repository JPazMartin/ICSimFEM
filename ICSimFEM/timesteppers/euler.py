from ICSimFEM        import TimeStepper
from ICSimFEM        import Physics
from ICSimFEM.Logger import Logger

import numpy as np
import dolfinx
import ufl

class euler(TimeStepper):

    def __init__(self, error):

        super().__init__("Euler")
        self.error = error

        self._dtLast = 0.0 
        self._uLast  = 0.0

    @property
    def error(self):
        return self.__error
    
    @error.setter
    def error(self, value):
        assert isinstance(value, float) and value > 0
        self.__error = value

    def generateVariationalFormulation(self, V: dolfinx.fem.functionspace,
                                        physics: Physics, inPotential):
        super().generateVariationalFormulation(V, physics, inPotential)

        a, iInduced, u = physics.getVariationalForm(V, self.beam, inPotential)

        self.u   = u
        self.u_n = dolfinx.fem.Function(V)
        v        = ufl.TestFunctions(V)

        for i in range(physics._nSpecies):
            a += (self.u[i] - self.u_n[i]) / self.dt * v[i]

        return a, iInduced, u
    
    def updateFunctions(self):
        super().updateFunctions()

        self._uLast  = np.copy(self.u_n.x.array)
        self._dtLast = np.copy(self.dt.value)

        self.u_n.x.array[:] = self.u.x.array

        self.updateTimeStep()

        self.nSteps  += 1
        self.t.value += self.dt.value

        self.beam.updateFunctions(self.t.value, 0)
        
    def _computeError(self):

        a = 2 * self._dtLast / (self._dtLast + self.dt.value) * self.u.x.array - 2 * self.u_n.x.array
        b = 2 * self.dt.value / (self._dtLast + self.dt.value) * self._uLast

        error = self.dt.value / (2 * self._dtLast) * (a + b)

        return self.releaseCharge * self.error / np.max(abs(error))
    
    def updateTimeStep(self):
        super().updateTimeStep()
        if self.nSteps > 3:
            LTE = self._computeError()
            newTimeStep   = self.dt.value * min((0.98 * np.max(LTE))**0.5, 3)
            self.dt.value = newTimeStep

        self._constrainPulseDuration()

    def printInfo(self, logging: Logger):

        super().printInfo(logging)
        logging.info("\n")

