"""
    Class to describe the physics used in the simulation of the 
    ionization chambers.
"""

import ufl
import dolfinx
import numpy as np

from ICSimFEM.utils   import readFile, constants, ktp
from ICSimFEM         import Specie
from ICSimFEM         import Beam
from ICSimFEM         import Logger
from typing           import Callable

class Physics:
    
    """
    Class to describe the physics used in the simulation of the 
    ionization chambers.
    """
    
    def __init__(self, physicsName: str) -> None:
        
        """
        Parameters
        ----------
        physicsName : str
            Name given to the physics                  
        """
        
        # Number of _species considered in this physics model
        self.physicsName = physicsName
        
        # Activate or not the electric field perturbation and simulate using a 
        # full copupling. For time-dependent problems the code becomes slower 
        # and no significant differences were observed. For stationary problems
        # it is mandatory to perform full coupling to simulate the electric
        # field perturbation.
        self.eFieldPerturbation = True
        self.efieldFullCoupling = False 
        
        # Symmetry of the problem (this value will be setter by solver)
        self.symmetry = "Cartesian"
        
        # Constants
        self.relativePermitivity = 1

        # Ambiental parameters
        self.temperature = 20       # ºC
        self.pressure    = 1013.25  # hPa
        self.rhumidity   = 50       # %
        
        # Class properties
        self._recombination  = []
        self._attachment     = []
        self._multiplication = []
        self._nSpecies       = 0
        self._species        = []
        self._speciesName    = []
        self._functions      = []

        # Ignore that the fraction of ionization must sum 2.000. Only 
        # for debugging purposes.
        self.ignoreIonSum    = False
        
        return
    
    @property
    def relativePermitivity(self) -> float:
        return self.__relativePermitivity
    
    @relativePermitivity.setter
    def relativePermitivity(self, value: float):
        assert value > 0
        self.__relativePermitivity = value

    @property
    def temperature(self) -> float:
        return self.__temperature
    
    @temperature.setter
    def temperature(self, value : float):
        self.__temperature = value

    @property
    def pressure(self) -> float:
        return self.__pressure
    
    @pressure.setter
    def pressure(self, value : float):
        assert value > 0
        self.__pressure = value

    @property
    def rhumidity(self) -> float:
        return self.__rhumidity
    
    @rhumidity.setter
    def rhumidity(self, value : float):
        assert value >= 0 and value <= 100
        self.__rhumidity = value

    @property
    def symmetry(self):
        return self.__symmetry
    
    @symmetry.setter
    def symmetry(self, value: str):
        assert value in ["Cartesian", "Cylindrical", "Spherical"], "Symmetry" 
        "must me Cartesian, Cylindrical or Spherical"
        self.__symmetry = value

    @property
    def eFieldPerturbation(self):
        return self.__eFieldPerturbation

    @eFieldPerturbation.setter
    def eFieldPerturbation(self, value: bool):
        self.__eFieldPerturbation = value

    @property
    def efieldFullCoupling(self):
        return self.__efieldFullCoupling

    @efieldFullCoupling.setter
    def efieldFullCoupling(self, value: bool):
        self.__efieldFullCoupling = value

    @property
    def ignoreIonSum(self):
        return self.__ignoreIonSum

    @ignoreIonSum.setter
    def ignoreIonSum(self, value: bool):
        self.__ignoreIonSum = value

    def generateFunctions(self, V1: dolfinx.fem.functionspace,
                           V2: dolfinx.fem.functionspace) -> None:
        
        """Generate the functions needed to include the velocity, diffusion, 
        attachment and multiplication

        Parameters
        ----------
        V1 : dolfinx.fem.functionspace
            Scalar function space.
        V2: dolfinx.fem.functionspace
            Vector function space.        
        """

        # Empty the array of functions
        self._functions     = []
        ambientalConditions = [self.rhumidity, self.pressure, self.temperature]
        
        for specie in self._species:
            
            # *-- Velocity (only if mobility is dependent of the electric field)
            if specie.mobType == 3:
                f_v = specie.getVelocity(*ambientalConditions)
                vel = dolfinx.fem.function.Function(V2)
                self._functions.append(["velocity", specie, f_v, vel])
                
            # *-- Longitudinal diffusion
            if specie.diffType != False:
                f_d = specie.getDiffusion(*ambientalConditions)
                d1  = dolfinx.fem.function.Function(V1)
                d2  = dolfinx.fem.function.Function(V1)
                d3  = dolfinx.fem.function.Function(V1)
                self._functions.append(["diffusion", specie, f_d, [d1, d2, d3]])
                
        # *-- Attachment
        i = 0
        for process in self._attachment:
            if process[-1] == 1:
                attx = dolfinx.fem.function.Function(V1)
                self._functions.append(["attachment", i, process[0], attx])
                i += 1

        # *-- Multiplication
        i = 0
        for process in self._multiplication:
            attx = dolfinx.fem.function.Function(V1)
            self._functions.append(["multiplication", i, process[0], attx])
            i += 1

        self.electricField = dolfinx.fem.Function(V2)
    
    def updateTransportParameters(self, electricField: dolfinx.fem.Function):

        """Update the electric-field dependent transport parameters
        
        Parameters
        ----------
        electricField: dolfinx.fem.Function
            Function with the values of the electric field
        """

        self.electricField.interpolate(electricField)

        # Calculate the module of the electric field
        E      = self.electricField.x.array.reshape(-1, 3).T
        EMod   = np.linalg.norm(E, axis = 0)

        # Update values
        for value in self._functions:
            if type(value[3]) == list:
                res = value[2](E, EMod)
                for i in range(len(value[3])): value[3][i].x.array[:] = res[i]
            else:
                value[3].x.array[:] = value[2](E, EMod)
    
    def addSpecie(self, specie: Specie) -> None:
        
        """Add specie to the simulation
        
        Parameters
        ----------
        specie: Specie
            Specie to be add
        """

        self._nSpecies += 1
        self._species.append(specie)
        self._speciesName.append(specie.name)
        
        return
    
    def setRecombination(self, coef, specie1: Specie, specie2: Specie) -> None:
        
        """Add a recombination term between specie1 and specie2

        Parameters
        ----------
        specie1 : Specie
            Specie that recombines with specie2

        specie2 : Specie
            Specie that recombines with specie1
        """

        assert specie1 in self._species, f"{specie1.name} is not defined in physics"
        assert specie2 in self._species, f"{specie2.name} is not defined in physics"

        self._recombination.append([coef, specie1, specie2])
        
        return
    
    def setAttachment(self, coef, specie1: Specie, specie2: Specie) -> None:
        
        """Add attachment from specie1 to specie2

        Parameters
        ----------
        specie1 : Specie
            Specie that attach to form specie2

        specie2 : Specie
            Specie formed by attachment of specie1   
        """
        
        assert specie1 in self._species, f"{specie1.name} is not defined in physics"
        assert specie2 in self._species, f"{specie2.name} is not defined in physics"

        ## Check if _species exist!
        self._attachment.append([lambda x0, x1: coef, specie1, specie2, 1])
        
        return
    
    def setDiscreteAttachment(self, xValues: np.array, yValues: np.array,
                               specie1: Specie, specie2: Specie) -> None:
        
        """Add a electric field dependent attachment from specie1 to specie2

        Parameters
        ----------
        xValues : np.array
            Values of the electric field in V/m
        yValues : np.array
            Values of the attachment rate in s^{-1}.
        specie1 : Specie
            Specie that attach to form specie2
        specie2 : Specie
            Specie formed by attachment of specie1   
        """
        
        def attx(E, E_mod): return np.interp(E_mod, xValues, yValues)

        self.xAttx = xValues
        self.yAttx = yValues

        assert specie1 in self._species, f"{specie1.name} is not defined in physics"
        assert specie2 in self._species, f"{specie2.name} is not defined in physics"

        self._attachment.append([attx, specie1, specie2, 1])
        
        return
    
    def setDiscreteAttachmentFromFile(self,fileName: str, specie1: Specie,
                                       specie2: Specie) -> None:
        
        """Add a electric field dependent attachment from specie1 to specie2 
        using a file. It is asumed that the file contain the attachment time
        in s.

        Parameters
        ----------
        filename : str
            File where the attachment is.
        specie1 : Specie
            Specie that attach to form specie2
        specie2 : Specie
            Specie formed by attachment of specie1   
        """

        xValues, yValues = readFile(fileName)
        self.setDiscreteAttachment(xValues, 1 / yValues, specie1, specie2)
        
        return
    
    def setSeparateDiscreteAttachmentFromFile(self,
                                              fileName2B: str,
                                              fileName3B: str,
                                              specie1: Specie,
                                              specie2: Specie) -> None:
        
        """Add a electric field dependent attachment (separted into 3-body 
        attachment and two-body attachment from specie1 to specie2 using a file.
        It is asumed that the file contain the attachment time in s in the file.
        This function corrects the attachment by pressure and temperature.

        Parameters
        ----------
        filename2B : str
            File where the 2-body attachment is.
        filename3B : str
            File where the 3-body attachment is.
        specie1 : Specie
            Specie that attach to form specie2
        specie2 : Specie
            Specie formed by attachment of specie1   
        """

        xValues2B, yValues2B = readFile(fileName2B)
        xValues3B, yValues3B = readFile(fileName3B)

        assert np.all(xValues2B == xValues3B), "Different electric field between"
        " files"

        def attx(E, E_mod):

            corr = ktp(self.temperature, self.pressure)
            return np.interp(E_mod, xValues2B / corr, 
                             1 / (yValues2B * corr) + 1 / (yValues3B * corr**2))
        
        self._attachment.append([attx, specie1, specie2, 1])
        
        return
    
    def setAttachmentFromFunction(self, function: Callable[[float, float], float],
                                   specie1, specie2) -> None:
        
        """Add a electric field dependent attachment from specie1 to specie2 
        using a user-defined function.

        Parameters
        ----------
        function : Callable[[float, float], float]
            Function of the electric field vector and module.
        specie1 : Specie
            Specie that attach to form specie2
        specie2 : Specie
            Specie formed by attachment of specie1   
        """
        
        assert specie1 in self._species, f"{specie1.name} is not defined in physics"
        assert specie2 in self._species, f"{specie2.name} is not defined in physics"

        self._attachment.append([function, specie1, specie2, 1])
        
        return
    
    def setDiscreteMultiplication(self, xValues: np.array, yValues: np.array,
                                   specie1: Specie, specie2: Specie) -> None:
        
        """Add multiplication of specie 1 and specie2. The multiplication 
        coefficient is assumed to be given at reference pressure and temperature.
        
        Parameters
        ----------
        xValues : np.array
            Value of the electric field strength in V/m
        yValues : np.array
            Value of multiplication coefficient in 1/s.
        specie1 : Specie
            Specie that is realeased in a multiplication collision
        specie1 : Specie
            Specie that is realeased in a multiplication collision
        """
        
        def mult(E, E_mod):
            corr = ktp(self.temperature, self.pressure)
            return np.interp(E_mod, xValues / corr, yValues / corr)
        
        assert specie1 in self._species, f"{specie1.name} is not defined in physics"
        assert specie2 in self._species, f"{specie2.name} is not defined in physics"

        self._multiplication.append([mult, specie1, specie2])
        
        return
    
    def setDiscreteMultiplicationFromFile(self,fileName: str, specie1: Specie,
                                           specie2: Specie) -> None:
        
        """Add multiplication of specie 1 and specie2. The multiplication 
        coefficient is assumed to be given at reference pressure and temperature.
        
        Parameters
        ----------
        fileName : str
            Name of the file where the multipliction coefficient is in 1/s.
        specie1 : Specie
            Specie that is realeased in a multiplication collision
        specie1 : Specie
            Specie that is realeased in a multiplication collision
        """

        xValues, yValues = readFile(fileName)
        self.setDiscreteMultiplication(xValues, yValues, specie1, specie2)
        
        return

    def getIdx(self, specie: Specie):

        """Returns the index of the specie in the list of species

        Parameters
        ----------
        specie: Specie
            Specie to be returned

        Returns
        -------
        int
            Index in the species list
        """
        
        return self._species.index(specie)
    
    def _getMyFunction(self, n_type, specie):
        
        for function in self._functions:
            
            if function[0] == n_type and function[1] == specie:
                
                return function[3]
            

    def getVariationalForm(self, V: dolfinx.fem.functionspace, beam: Beam,
                            uE: dolfinx.fem.function)-> list[ufl.algebra.Sum, 
                                                             list, 
                                                             dolfinx.fem.function]:
        
        """For the given physics, return the variational formulation of the 
        transport equations.

        Parameters
        ----------
        V: dolfinx.fem.functionspace
            Scalar function space of the problem
        beam: Beam
            Beam for the simulation
        uE: dolfinx.fem.function
            Potential function
        
        Returns
        -------
        ufl.algebra.Sum
            Variational formulation
        list
            List with the ufl.algebra.Sum objects to compute the instantanous
            current.
        dolfinx.fem.function
            Function of the charge species.
        """
        
        # Create the charge density functions together with the test functions.
        u  = dolfinx.fem.function.Function(V)
        ui = ufl.split(u)
        v  = ufl.TestFunctions(V)

        # A useful constant for the electric field calculation
        const  = constants.electronCharge / constants.vacuumPermitivity
        const /= self.relativePermitivity

        # If full coupling is enabled, the potential is the last value of the
        # list
        if self.efieldFullCoupling: uE = ui[-1]

        # The space coordinates.      
        x = ufl.SpatialCoordinate(V.mesh)

        # Variational formulation
        a = 0

        # List of induced currents
        iInduced = []

        # Chech that ionization equals 2.
        f_ionsum = round(np.sum([specie.fIonization for specie in self._species]), 3)
        assert f_ionsum == 2.000 or self.ignoreIonSum, "Error: Ionization "
        "fraction do not sum 2."
        
        # Iterate over the species, produce the variational formulations and the 
        # expressions to calculate the induced current.

        i = 0
        for specie in self._species:

            # *-- Transport term:
            # If mobility is a function of the electric field.
            if specie.mobType == 3:
                
                vel = self._getMyFunction("velocity", specie)

                # Derivate with respect to the charge density:
                a += ufl.inner(vel, ufl.grad(ui[i])) * v[i]
                
                # Derivate with respect to the velocity. It depends on the
                # symmetry of the problem.
                if self.symmetry == "Spherical":
                    b = (1 / x[0]**2) * ufl.Dx(x[0]**2 * vel[0], 0) + vel[1].dx(1)

                if self.symmetry == "Cylindrical":
                    b = (1 / x[0]) * ufl.Dx(x[0] * vel[0], 0) + vel[1].dx(1)

                if self.symmetry == "Cartesian":
                    b = ufl.div(vel)

                a += b * ui[i] * v[i]

            # If mobility is not a function of electric field the derivative can
            # be expressed explicitly.
            if specie.mobType == 1 or specie.mobType == 2:

                f_v = specie.getVelocity(self.rhumidity, self.pressure,
                                          self.temperature)
                
                a  += - ufl.inner(f_v(ufl.grad(uE), 1), ufl.grad(ui[i])) * v[i]

                if self.eFieldPerturbation:
                    j = 0
                    for specie2 in self._species:    
                        a    += f_v(1, 1) * const * ui[j] * ui[i] * specie2.charge * v[i]
                        j    += 1

                vel = - f_v(ufl.grad(uE), 1)

            # In both cases the induced current is computed in the same way:
            iInduced.append(specie.charge * vel * u[i])
            
            # *-- Source term:
            if specie.fIonization != 0:
                a += -specie.fIonization * beam.getFunction(i, 0) * v[i]
            
            # *-- Diffusion term: 
            if specie.diffType != False and type(v) != int:
                
                """To perform 3D simulation the full diffusion matrix should be
                introduced. This approximation will not work for 3D."""

                d1, d2, d12 = self._getMyFunction("diffusion", specie)
                a += (d1 * ui[i].dx(0) * v[i].dx(0) + d2 * ui[i].dx(1) * v[i].dx(1) 
                      + d12 * ui[i].dx(1) * v[i].dx(0))
            
            # *-- Attachment
            j = 0
            for process in self._attachment:
                attx = self._getMyFunction("attachment", j)
                if process[1] == specie:
                    a += attx * ui[i] * v[i]
                    
                if process[2] == specie:
                    idx = self.getIdx(process[1])
                    a -= attx * ui[idx] * v[i]
                    
                j += 1

            # *-- Recombination      
            for process in self._recombination:
                if process[1] == specie:
                    idx = self.getIdx(process[2])
                    a  += process[0] * ui[idx] * ui[i] * v[i]
                    
                if process[2] == specie:
                    idx = self.getIdx(process[1])
                    a  += process[0] * ui[idx] * ui[i] * v[i]
            
            # *-- Multiplication
            j = 0
            for process in self._multiplication:
                mult = self._getMyFunction("multiplication", j)
                if process[1] == specie:
                    a -= mult * ui[i] * v[i]
                    
                if process[2] == specie:
                    idx = self.getIdx(process[1])
                    a -= mult * ui[idx] * v[i]

                j += 1

            i += 1

        # If full coupling the electric field is included in this variational 
        # formulation
        if self.eFieldPerturbation and self.efieldFullCoupling:
            a += self.getVariationalFormPotential(u, ui[-1], v[-1])

        return a, iInduced, u
    
    def getVariationalFormPotential(self, u: dolfinx.fem.function, 
                                    u_E: dolfinx.fem.function, 
                                    v_E: dolfinx.fem.function) -> ufl.algebra.Sum:
        
        """Returns the variational formulation for the potential calculation

        Parameters
        ----------
        u : dolfinx.fem.function
            Function with the charge densities
        u_E : dolfinx.fem.function
            Function with the charge densities
        v_E : dolfinx.fem.function
            Function with the charge densities

        Returns
        -------
        ufl.algebra.Sum
            Variational formulation.
        """

        a_E = 0

        # A useful constant for the electric field calculation
        const  = constants.electronCharge / constants.vacuumPermitivity
        const /= self.relativePermitivity

        # Electric field basic calculation:
        a_E += u_E.dx(0) * v_E.dx(0) + u_E.dx(1) * v_E.dx(1) 
        a_E += u_E.dx(2) * v_E.dx(2)

        # Electric field perturbation
        i = 0
        for specie in self._species:
                a_E  -=  const * u[i] * specie.charge * v_E
                i    += 1

        return a_E
    
    def printInfo(self, logger: Logger):
        
        """Print the information of the physics to the output file.

        Parameters
        ----------
        logger: Logger
            Object managing the output.
        """
        
        efieldDescription = "Off"
        if self.eFieldPerturbation: efieldDescription = "On"
        
        logger.printHeader('*-- Physics:')
        logger.printData("Physics name"               , f"{self.physicsName}")
        logger.printData("Temperature"                , f"{self.temperature:.2f} \u00b0C")
        logger.printData("Pressure"                   , f"{self.pressure:.2f} hPa")
        logger.printData("Relative humidity"          , f"{self.rhumidity:.2f} %")
        logger.printData("Relative permittivity"      , f"{self.relativePermitivity}")
        logger.printData("Electric field perturbation", f"{efieldDescription}")

        if self.eFieldPerturbation: 
            logger.printData("Electric field full coupled", f"{self.efieldFullCoupling}")
        
        logger.info('\nList of species to be simulated:')
        [specie.printInfo(logger) for specie in self._species]
        
        logger.info('\nList of process included:\n')
        i = 1
        for process in self._recombination:
            logger.info(f" {i}.- {process[1].name} + {process[2].name}"
                        f" ->  Neutral [{process[0]:.4E} m^3/s]")
            i += 1

        for process in self._attachment:
            logger.info(f" {i}.- {process[1].name} -> {process[2].name}"
                        f" [Function of electric field]")
            i += 1
            
        for process in self._multiplication:
            logger.info(f" {i}.- {process[1].name} -> 2*{process[1].name}"
                        f" + {process[2].name} [Function of electric field]")
            i += 1
            
        logger.info('\n')
