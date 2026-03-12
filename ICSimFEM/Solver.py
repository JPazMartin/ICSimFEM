"""
    Class that solves the transport problen for a given geometry and a given
    physics.
"""
import ufl 
import dolfinx
import basix
import os
import mpi4py
import time
import numpy as np

from .Chamber          import Chamber
from .Physics          import Physics
from .Beam             import Beam
from .TimeStepper      import TimeStepper
from .Saver            import Saver
from .Logger           import Logger
from ICSimFEM.utils    import jitOptions, constants, deleteCache
from scipy.integrate   import trapezoid
from dolfinx.fem.petsc import NonlinearProblem

class Solver:

    """Class that solves the transport problen for a given geometry and a given
    physics.
    """

    def __init__(self):
        
        self.saveData      = True

        # Print information to screen or only for the log file
        self.printScreen   = True
        # Number of steps between saving simulation data to binary file
        self.saveEach      = 1  
        # Number of steps in wich reports information about the simulation state
        self.reportEach    = 50  

        # Activate/deactivate enriched space.
        self.enrichedSpace = False

        # Degree of the used polynomials
        self.transportEquationsPDegree = 1
        self.electricFieldPDegree      = 2

        # Default solvers
        self.solver         = "preonly"
        self.preconditioner = "lu"
    
    @property
    def chamber(self) -> Chamber:
        return self.__chamber
    
    @chamber.setter
    def chamber(self, value: Chamber) -> None:
        assert isinstance(value, Chamber)
        self.__chamber = value
    
    @property
    def physics(self) -> Physics:
        return self.__physics
    
    @physics.setter
    def physics(self, value: Physics) -> None:
        assert isinstance(value, Physics)
        self.__physics = value

    @property
    def beam(self) -> Beam:
        return self.__beam
    
    @beam.setter
    def beam(self, value: Beam) -> None:
        assert isinstance(value, Beam)
        self.__beam = value    

    @property
    def transportEquationsPDegree(self) -> int:
        return self.__transportEquationsPDegree
    
    @transportEquationsPDegree.setter
    def transportEquationsPDegree(self, value: int) -> None:
        if not value >= 1:
            raise RuntimeError("The degree for the transport equations must be > 0.")
        self.__transportEquationsPDegree = value

    @property
    def electricFieldPDegree(self) -> int:
        return self.__electricFieldPDegree
    
    @electricFieldPDegree.setter
    def electricFieldPDegree(self, value: int) -> None:
        if not value >= 1:
            raise RuntimeError("The degree for the transport equations must be > 0.")
        self.__electricFieldPDegree = value

    @property
    def timeStepper(self) -> TimeStepper:
        return self.__timeStepper
    
    @timeStepper.setter
    def timeStepper(self, value: TimeStepper) -> None:
        assert isinstance(value, TimeStepper)
        self.__timeStepper = value

    def _checkRequirements(self) -> None:

        """Evaluates if all the pieces needed by the solver are already supplied
        by the user.
        """

        minList = ["chamber", "physics", "beam"]
        for requirement in minList:
            assert hasattr(self, requirement), f"{requirement} is not set in" 
            "Solver. Simulation aborted"

        # In addition, if the beam is pulsed, time stepper is needed
        if self.beam.pulsed:
            assert hasattr(self, "timeStepper"), "Time stepper is not provided"
        
    def _generateSpaces(self):

        """Generate the polynomials needed for the calculations"""

        self.chamber.loadMesh()

        if self.enrichedSpace:
            QE = basix.ufl.element("Lagrange", self.chamber.cellName,
                                    self.transportEquationsPDegree)
            BE = basix.ufl.element(  "Bubble", self.chamber.cellName,
                                    self.chamber.dim + 1)
            Pn = basix.ufl.enriched_element([QE, BE])

        else:
            Pn = basix.ufl.element("Lagrange", self.chamber.cellName,
                                    self.transportEquationsPDegree)

        nElements = self.physics._nSpecies
        if self.physics.eFieldPerturbation and self.physics.efieldFullCoupling:
            nElements += 1

        elements = basix.ufl.mixed_element([Pn] * nElements)

        # Function space
        self.V   = dolfinx.fem.functionspace(self.chamber._meshData.mesh,
                                 elements)
        self.V1  = dolfinx.fem.functionspace(self.chamber._meshData.mesh,
                                 ("CG", 1))
        self.V2  = dolfinx.fem.functionspace(self.chamber._meshData.mesh,
                                 ("CG", 1, (3, )))
        
        # Generate the spaces for the (perturbed) electric field
        if self.physics.eFieldPerturbation and not self.physics.efieldFullCoupling:
            PE = basix.ufl.element("Lagrange", self.chamber.cellName, 
                                   self.electricFieldPDegree)
            self.V_Efield = dolfinx.fem.functionspace(self.chamber.mesh,
                                         PE)
            self.uE = dolfinx.fem.function.Function(self.V_Efield)
            self.v_E = ufl.TestFunction(self.V_Efield)

        return

    def _computeVariationalForm(self):

        """Computes the variational formulation for the solution of the problem
        """

        if not self.physics.eFieldPerturbation:    
            potential = self.inPotential
        elif self.physics.efieldFullCoupling:
            potential = 1
        else:
            potential = self.uE
            potential.interpolate(self.inPotential)

        scale = self.chamber.getScalingFactor()
        
        if self.beam.pulsed:
            self.a, iInduced, u = (
                self.timeStepper.generateVariationalFormulation(self.V, 
                                                    self.physics, potential))

        else:
            self.a, iInduced, u = (
                self.physics.getVariationalForm(self.V, self.beam, potential))

        if self.physics.efieldFullCoupling:
            u.sub(self.physics._nSpecies).interpolate(self.inPotential)
            potential = ufl.split(u)[-1]

        self.a = self.a * scale * ufl.dx(domain = self.chamber._meshData[0])

        # If the dymension of the problem is 1D the area has to be included 
        # here.
        area = 1
        if self.chamber.dim == 1:
            area = self.chamber.volume / self.chamber._realVolume
         
        # Compute time-resolve signal.
        for i in iInduced:
            intensity = ufl.inner(- self.weightedField, i) * area * scale * ufl.dx
            
            self.inducedI.append(dolfinx.fem.form(intensity, 
                                                   jit_options = jitOptions()))
        
        # Compute the carriers left in the volume:
        self.uSum = []
        for i in range(self.physics._nSpecies):

            integral = u[i] * scale * ufl.dx
            
            self.uSum.append(dolfinx.fem.form(integral,
                                               jit_options = jitOptions()))

        self.u = u

        if (self.physics.eFieldPerturbation and not 
                                    self.physics.efieldFullCoupling):
            self.a_E = self.physics.getVariationalFormPotential(u, self.uE,
                                                                 self.v_E)
            self.a_E = self.a_E * scale * ufl.dx(
                domain = self.chamber._meshData[0])

        U_grad     = - ufl.grad(potential)
        self.exprE = dolfinx.fem.Expression(U_grad, 
            self.V2.element.interpolation_points, jit_options = jitOptions())

    def _getNewtonSolver(self, equation: ufl.form.Form, u: dolfinx.fem.Function,
                          bcs: list):

        """Returns a newton solver for a given equation, solution function and 
        boundary conditions.

        TODO: Adapt solver to manage when stationary is reached

        Parameters
        ----------
        equation : ufl.form.Form
            Equation to be solved
        u : dolfinx.fem.Function
            Function where solution is stored
        bcs: list
            List of the boundary conditions to be applied.        
        """

        # TODO: Evaluate in cluster if pc_factor_mat_solver_type = petscactually
        # works
        petsc_options = {
            "snes_atol": 1e-5,
            "snes_rtol": 1e-10,
            "ksp_type" : self.solver,
            "pc_type"  : self.preconditioner,
            "pc_factor_mat_solver_type": "petsc",
            "snes_max_it": 1000
        }
    
        problem = NonlinearProblem(equation, u, bcs = bcs,
                                   petsc_options_prefix = "Ct_",
                                   petsc_options = petsc_options, 
                                   jit_options   = jitOptions())
        
        return problem

    def _initializeValues(self):

        """Load the initial values and generate the solvers to start the
        simulation
        """
        
        # Set the initial values
        if self.beam.pulsed:
            i = 0
            for specie in self.physics._species:
                self.beam.setInitialValues(self.timeStepper.u_n.sub(i), specie)
                i += 1
        
        # If full coupling set uE to the last function of u
        if (self.physics.eFieldPerturbation and 
                                        self.physics.efieldFullCoupling):
            self.u.split()[-1].interpolate(self.inPotential)
            self.uE = self.u.split()[-1]

        # Update the transport parameters with the calculated
        # value of the electric field
        self.physics.updateTransportParameters(self.exprE)

        # Get the boundary conditions of the charge transport problem
        bc = []; i = 0
        for specie in self.physics._species:
            bcAdd = self.chamber.getChargeBoundaryConditions(self.V.sub(i),
                                                              specie.charge)
            [bc.append(bCondition) for bCondition in bcAdd]
            i += 1

        # Get the boundary conditions for the electric field calculation. If 
        # the problem is full coupled then add them to the bc array
        if (self.physics.eFieldPerturbation and not 
                                    self.physics.efieldFullCoupling):
            bcEField = self.chamber.getPotentialBoundaryConditions(self.V_Efield)
        
        if self.physics.efieldFullCoupling:
            bcEField = self.chamber.getPotentialBoundaryConditions(
                self.V.sub(self.physics._nSpecies))
            [bc.append(bCondition) for bCondition in bcEField]

        # Get the newton solve for the charge transport problem and electric
        # field perturbation if full copupled.
        self.solverObject = self._getNewtonSolver(self.a, self.u, bc)

        # If electric field perturbation is on and not full coupled, then 
        # get the solver for the electric field perturbation.
        if (self.physics.eFieldPerturbation and not
                                        self.physics.efieldFullCoupling):
            self.solverEObject  = self._getNewtonSolver(self.a_E, self.uE,
                                                         bcEField)

    def _doAStep(self) -> list[int, bool, bool]:

        """Performs a step in the simulation

        Returns
        -------
        int 
            Number of iterations needed by the simulation to converge
        bool
            Convergency status of the simulation
        bool
            True if the simulation if finished (because no left charge carriers
            or because it is a continous simulation with only 1 step)
        """

        self.u = self.solverObject.solve()
        it     = self.solverObject.solver.getIterationNumber()
        conv   = self.solverObject.solver.getConvergedReason() > 0

        # Remove non-physical (negative) solutions for the charge densities.
        for i in range(self.physics._nSpecies):
            idxNeg = np.where(self.u.sub(i).collapse().x.array < 0)[0]
            if len(idxNeg) > 0: self.u.sub(i).collapse().x.array[:][idxNeg] = 0

        if self.beam.pulsed:

            # Calculate the induced current:
            self.tArray.append(self.timeStepper.time)
            self.IArray.append([dolfinx.fem.assemble_scalar(I) for
                                I in self.inducedI])
        
            uSum = np.sum([dolfinx.fem.assemble_scalar(u) for u in self.uSum])
            self.cLeft = uSum / (
                2 * self.beam.releaseCharge * self.chamber._realVolume)

            # The simulation ends when the (simulation) elapsed time is larger
            # than the pulse duration and the number of carriers left is below
            # 0.01 %.
            endSim = False
            if (self.timeStepper.time > self.beam.pulseDuration and
                                                self.cLeft < 10E-5):
                endSim = True

            # If the electric field perturbation is activated, compute the
            # electric field and update the transport parameters
            if (self.physics.eFieldPerturbation and not 
                                        self.physics.efieldFullCoupling):
                
                it, conv = self.solverEObject.solve(self.uE)
                self.physics.updateTransportParameters(self.exprE)

            # If simulation is not finished update the time step.
            if not endSim: 
                self.timeStepper.updateFunctions()

            return it, conv, endSim
        
        else:

            I           = [dolfinx.fem.assemble_scalar(I) for I in self.inducedI]
            self.IArray = [I]

            # Iterate (the transport parameters must be updated each time)
            if self.physics.efieldFullCoupling:

                while True:
                    self.physics.updateTransportParameters(self.exprE)
                    it, conv    = self.solverObject.solve(self.u)
                    I           = [dolfinx.fem.assemble_scalar(I) for 
                                   I in self.inducedI]
                    maxDiff     = np.max((np.array(self.IArray) / np.array(I)
                                           - 1))
                    self.IArray = [I]
                    if abs(maxDiff * 100) < 0.01: break
        
        return it, conv, True        

    def _initialize(self):

        """Initialize the simulation generating the spaces and the functions
        """

        self.inducedI = []; self.uSum = []; self.u = None
        # Generate function spaces
        self._generateSpaces()

        # Pass geometry to beam:
        self.beam.setSpace(self.V1)

        # Generate functions for physical parameters
        self.physics.generateFunctions(self.V1, self.V2)

    def run(self, name: str, pathResults: str = "."):
        
        """Run the simulation

        Parameters
        ----------
        name : str
            Name given to the simulation.
        pathResults : str
            Path where the data of the simulation will be saved.

        Returns
        -------
        float
            Charge collection efficiency
        list[float, ...]
            Values of the collected charge for the different charge species
            simulated (in order)
        """

        self._checkRequirements()

        # Ensure if there is a directory with the name of the ionization
        # chamber is created. Otherwise, create it.
        folder_name = f"{pathResults}/ICSim_{self.chamber.name}" \
                      f"_SN{self.chamber.SN}"
        os.makedirs(folder_name, exist_ok = True)

        # Transfer the simetries properties to physics class
        # TODO: Avoid this
        self.physics.symmetry = self.chamber.symmetry

        # *-- Open the mesh and write geometry information:
        self.chamber.loadMesh()

        # Initialize the spaces
        self._initialize()

        # *-- Open the log file:
        logName = f"{folder_name}/{self.chamber.name}_{self.chamber.SN}" \
                  f"_{name}.ICSimlog"
        
        logFile  = Logger(logName, self.printScreen)

        # Write the information about the simulation:
        for obj in [self.chamber, self.physics, self.beam]: 
            obj.printInfo(logFile)

        if self.beam.pulsed:
            self.timeStepper.printInfo(logFile)
        
        self.printInfo(logFile)

        # Simulation starts
        logFile.printHeader('*-- Simulation:')

        # Calculated the weighted potential:
        logFile.info('1.- Calculation of the weighted potential:')
        
        tStart = time.time()
        (self.weightedPotential, self.weightedField) = (
            self.chamber.computeWeightedField(polDegree = 
                                              self.electricFieldPDegree))
        tEnd   = time.time() 
        
        logFile.info(f'    Elapsed time: {tEnd - tStart:.2f} s')
        
        # Calculated the physical potential:
        logFile.info('2.- Calculation of the potential:')
        tStart = time.time()

        (self.inPotential, self.inEField) = (
            self.chamber.computeElectricField(polDegree = 
                                              self.electricFieldPDegree))
        tEnd   = time.time()
        
        logFile.info(f'    Elapsed time: {tEnd - tStart:.2f} s')

        # Assemble the physics of the problem.
        logFile.info('3.- Assembling the system of physical equations')

        self.beam.initialize(self.chamber, self.physics)

        if self.beam.pulsed:
            self.timeStepper.initialize(self.chamber, self.beam)
        
        # Compute variational form
        self._computeVariationalForm()
        self._initializeValues()
        
        if self.saveData:

            filename   = f"{folder_name}/Data_{self.chamber.name}" \
                         f"_{self.chamber.SN}_{name}"
            
            self.saver = Saver(filename, self.chamber, self.physics._nSpecies)

            self.saver.writeField(self.weightedPotential, self.weightedField)
            self.saver.writeField(self.inPotential, self.inEField)


        if self.beam.pulsed:
            # Open the file to save the time-resolved values:
            f = open(f"{folder_name}/Current_{self.chamber.name}_" \
                     f"{self.chamber.SN}_{name}.csv", "w+")

            # Write header
            f.write("t (s)"); [f.write(f",I [{specie}] (A)") for specie
                                in self.physics._speciesName]

        
        # Charge transport starts
        logFile.info('4.- Charge transport simulation starts:\n')
      
        tStart = time.time()

        # Initialize quantities for simulation
        self.tArray = []
        self.IArray = []
        start       = tStart
        n           = 0
        end_sim     = False

        while not end_sim:   

            n += 1
            # Performs a step
            it, conv, end_sim = self._doAStep()
            
            if len(self.tArray) > 0 and self.beam.pulsed:
                f.write("\n")
                f.write(f"{self.tArray[-1]:.6E}")
                [f.write(f",{I * constants.electronCharge:.6E}") for
                  I in self.IArray[n - 1]]

            # Save the data
            if self.saveData and (n % self.saveEach == 0) and self.beam.pulsed:   

                self.saver.writeSpecies(self.u, self.timeStepper.time)

                if self.physics.eFieldPerturbation:
                    self.saver.writeField(self.uE, self.physics.electricField)
                else:
                    self.saver.writeField(self.inPotential, self.inEField)

            if self.saveData and not self.beam.pulsed:

                self.saver.writeSpecies(self.u, 0)

                if self.physics.eFieldPerturbation:
                    self.saver.writeField(self.uE, self.physics.electricField)
                else:
                    self.saver.writeField(self.inPotential, self.inEField)

            if (n % self.reportEach == 0) and (self.beam.pulsed):
                
                tEnd = time.time()
                
                logFile.info(f"    Step number = {n:>5}; Elapsed time = "
                             f"{tEnd - start:.3f} s")
                logFile.info(f"       -> Time in simulation          = " 
                             f"{self.timeStepper.time:.3E} s")
                logFile.info(f"       -> Fraction of carriers left   = " 
                             f"{self.cLeft:.5f}")
                
                start = tEnd
                
        logFile.info('Charge transport ends\n')

        # Close the file if save data was active
        if self.saveData: self.saver.close()
        
        # Results section
        logFile.info('\n')
        logFile.info('*-- Results:')
        logFile.info('============')
        
        # Calculate interesting values:
        self.IArray = np.array(self.IArray)
        self.tArray = np.array(self.tArray)
        
        # 1.- Collected charge
        logFile.info('1.- Induced signal:')

        # If it is not pulsed:
        if not self.beam.pulsed:
            for value, specie in zip(self.IArray[0], self.physics._speciesName):
                logFile.info(f"   I ({specie}) = " 
                             f"{value * constants.electronCharge:.4E} A")

            Q = [I for I in self.IArray] # m^{-2}

        # if it is pulsed:
        else:
            Q = [trapezoid(self.IArray[:, i], self.tArray) for i in 
                 range(len(self.IArray[0]))]
            for value, specie in zip(Q, self.physics._speciesName):
                logFile.info(f"   Q ({specie}) = "
                             f"{value * constants.electronCharge:.4E} C")

        # 2.- Charge collection efficiency
        logFile.info('2.- Charge collection efficiency:')
        CCE = abs(np.sum(Q)) / (self.beam.releaseCharge * self.chamber.volume)
        logFile.info(f"   CCE = {abs(CCE):.6f}")
        
        # Close file for the current
        if self.beam.pulsed: f.close()
        
        # End the simulation
        tEnd = time.time()         
        logFile.info(f"\n\nTotal elapsed time in simulation: " 
                     f"{tEnd - tStart:.2f} s")

        logFile.close()
        deleteCache()

        return CCE, [charge * constants.electronCharge for charge in Q]
        
    def printInfo(self, logger: Logger) -> None:

        """Prints the information of the beam to the output file.

        Parameters
        ----------
        logger: Logger
            Object managing the output.
        """
        
        logger.printHeader("*-- Solver")
        logger.printData("Bubble enriched space", f"{self.enrichedSpace}")
        logger.printData("Transport equations space degree", 
                         f"{self.transportEquationsPDegree}")
        
        logger.printData("Electric Field space degree", 
                         f"{self.electricFieldPDegree}")
        
        logger.printData("Solver", f"{self.solver}")
        logger.printData("Preconditioner", f"{self.preconditioner}")

        logger.info('\n')
        logger.info('\n')
