"""
    Implements the base class to describe chambers.
"""
import dolfinx
import ufl
import os
import basix
import numpy as np
import mpi4py

from .utils            import jitOptions, deleteCache, constants
from dolfinx.nls.petsc import NewtonSolver
from ICSimFEM          import Logger
from dolfinx.fem.petsc import NonlinearProblem

class Chamber:
    
    """Class for the description of the ionization chamber.

    It encapsules the base properties of an ionization chamber, namely its
    calibration coefficient (Ndw) in Gy/C, volume (volume) in m^3, serial
    number (SN) and symmetry (symmetry) and connect this with the chamber
    geometry and boundary conditions.

    To load the desired geometry this class allows the use of geometry files
    with extension ".xdmf". For that purpose, two files are necessary:
    
        1.- File with extension with the name and path specify using the method
            setFilePath.
        2.- File containing the lower dimension geometries tagged for applying
            boundary conditions. This file must have the same name with a "_mf"
            added at the end.
    """
    
    def __init__(self, name: str, Ndw: float = 1.0, volume: float = 1,
                 SN: str = "0000", symmetry: str = "Cartesian") -> None:
        
        """
        Parameters
        ----------
        name : str
            Name given to the ionization chamber.
        Ndw : float
            Calibration coefficient for a reference beam quality Q in Gy/C.
        SN : str
            Serial number of the ionization chamber (only used for naming).
        symmetry : str
            Symmetry of the geometry. This parameter is used for the volume
            element.
        """

        # Chamber basic properties
        self.name     = name
        self.Ndw      = Ndw
        self.SN       = SN
        self.symmetry = symmetry


        # Volume of the ionization chamber. This parameter is used to calculate
        # the release charge in the medium when doing charge transport
        # simulations.
        self.volume   = volume # m^3

        # Voltage applied to the chamber in V.
        self.voltage  = 0.0 

        # Filename (including path) of the base file containing the geometry
        # of the chamber.
        self._fMesh = ""  

        # Flag to indicate if the file if the mesh is already loaded or not .    
        self._opened = False

        # Volume calculate with the actual geometry. This volume may have 
        # different units such as m or m^2 depending on the dimension
        # of the geometry and the symmetry.
        self._realVolume   = 0

        # List containing the mesh, subdomains and boundary of the present
        # geometry respectively.
        self._meshData = []

        # Number of cells in the present geometry.
        self._numCells = 0

        # Cell names
        self._cellName = ""

        # List keeping track of the cache paths to clean after simulation.
        self._cachePaths = []

        # Dimension of the geometry.
        self._dim = 0

        # Tags for boundary conditions.
        self.highVoltageTag   = None
        self.collectionTag    = None
        self.guardRingTag     = None

        # Inverse the polarity (If true the polarization is applied in the 
        # internal electrode).
        self.inversePolarity  = False     

        # This condition is here to simulate recombination by backdiffusion.
        self.boundaryConditionsEverywhere = False

        # If no calibration coefficient is provided, then estimate it from the
        # volume of the chamber
        if self.Ndw == 0:
            norm     = (constants.electronCharge * constants.airDensity * self.volume)
            self.Ndw = constants.averageEnergyPerIonPair / norm # Gy/C

    @property
    def Ndw(self):
        return self.__Ndw
    
    @Ndw.setter
    def Ndw(self, value: float):
        assert value >= 0, "Calibration coefficient must me > 0."
        self.__Ndw = value

    @property
    def volume(self):
        return self.__volume
    
    @volume.setter
    def volume(self, value: float):
        assert value >= 0, "Volume must me > 0."
        self.__volume = value

    @property
    def symmetry(self):
        return self.__symmetry
    
    @symmetry.setter
    def symmetry(self, value: str):
        assert value in ["Cartesian", "Cylindrical", "Spherical"], "Symmetry" 
        "must me Cartesian, Cylindrical or Spherical"
        self.__symmetry = value

    @property
    def dim(self):
        assert self._opened, "The file is not yet open."
        return self._dim
    
    @property
    def numCells(self):
        return self._numCells
    
    @property
    def fileName(self):
        return self._fMesh
    
    @fileName.setter
    def fileName(self, value: str):

        # Remove the extension
        base_name = value.replace(".xdmf", "")

        # Check if the file exist:
        if not os.path.isfile(f"{base_name}.xdmf") and not \
            os.path.isfile(f"{base_name}_mf.xdmf"):
            raise Exception("Mesh file do not exist")
        
        self._fMesh = base_name

    @property
    def numCells(self):
        return self._numCells
    
    @property
    def inversePolarity(self) -> bool:
        return self.__inversePolarity
    
    @inversePolarity.setter
    def inversePolarity(self, value: bool) -> None:
        self.__inversePolarity = value

    @property
    def guardRingTag(self) -> int:
        return self.__guardRingTag
    
    @guardRingTag.setter
    def guardRingTag(self, value: int) -> None:
        self.__guardRingTag = value

    @property
    def highVoltageTag(self) -> int:
        return self.__highVoltageTag
    
    @highVoltageTag.setter
    def highVoltageTag(self, value: int) -> None:
        self.__highVoltageTag = value

    @property
    def collectionTag(self) -> int:
        return self.__collectionTag
    
    @collectionTag.setter
    def collectionTag(self, value: int) -> None:
        self.__collectionTag = value

    @property
    def voltage(self) -> None:
        return self.__voltage
    
    @voltage.setter
    def voltage(self, value: float) -> None:
        self.__voltage = float(value)

    @property
    def cellName(self) -> str:
        return self._cellName
    
    @property
    def boundaryConditionsEverywhere(self) -> bool:
        return self.__boundaryConditionsEverywhere
    
    @boundaryConditionsEverywhere.setter
    def boundaryConditionsEverywhere(self, value: bool) -> None:
        self.__boundaryConditionsEverywhere = value


    def loadMesh(self) -> None:

        """Load the mesh. This function load the mesh specify in variable _fMesh.    
        """

        self._meshData = self._openXDMF()
        
    def _openXDMF(self) -> list:
        
        """ Open the mesh, physical regions an facets regions of a .xdmf file
        """

        assert self._fMesh != "", "File mesh is not set"
        
        self._opened = True

        # Open the file and load data with the dolfinx io        
        with dolfinx.io.XDMFFile(mpi4py.MPI.COMM_WORLD, 
                                 f"{self._fMesh}.xdmf", "r") as xdmf:
            
            mesh       = xdmf.read_mesh(name = "Grid")
            subdomains = xdmf.read_meshtags(mesh, name = "Grid")
            
        topology       = mesh.topology
        self._dim      = topology.dim
        topology.create_connectivity(self._dim, self._dim - 1)
        self._cellName = topology.cell_name()
        
        # Open the boundaries and load them.
        with dolfinx.io.XDMFFile(mpi4py.MPI.COMM_WORLD, 
                                 f"{self._fMesh}_mf.xdmf", "r") as xdmf:
            
            boundaries = xdmf.read_meshtags(mesh, name = "Grid")

        self._numCells = topology.index_map(self._dim).size_local

        return [mesh, subdomains, boundaries]
    
    def getScalingFactor(self):

        """This function returns the volume element depending on the assumed
        simmetry
        """

        x = ufl.SpatialCoordinate(self._meshData[0])

        if   self.__symmetry == "Cartesian"  : return 1
        elif self.__symmetry == "Cylindrical": return 2 * np.pi * x[0]
        elif self.__symmetry == "Spherical"  : return 2 * np.pi * x[0]**2
    
    def computeVolume(self) -> float:

        """Compute the (physical) volume of the chamber
        """

        if not self._opened: self.loadMesh()

        dx = ufl.Measure('dx', domain = self._meshData[0])

        # Integrate over volume tag
        form = dolfinx.fem.form(self.getScalingFactor() * dx,
                                 jit_options = jitOptions())
        
        # Note that _realVolume might have m units if geometry is 1D.
        self._realVolume = dolfinx.fem.assemble_scalar(form)

        return self._realVolume

    def _generateBoundaryCondition(self, boundaryTag: int, value: float,
                    V: dolfinx.fem.functionspace) -> dolfinx.fem.dirichletbc:

        """Generate a boundary conditions for the FEM simulation.

        Parameters
        ----------
        boundaryTag : int
            Number of the boundary to apply condition
        value : float
            Value on the boundary
        V : dolfinx.fem.functionspace
            Function space
        """

        facets = self._meshData[2].find(boundaryTag)
        dolfC  = dolfinx.fem.Constant(self._meshData[0], value)
        self._meshData[0].topology.create_connectivity(self.dim - 1, self.dim)
        dofs   = dolfinx.fem.locate_dofs_topological(V, self.dim - 1, facets)
        
        return dolfinx.fem.dirichletbc(dolfC, dofs, V)
    
    def getPotentialBoundaryConditions(self, V: dolfinx.fem.functionspace,
                        voltageList: list = []) -> list[dolfinx.fem.dirichletbc]:

        """Get the boundary conditions to calculate the potential and/or the
        electric field.

        Parameters
        ----------
        V : dolfinx.fem.functionspace
            Function space.
        voltageList: list
            List with the values of the voltage on: High voltage, collection
            electrode and guard ring.
        """

        onTags = [self.highVoltageTag, self.collectionTag, self.guardRingTag]

        assert self._opened, "Geometry is not opened"
        assert onTags[0] != None and onTags[1] != None, "High voltage electrode" 
        " or collection electrode is not set."
       
        # If no other voltage configuration is supplied do a standard 
        # configuration
        if voltageList == []:
            # List in order: high voltage electrode, collection electrode and 
            # guard ring electrode respectively.
            voltageList = [self.__voltage, 0.0, 0.0]
            if self.__inversePolarity == True:
                voltageList = [0.0, self.__voltage, self.__voltage]

        bcList = []
        for value, tag in zip(voltageList, onTags):
            if tag != None:
                bcList.append(self._generateBoundaryCondition(tag, value, V))

        return bcList
    
    def getChargeBoundaryConditions(self, V: dolfinx.fem.functionspace,
                                 charge: int) -> list[dolfinx.fem.dirichletbc]:

        """Get the boundary conditions to simulate charge transport for a given
        charge specie
        
        Parameters
        ----------
        V : dolfinx.fem.functionspace
            Function space.
        charge: int
            int with the sign of the charge, used to determine the direction of 
            the charge moving in the chamber.
        """

        bcList = []
        onTags = [self.highVoltageTag, self.collectionTag, self.guardRingTag]

        f = 1
        if self.inversePolarity == True: f = -1

        # High voltage electrode:
        if (f * charge * self.__voltage) > 0 or self.boundaryConditionsEverywhere:
            bcList.append(self._generateBoundaryCondition(onTags[0], 0.0, V))

        # Collection electrode:
        if (f * charge * self.__voltage) < 0 or self.boundaryConditionsEverywhere:
            bcList.append(self._generateBoundaryCondition(onTags[1], 0.0, V))

            # Guard ring (if it exist):
            if onTags[2] != None:
                bcList.append(self._generateBoundaryCondition(onTags[2], 0.0, V))

        return bcList
        
    def computeElectricField(self, voltageList: str = [], save: str = "",
                                    polDegree: int = 1) -> dolfinx.fem.function:
        
        """Function to calculate the potential and electric field for a given
        geometry.
        
        Parameters
        ----------
        voltageList : list
            Optional. Voltage list with the values of the voltage on the: high
            voltage electrode, collection electrode and guard ring electrode.
        save: str
            Optional. String with the path where the data is save. If not
            supplied dat will not be saved.
        polDegree: int
            Optional. Degree of the polinomial for the FEM calculation.        
        """
    
        # if not open, open the file
        if not self._opened: self.loadMesh()
        
        P1 = basix.ufl.element("Lagrange", self._meshData[0].topology.cell_name(),
                                polDegree)
        
        V  = dolfinx.fem.functionspace(self._meshData[0], P1)

        # Solution and test function for the problem.
        u = dolfinx.fem.function.Function(V)
        v = ufl.TestFunction(V)

        # Get the boundary conditions
        bc = self.getPotentialBoundaryConditions(V, voltageList)
        
        # Poisson equation:
        a = ufl.inner(ufl.grad(u), ufl.grad(v)) * self.getScalingFactor() * ufl.dx

        petsc_options = {
            "snes_type"                : "newtonls",
            "snes_linesearch_type"     : "bt",
            "snes_atol"                : 1E-5,
            "snes_rtol"                : 1E-5,
            "ksp_type"                 : "preonly",
            "pc_type"                  : "lu",
            "pc_factor_mat_solver_type": "petsc",
            "snes_max_it"              : 1000
        }

        problem = NonlinearProblem(a, u, bcs = bc, petsc_options_prefix = "Ef_", 
                                   petsc_options = petsc_options, 
                                   jit_options = jitOptions())
        u       = problem.solve()

        # Compute electric field
        U_grad = - ufl.grad(u)
        W      = dolfinx.fem.functionspace(self._meshData[0], ("CG", 1, (3, )))
        expr   = dolfinx.fem.Expression(U_grad, W.element.interpolation_points,
                                         jit_options = jitOptions())
        fun    = dolfinx.fem.Function(W)
        fun.interpolate(expr)

        # Compute effective area
        if self._realVolume == 0: self.computeVolume()
        
        # Save the calculated values if requested
        if save != "":
            
            W13 = dolfinx.fem.functionspace(self._meshData.mesh, ("CG", 1, (3, )))
            fun13    = dolfinx.fem.Function(W13)
            fun13.interpolate(fun)

            W11 = dolfinx.fem.functionspace(self._meshData.mesh, ("CG", 1, ))
            fun11    = dolfinx.fem.Function(W11)
            fun11.interpolate(u)

            with dolfinx.io.XDMFFile(self._meshData[0].comm,
                                      f"{save}_Potential.xdmf", "w") as xdmf:
                xdmf.write_mesh(self._meshData[0])
                xdmf.write_function(fun11)

            with dolfinx.io.XDMFFile(self._meshData[0].comm,
                                      f"{save}_ElectricField.xdmf", "w") as xdmf:
                xdmf.write_mesh(self._meshData[0])
                xdmf.write_function(fun13)

        deleteCache() 

        return u, fun
    
    def computeWeightedField(self, save: str = "", polDegree: int = 1):

        """Compute the weighted electric field: 1 V on the collection
         electrode and the rest set to 0 V."""

        weightedField  = self.computeElectricField([0.0, 1.0, 0.0], save, polDegree)
        return weightedField
        
    def printInfo(self, logger: Logger):
        
        """Print the information of the beam to the output file.

        Parameters
        ----------
        logger: Logger
            Object managing the output.
        """

        description = "High voltage electrode"
        if self.__inversePolarity: description = "Collection electrode"
        
        logger.printHeader('*-- Chamber description:')
        logger.printData("Chamber name"              , f"{self.name}")
        logger.printData("Serial number"             , f"{self.SN}")
        logger.printData("Calibration coefficient"   , f"{self.Ndw:.3E} Gy/C")
        logger.printData("Applied voltage"           , f"{self.voltage:+.2f} V")
        logger.printData("Voltage applied on"        , description)
        logger.printData("Mesh file name"            , f"{self._fMesh}")
        logger.printData("High voltage electrode tag", f"{self.highVoltageTag}")
        logger.printData("Collection electrode tag"  , f"{self.collectionTag}")
        logger.printData("Guard ring electrode tag"  , f"{self.guardRingTag}")
        logger.printData("Number of elements"        , f"{self._numCells}")
        logger.printData("Symmetry of the geometry"  , f"{self.symmetry}")
        
        logger.info('\n')
