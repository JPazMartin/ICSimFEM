"""
    Class to define a 1D parallel plate ionization chamber
    with cylindrical symmetry and generate a mesh.
"""

from ICSimFEM import Chamber

import numpy as np
import gmsh
import dolfinx
import mpi4py

class ParallelPlateIC1D(Chamber):

    """Class to define a 1D parallel plate ionization chamber
    with cylindrical symmetry and generate a mesh.
    """

    def __init__(self, name: str, Ndw: float, gap: float, radius: float,
                 SN: str = "0000") -> None:
        
        """Class to define a 1D parallel plate ionization chamber
        with cylindrical symmetry and generate a mesh.

        Parameters
        ----------
        name : str
            Name given to the ionization chamber.
        Ndw  : float
            Calibration coefficient for a reference beam quality Q in Gy/C.
        gap : float
            Distance between electrodes in m.
        radius : float
            Radius of the ionization chamber in m. It is used to calculate the 
            volume of the cavity.
        SN : str
            Optional. Serial number of the ionization chamber
        """

        self.gap    = gap                     # m
        self.radius = radius                  # m 
        self.nSteps = 500

        # Calculated volume of the chamber. This value can be accesed and 
        # modified if needed.
        volume  = np.pi * radius**2 * gap # m^3

        # Call the Chamber constructor.
        super().__init__(name, Ndw, volume, SN, "Cartesian")

        # _realVolume is used only when the geometry is 1D to convert the 
        # instantanous current from A/m^2 to A.
        self._realVolume = gap

        # Print information about the mesh generation in the output file !
        self._fMesh = "Mesh generated using ParallelPlate1D class"

    @property
    def gap(self) -> float:
        return self.__gap
    
    @gap.setter
    def gap(self, value: float) -> None:
        assert isinstance(value, float) and value > 0.0
        self.__gap = value
    
    @property
    def radius(self) -> float:
        return self.__radius
    
    @radius.setter
    def radius(self, value: float) -> None:
        assert isinstance(value, float) and value > 0.0
        self.__radius = value
    
    @property
    def nSteps(self) -> int:
        return self.__nSteps
    
    @nSteps.setter
    def nSteps(self, value: int) -> None:
        assert isinstance(value, int) and value > 0
        self.__nSteps = value

    def loadMesh(self):

        """Generate the 1D and fill variables such as dimensions, the number of
        cells and the name of the cells.
        """

        if gmsh.isInitialized():
            gmsh.finalize()
    
        self._meshData = self._generateMesh()

        topology       = self._meshData.mesh.topology
        self._dim      = topology.dim
        self._cellName = topology.cell_name()
        self._numCells = topology.index_map(self._dim).size_local 

    def _generateMesh(self):

        """Generate the 1D mesh using gmsh.

        Returns
        -------

        list[mesh, cell_tags, facet_tags]
            class mesh, the cell_tags and the facet tags.

        """

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)

        # Name of the model (not relevant)
        gmsh.model.add(f"{self.name}_SN{self.SN}")

        # Size of the elemets
        lc = self.gap / self.nSteps

        # Create two points separated by a distance equal to gap
        gmsh.model.geo.add_point(1,        0, 0, lc, 1)
        gmsh.model.geo.add_point(1, self.gap, 0, lc, 2)

        # Create a line
        gmsh.model.geo.add_line(1, 2, 1)

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(1)

        # Add the physical group:
        gmsh.model.addPhysicalGroup(1, [1], 1) # Active volume.

        # Add the High voltage and the collection electrode to apply
        # the boundary conditions
        gmsh.model.addPhysicalGroup(0, [2], 2)  # High voltage electrode.
        gmsh.model.addPhysicalGroup(0, [1], 3)  # Collection.

        meshData = dolfinx.io.gmsh.model_to_mesh(gmsh.model, 
                                            mpi4py.MPI.COMM_WORLD, 0, gdim = 3)

        # Include the collection and high voltage electrode tags:
        self.collectionTag   = 3
        self.highVoltageTag  = 2

        # Let the Chamber class know the "file" is open
        self._opened = True

        # End gmsh
        gmsh.finalize()

        return meshData
