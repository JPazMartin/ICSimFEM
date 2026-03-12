"""
    Class to save data into the .ICSimdata file
"""

import dolfinx
import basix
import struct
import numpy as np
import array

from .utils   import jitOptions
from .Chamber import Chamber

class Saver:

    """Class to save data into the .ICSimdata file
    """

    def __init__(self, fileName: str, chamber: Chamber, nSpecies: int):

        """
        Parameters
        ----------
        fileName: str
            Name of the file to save the data
        chamber: Chamber
            Chamber class.
        nSpecies: int
            Number of charge species simulated.
        """

        self.fileName   = fileName
        self.mesh       = chamber._meshData[0]
        self.nSpecies   = nSpecies
        self.dim        = chamber.dim

        # Open the file
        self.file = open(f'{self.fileName}.ICSimdata', 'wb')

        # Generate spaces to interpolate/save functions
        self.P1 = basix.ufl.element("CG", self.mesh.topology.cell_name(), 1)
        self.el = basix.ufl.mixed_element([self.P1] * self.nSpecies)

        self.V  = dolfinx.fem.functionspace(self.mesh, self.el)
        self.V2 = dolfinx.fem.functionspace(self.mesh, ("CG", 1, (3, )))

        self.u  = dolfinx.fem.function.Function(self.V)
        self.u2 = dolfinx.fem.function.Function(self.V2)

        self.writeHeader()
        self.writeMesh()

    def writeHeader(self):

        """Write the header with the information of the file.
        """

        self.file.write(struct.pack("I", self.nSpecies))

    def writeMesh(self):

        """Write the mesh into the file.
        """

        n = 0

        # Write the cell type.
        if self.mesh.topology.cell_type == dolfinx.cpp.mesh.CellType.interval:
            n = 1
        if self.mesh.topology.cell_type == dolfinx.cpp.mesh.CellType.triangle:
            n = 2
        if self.mesh.topology.cell_type == dolfinx.cpp.mesh.CellType.tetrahedron:
            n = 3

        points = self.mesh.geometry.x

        numFacets = self.mesh.topology.index_map(self.dim).size_local
        geometryEntities = dolfinx.cpp.mesh.entities_to_geometry(self.mesh._cpp_object,
                             self.dim, np.arange(numFacets, dtype=np.int32), False)

        self.file.write(struct.pack("H", n))
        self.file.write(struct.pack("H", self.dim))
        self.file.write(struct.pack("I", len(points)))
        self.file.write(struct.pack("I", numFacets))
        
        [array.array('f', points[:, i]).tofile(self.file) for i in range(3)]
        [array.array('I', geometryEntities[:, i]).tofile(self.file) 
         for i in range(n + 1)]
    
    def writeSpecies(self, u, t):

        """Writhe the charge densities into the file
        """
        
        self.file.write(struct.pack("f", t))
        
        for i in range(self.nSpecies):

            self.u.sub(i).interpolate(u.sub(i))
            array.array('f', self.u.sub(i).collapse().x.array).tofile(self.file)

    def writeField(self, uV, uE):

        """Write the electric field and potential into the file
        """

        self.u.sub(0).interpolate(uV)
        array.array('f', self.u.sub(0).collapse().x.array).tofile(self.file)

        self.u2.interpolate(uE)
        array.array('f', self.u2.x.array).tofile(self.file)

    def close(self):

        """Delete cache and close file
        """

        self.file.close()
