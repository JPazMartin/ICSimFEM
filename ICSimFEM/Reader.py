"""
    Class used to read the output files of the simulation.
"""

import struct
import os 
import dolfinx
import ufl
import basix

import numpy as np

from mpi4py import MPI


class Reader:

    """Class used to read the output files of the simulation.
    """

    def __init__(self, fileName: str) -> None:

        """
        Parameters
        ----------
        fileName : str
            Name of the file. It is expected to provide the name of the
            .ICSimlog file with or without extension.
        """

        self.fileName = fileName

        # Get the name of the filename without extension.
        self.pathfileName = self.fileName.split("/")
        if len(self.pathfileName) >= 2:
            self.truefileName = self.pathfileName[-1]
            self.pathfileName = "/".join(self.pathfileName[:-1])
        else:
            self.truefileName = self.pathfileName[0]
            self.pathfileName = ""

        # Remove the extension if necessary.
        self.truefileName = self.truefileName.split(".ICSimlog")
        if len(self.truefileName) >= 2:
            self.truefileName = self.truefileName[-2]
        else:
            self.truefileName = self.truefileName[0]

        self.readLog()

    def readLog(self):

        """Read the log file and load the dose per pulse, the charge collection
        efficiency, the voltage, the collected charge and the elapsed time.
        """

        ## Properties of the simulation:
        self.dpp = 0
        self.CCE = 0
        self.V   = 0
        self.Q   = 0

        self.elapsedTime = 0

        # Open the file and iterate to find all the keys:
        f = open(f"{self.pathfileName}/{self.truefileName}.ICSimlog", "r")
        line = f.readline()

        while line != "":
            
            # Dose per pulse:
            if line.startswith("Dose per pulse"):
                self.dpp = float(line.split("..")[-1].split("Gy")[0])
            
            # Voltage:
            if line.startswith("Applied voltage"):
                self.V = float(line.split("..")[-1].split("V")[0][1:])

            # Total induced charge:
            if line.startswith("1.- Induced signal:"):

                line = f.readline()
                while line.startswith("   Q"):
                    self.Q += float(line.split("=")[-1][:-3])
                    line = f.readline()

            # Charge collection efficiency:
            if line.startswith("   CCE"): self.CCE = float(line[9:])

            # Total elapsed time:
            if line.startswith("Total elapsed time in simulation:"):
                self.elapsedTime = float(line[33:].split("s")[0])

            line = f.readline()

        f.close()

    def openFile(self):

        """Open the binary file and load the basic information
        """

        self.f = open(f"{self.pathfileName}/Data_{self.truefileName}.ICSimdata",
                       'rb')

        # Get the size of the file and point to the beginning of the file.
        self.fileSize = self.f.seek(0, os.SEEK_END)
        self.f.seek(0, 0)

        # Get the values.
        self.nSpecies  = struct.unpack('I', self.f.read(4))[0]
        self.cellType  = struct.unpack('H', self.f.read(2))[0]
        self.dimension = struct.unpack('H', self.f.read(2))[0]
        self.nPoints   = struct.unpack('I', self.f.read(4))[0]
        self.nCells    = struct.unpack('I', self.f.read(4))[0]

        # Create mesh
        self.x     = np.zeros((self.nPoints, 3))
        self.cells = np.zeros((self.nCells, self.cellType + 1), dtype = np.int32)

        # Get the coordinates
        for i in range(3):
            for j in range(self.nPoints):
                self.x[j, i] = struct.unpack('<f', self.f.read(4))[0]

        # Get the cells
        for i in range(self.cellType + 1):
            for j in range(self.nCells):
                self.cells[j, i] = struct.unpack('I', self.f.read(4))[0]

        self.cell     = basix.CellType(self.cellType)
        self.element  = basix.ufl.element("CG", self.cell, 1,
                                           shape = (self.dimension,))
        self.domain   = ufl.Mesh(self.element)
        self.mesh     = dolfinx.mesh.create_mesh(MPI.COMM_WORLD, self.cells,
                                                  self.x, self.domain)
        self.idx      = self.mesh.geometry.input_global_indices
        
        # Generate spaces to load data
        self.P1  = basix.ufl.element("Lagrange", self.mesh.topology.cell_name(),
                                     1)
        V        = dolfinx.fem.functionspace(self.mesh, self.P1)
        V2       = dolfinx.fem.functionspace(self.mesh, ("CG", 1, (3, )))
        self.u   = dolfinx.fem.function.Function(V)
        self.u_E = dolfinx.fem.function.Function(V2)

        # Read weighted field and weighted potential:
        self.Vw = [struct.unpack('<f', self.f.read(4))[0] for i in 
                   range(self.nPoints)]
        self.Ew = [[struct.unpack('<f', self.f.read(4))[0] for j in range(3)] 
                   for i in range(self.nPoints)]

        # Read initial electric field and potential:
        V = [struct.unpack('<f', self.f.read(4))[0] for i in range(self.nPoints)]
        E = [[struct.unpack('<f', self.f.read(4))[0] for j in range(3)] 
             for i in range(self.nPoints)]

    def getNextStep(self, saveFrame = ""):

        """Function to read each step saved in the file

        Parameters
        ----------
        saveframe : str
            Path to save the data in .xmdf format
        """

        # Read only if still bites in the file.
        if (self.fileSize - self.f.tell()) > 0:

            # Read time in s.
            t   = struct.unpack('<f', self.f.read(4))[0]

            # Read the charge densities in m^{-3}
            n   = [[struct.unpack('<f', self.f.read(4))[0] 
                    for i in range(self.nPoints)] for j in range(self.nSpecies)]
            
            # Read potential and perturbed electric field.
            V_t = [struct.unpack('<f', self.f.read(4))[0] 
                   for i in range(self.nPoints)]
            E_t = [[struct.unpack('<f', self.f.read(4))[0] for j in range(3)] 
                   for i in range(self.nPoints)]
            
            E_t = np.array(E_t)

            # Sort the electric field
            E_t_sort = np.copy(E_t)
            E_t_sort[:, 0] = E_t_sort[self.idx, 0]
            E_t_sort[:, 1] = E_t_sort[self.idx, 1]
            E_t_sort[:, 2] = E_t_sort[self.idx, 2]

            # If saveFrame is provided, save it in xdmf format.
            if saveFrame != "":
                
                for i in range(self.nSpecies):
                    self.u.x.array[:] = np.array(n[i])[self.idx]
                    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{saveFrame}_specie{i}.xdmf", "w") as xdmf:
                        xdmf.write_mesh(self.mesh)
                        xdmf.write_function(self.u)

                self.u_E.x.array[:] = E_t[self.idx].flatten()
                with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{saveFrame}_field{i}.xdmf", "w") as xdmf:
                    xdmf.write_mesh(self.mesh)
                    xdmf.write_function(self.u_E)

                self.u.x.array[:] = np.array(V_t)[self.idx]
                with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{saveFrame}_voltage{i}.xdmf", "w") as xdmf:
                    xdmf.write_mesh(self.mesh)
                    xdmf.write_function(self.u)
            
            return t, n, E_t, V_t
                
        else:
            
            # If there is no data in the file, close it
            self.close() 
            return False

    def close(self):

        """Close the file.
        """

        self.f.close()
