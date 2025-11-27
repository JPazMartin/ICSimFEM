"""
    Class used to manage the information output by the simulation.
"""

import datetime
import ICSimFEM
import dolfinx
import gmsh
import sys

class Logger:

    """Class used to manage the output of the simulation
    """
    
    def __init__(self, fileName: str, printScreen: bool):

        """
        Parameters
        ----------
        fileName : str
            Where the information of the information of the simulation will be 
            printed.
        printScreen: bool
            If true, prints information to the screen
        """
        
        # Save the name and open the file in write mode (deletes previous file)
        self.fileName     = fileName
        self.f            = open(fileName, "w+")
        self.printScreen  = printScreen

        # Maximum space to print in data mode.
        self.spaceForPrinting = 40

        # Maximum size allowed in data mode.
        self.maxSize          = 100

        self.start()

    def start(self):

        """It starts printing the header of the file with the program version, 
        python version, dolfinx version and GMSH version."""

        t_now = datetime.datetime.now()
        t_now_s = t_now.strftime("%d/%m/%Y, %H:%M:%S")

        self.info('=' * self.maxSize)
        self.info(f"  ICSimFEM code version {ICSimFEM.__version__}" +
                   (self.maxSize - 50) * " " + f"{t_now_s}  ")
        
        self.info('=' * self.maxSize)
        self.printData("Python version", f"{sys.version[:7]}")
        self.printData("Dolfinx version", f"{dolfinx.__version__}")
        self.printData("GMSH version", f"{gmsh.__version__}")
        self.info('=' * self.maxSize)
        self.info('\n')
        
    def info(self, string):

        """Print to the file in "information" mode.
        """
        
        if (self.printScreen): print(string)
        f_ax = open(self.fileName, "a")
        f_ax.write(string + "\n")
        f_ax.close()

    def printData(self, string: str, value: str) -> None:

        """Print to the file in "data" mode: It print a name (string) with a 
        value separated by dots. The maximum size is given by spaceForPrinting.
        The output should be something like:

        string..........................value
        
        Parameters
        ----------
        string : str
            String corresponding to value
        value: str
            Value formatted as string.
        """

        # If the string is larger than spaceForPrinting it is not possible to 
        # print it.
        nDots = self.spaceForPrinting - len(string)

        assert nDots >= 0, "Formatting error: Not able to print this string"

        nLeft  = self.maxSize - self.spaceForPrinting
        nLines = int(len(value) / nLeft) + 1

        # If the string is larger that spaceForPrinting, divide it.
        self.info(string + nDots * '.' + value[:nLeft])

        for i in range(nLines - 1):

            idxIn  = (i + 1) * nLeft
            idxFin = (i + 2) * nLeft

            self.info(self.spaceForPrinting * " " + value[idxIn:idxFin])
    
    def printHeader(self, name: str):

        """Print the header of each section of the simulation underliying it 
        with some equal symbols "=" with the same size as name.

        Parameters
        ----------
        name: str
            Name of the corresponding section.
        """

        self.info(name)
        self.info(len(name) * "=" + "\n")

    def close(self):
        
        """Close the file printing the time and the actual time
        """

        t_now = datetime.datetime.now()
        t_now_s = t_now.strftime("%d/%m/%Y, %H:%M:%S")
        self.info('=' * self.maxSize)

        self.info(f"  End of simulation" + (self.maxSize - 40) * " " 
                  + f"{t_now_s}  ")
        self.info('=' * self.maxSize)
    
        self.f.close()