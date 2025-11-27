"""
    Class to describe the species in the simulation
"""

import numpy as np

from .utils   import readFile, constants, ktp
from typing   import Callable
from ICSimFEM import Logger

class Specie:
    
    """Class to describe the species in the simulation.    
    """
    
    def __init__(self, name: str = "", charge: int = 1) -> None:
        
        """
        Parameters
        ----------

        name : str
            Name of the specie
        charge : int
            Charge associated to the specie
        """
        
        self.name      = name
        self.charge    = charge

        # Set the fraction of ionization. It can vary from 0 to 1
        self.__fIonization = 0
        
        # Type of mobility
        self.mobType  = 1
        self.diffType = False

        self.filePathVelocity  = ""
        self.filePathDiffusion = ""

    @property
    def fIonization(self):
        return self.__fIonization
    
    @fIonization.setter
    def fIonization(self, value: float):
        assert value <= 1 and value >= 0, "Fraction of ionization must be "
        "between 0 and 1"
        self.__fIonization = value

    @property
    def charge(self):
        return self.__charge
    
    @charge.setter
    def charge(self, value: int):
        self.__charge = value

    def setConstantMobility(self, mobility: Callable[[float, float, float],
                                                      float]) -> None:
        """
        Set a constant value of the mobility with respect of the electric field.
        It must be a function of  the temperature (in degC), pressure (in hPa)
        and relative humidity in (%).

        Parameters
        ----------
        mobility: function
            A function of relative humidity (in %), pressure (in hPa) and 
            temperature (in celsius degree). The order must be strict. The
            function must return the value of the mobility in m^2/(Vs).
        """

        self.mobType = 1
        self.mobility = mobility
        
        return
        
    def setConstantMobilitySTP(self, mobility: float) -> None:
        
        """
        Set a constant value of the mobility with respect of the electric field.
        Normal scaling rules of the mobility will be used in this case assuming
        the constant value is assumed to be given at 20 C and 1013 hPa.

        Parameters
        ----------
        mobility : float
            Value of the mobility in m^2/(Vs).
        """

        self.mobType  = 2
        self.mobility = lambda h, p, t: mobility * ktp(t, p)
        
        return
    
    def setDiscreteVelocityFromUser(self, xValues: np.array,
                                     yValues: np.array) -> None:
        """
        Set a look up table for the calculation of the mobility with respect of 
        the electric field.

        Parameters
        ----------
        xValues : np.array
            Values of the electric field in V/m.
        yValues : np.array
            Values of the velocity in m/s.
        """

        self.mobType = 3

        self.xVel = np.array(xValues)
        self.yVel = np.array(yValues)

        return

    
    def setDiscreteVelocityFromFile(self, filePath) -> None:
        
        """
        Set a look up table for the calculation of the mobility with respect to
        the electric field from a file.

        Parameters
        ----------
        filePath : str
            Name of the file.
        """

        xVel, yVel            = readFile(filePath)
        self.filePathVelocity = filePath

        self.setDiscreteVelocityFromUser(xVel, yVel)

        return

    def setNerstTowsendLDiffusion(self) -> None:
        
        """
        Use the Nerst-Townsend relation for the calculation of the longitudinal 
        diffusion coefficient.
        """

        self.diffType = 1

        return

    def setConstantLDiffusion(self, diffusionCoefficient: float) -> None:
        
        """
        Set a constant value of the lognitudinal diffusion coefficient with 
        respect to the electric field strength.
        
        Parameters
        ----------
        diffusionCoefficient : float
            Value of the diffusion in m^2/s.
        """

        self.diffType             = 2
        self.diffusionCoefficient = diffusionCoefficient

        return
    
    def setDiscreteLDiffusion(self, xValues: np.array, yValues: np.array) -> None:
        
        """
        Set look up table for the calulation of the longitudinal diffusion with
        respect to the electric field.

        Parameters
        ----------
        xValues : np.array
            Array with the values of the electric field in V/m.

        yValues : np.array
            Array with the values of the longitudinal diffusion coefficient in
            m^2/s.
        """

        self.diffType = 3
        
        self.xDiff = xValues
        self.yDiff = yValues

        return
    
    def setDiscreteLDiffusionFromFile(self, filePath: str) -> None:
        
        """
        Set a look up table for the calculation of the longitudinal diffusion
        with respect to the electric field from a file.

        Parameters
        ----------
        filePath : str
            Name of the file.
        """

        self.xDiff, self.yDiff = readFile(filePath)
        self.setDiscreteLDiffusion(self.xDiff, self.yDiff)

        return
        

    def _nerstTownsend(self, t: float) -> float:

        """Returns the value of the diffusion coefficient using the 
        Nerst-Townsend relationship.

        Parameters
        ----------
        t : float
            Temperature in Celsisus degree.

        Returns
        -------
        float
            Nerst townsed relation between mobility and diffusion

        """
        return constants.kB * (273.15 + t) / constants.electronCharge
        
    def getVelocity(self, h: float, p: float, t: float) -> Callable[
                                                        [float, float], float]:
        
        """Returns the velocity of the specie for a certain values of temperature, 
        pressure and relative humidity as a function of the electric field and
        the electric field module
        
        Parameters
        ----------
        h : float
            Relative humidity in %
        p : float
            Pressure in hPa.
        t : float
            Temperature in degC.
        """
        
        # If velocity is supplied.
        if self.mobType == 3:

            def getVel(E, E_mod):

                # Reduced electric field.
                xRed = self.xVel / ktp(t, p)

                vel_mod = np.interp(E_mod, xRed, self.yVel) * self.charge
                vel     = E * (vel_mod / E_mod)
                return vel.T.flatten()
            
            return getVel

        # If mobility is supplied.
        return lambda E, EMod: self.mobility(h, p, t) * self.charge * E
    
    def getDiffusion(self, h: float, p: float, t: float):
        
        """Returns the longitudinal diffusion of the specie for a certain 
        values of temperature, pressure and relative humidity as a function of
        the electric field and the electric field module. This implementation
        of the longitudinal diffusion is only valid for 1D and 2D coordinate
        system.

        TODO: Implement the full diffusion matrix
        
        Parameters
        ----------
        h : float
            Relative humidity in %
        p : float
            Pressure in hPa.
        t : float
            Temperature in degC.
        """
        
        match self.diffType:
            case 1:
                
                if self.mobType == 3:

                    self.xDiff  = self.xVel[self.xVel > 0]
                    self.yDiff  = self.yVel[self.xVel > 0] / self.xDiff 
                    self.yDiff *= self._nerstTownsend(t) / ktp(t, p)
                
                if self.mobType == 1 or self.mobType == 2:

                    def getDiff(E, E_mod):
                        
                        D = self.mobility(h, p, t) * self._nerstTownsend(t)

                        # Calculate the angle
                        alpha = np.arctan(E[1] / E[0])

                        # Compute the components of the diffusion matrix
                        d1    = D * np.cos(alpha)**2
                        d2    = D * np.sin(alpha)**2
                        d12   = 2 * D * np.cos(alpha) * np.sin(alpha)

                        return [d1, d2, d12]
                    
                    return getDiff

            case 2:
            
                def getDiff(E, E_mod):

                    # Calculate the angle
                    alpha = np.arctan(E[1] / E[0])

                    # Compute the components of the diffusion matrix
                    d1    = self.diffusionCoefficient * np.cos(alpha)**2
                    d2    = self.diffusionCoefficient * np.sin(alpha)**2
                    d12   = 2 * self.diffusionCoefficient * (
                        np.cos(alpha) * np.sin(alpha))

                    return [d1, d2, d12]
                
                return getDiff


        def getDiff(E, E_mod):
            
            D = np.interp(E_mod, self.xDiff / ktp(t, p), self.yDiff * ktp(t, p))

            # Calculate the angle
            alpha = np.arctan(E[1] / E[0])

            # Compute the components of the diffusion matrix
            d1    = D * np.cos(alpha)**2
            d2    = D * np.sin(alpha)**2
            d12   = 2 * D * np.cos(alpha) * np.sin(alpha)

            return [d1, d2, d12]
            
        return getDiff
        

    def printInfo(self, logger: Logger):
        
        """Prints the information of the beam to the output file.

        Parameters
        ----------
        logger: Logger
            Object managing the output.
        """

        mobDescriptor = ""
        match self.mobType:
            case 1:
                mobDescriptor = "Constant mobility and custom scaling rules"
            case 2: 
                mobDescriptor = "Constant mobility and standard scaling rules"
            case 3:
                mobDescriptor = "Variable mobility and standard scaling rules"

        diffDescriptor = ""
        match self.diffType:
            case 1:
                diffDescriptor = "Nerst-Towsend relationship for diffusion"
            case 2:
                diffDescriptor = "Constant diffusion"
            case 3:
                diffDescriptor = "Variable diffusion with electric field"
            case _:
                diffDescriptor = "No longitudinal diffusion"
        
        logger.info(f'\n  -> {self.name}:')
        logger.printData(5 * " " + "Charge"             ,
                          f"{self.charge}")
        logger.printData(5 * " " + "Mobility definition", 
                         f"{mobDescriptor}")
        if self.mobType == 3 and self.filePathVelocity != "":
            logger.printData(16 * " " + "- File", 
                             f"{self.filePathVelocity}")
        logger.printData(5 * " " + "Longitudinal diffusion",
                          f"{diffDescriptor}")
        if self.diffType == 3 and self.filePathDiffusion != "":
            logger.info(16 * " " + "- File", 
                        f"{self.filePathDiffusion}")
        logger.printData(5 * " " + "Fraction of ionization in gas", 
                         f"{self.fIonization}")
