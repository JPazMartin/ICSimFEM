
import os
import datetime
import shutil
import numpy as np

HOMEPATH = os.environ["HOME"]
cache    = []

# Put recomended values from the books
class Constants:

    """Class defining commonly used physical constants
    """
    
    @property
    def electronCharge(self) -> float:
        return 1.602176634E-19 # C
    
    @property
    def averageEnergyPerIonPair(self) -> float:
        return 33.97 * self.electronCharge # J
    
    @property
    def vacuumPermitivity(self) -> float:
        return 8.8541878188E-12 # F/m
    
    @property
    def referenceTemperature(self) -> float:
        return 20 # degC
    
    @property
    def kB(self) -> float:
        return 1.380649E-23 # J K^{-1}
    
    @property
    def referencePressure(self) -> float:
        return 1013.25 # hPa
        
    @property
    def airDensity(self) -> float:
        """Air density at 20 degC and 1013.25 hPa"""
        return 1.204 # kg/m^3

constants = Constants()


def ktp(T: float, P: float):

    """Temperature and pressure ideal correction
    
    Parameters
    ----------
    T : float
        Temperature value in degC
    P : float
        Pressure value in hPa
    """

    kt = (T + 273.15) / (constants.referenceTemperature + 273.15)
    kp = constants.referencePressure / P

    return kt * kp


def jitOptions() -> dict:

    number = np.random.randint(0, 1E6)

    ts = int(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))
    folder_cache = f"{HOMEPATH}/.ICSimCache/ICSimCache_{number}_cache_{ts}"
    opts = {}
    opts["cache_dir"]               = folder_cache
    opts["cffi_extra_compile_args"] =  ["-Ofast", "-march=native"]

    cache.append(folder_cache)

    return opts

def generateCacheOptions() -> dict:

    number = np.random.randint(0, 1E6)

    ts = int(datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"))
    folder_cache = f"{HOMEPATH}/.ICSimCache/ICSimCache_{number}_cache_{ts}"
    opts = {}
    opts["cache_dir"]               = folder_cache
    opts["cffi_extra_compile_args"] =  ["-Ofast", "-march=native"]

    cache.append(folder_cache)

    return opts

def deleteCache():

    """Delete the generated cache during calculations
    """

    for path in cache:
        shutil.rmtree(path, ignore_errors = True)

def readFile(filePath: str) -> None:

    x = []; y = []
    f = open(filePath, "r")
    for line in f:
        if line[0] == "#": pass
        else:
            x.append(float(line.split(",")[0]))
            y.append(float(line.split(",")[1]))
    f.close()

    return np.array(x), np.array(y)
