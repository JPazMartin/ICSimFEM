import matplotlib.pylab as plt

from ICSimFEM import Reader

class Plotter1D:

    def __init__(self, reader: Reader, each = 1):

        self.reader = reader
        self.each   = each

        self.reader.openFile()

        # matplotlib.use("qt5agg")
        assert self.reader.dimension == 1, "File contain a high dimension mesh."

    @property
    def each(self) -> int:
        return self.__each
    
    @each.setter
    def each(self, value: int) -> None:
        assert value > 0 and isinstance(value, int)
        self.__each = value

    def plotAnimation(self) -> None:

        self.reader.openFile()
        nSpecies = self.reader.nSpecies

        # plt.ion()
        fig, ax = plt.subplots(1, 2, figsize = (12, 6))

        solution = self.reader.getNextStep()
        i = 0; j = 0
        while solution != False:
            
            if i % self.each == 0:

                fig.suptitle(fr"Time in simulation = {solution[0] * 1E6:.2f} $\upmu$s")

                ax[0].cla()
                ax[1].cla()

                for i in range(nSpecies):
                    ax[0].plot(self.reader.x[:, 0] * 1E3, solution[1][i], linewidth = 1.5)
                    ax[1].plot(self.reader.x[:, 0] * 1E3, solution[2][:, 0] / 1E3, "-k", linewidth = 1.5)

                ax[0].set_xlabel("Distance (mm)")
                ax[1].set_xlabel("Distance (mm)")

                ax[0].set_ylabel("Carrier density (m$^{-3}$)")
                ax[1].set_ylabel("Electric field (kV$\,$m$^{-1}$)")

                fig.tight_layout(rect = [0, 0, 1, 1.05])
                plt.pause(1E-3)

                fig.savefig(f"solution/step_{j}.png", dpi = 250)

                j += 1

            solution = self.reader.getNextStep()

            i += 1

        self.reader.close()


        


