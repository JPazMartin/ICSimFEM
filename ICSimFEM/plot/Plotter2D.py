import matplotlib.pylab  as plt
import matplotlib.tri    as tri
import matplotlib.ticker as ticker
import numpy             as np

import os
import shutil

from ICSimFEM import Reader

class Plotter2D:

    def __init__(self, reader: Reader, each = 1):

        self.reader = reader
        self.each   = each

        self.reader.openFile()

        # matplotlib.use("qt5agg")
        assert self.reader.dimension == 2, "File contain a high dimension mesh."

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
        fig, ax = plt.subplots(1, 5, figsize = (20, 8))

        triangulation = tri.Triangulation(self.reader.x[:, 0], self.reader.x[:, 1],
                                          self.reader.cells)
        zeros = np.zeros(len(self.reader.x[:, 0]))

        field = [ax[i].tripcolor(triangulation, zeros, cmap = "plasma",
                                  shading = 'gouraud') for i in range(4)]

        ax[0].title.set_text("Positive ions")
        ax[1].title.set_text("Negative ions")
        ax[2].title.set_text("Electrons")
        ax[3].title.set_text("Electric field")

        [axis.axis('off') for axis in ax]

        def fmt(x, pos):
            a, b = '{:.1e}'.format(x).split('e')
            b = int(b)
            return r'${} \times 10^{{{}}}$'.format(a, b)

        bbox_ax = ax[0].get_position()

        cbar_im1a_ax = fig.add_axes([0.80, bbox_ax.y0, 0.02, bbox_ax.y1-bbox_ax.y0])
        cbar_im1a = plt.colorbar(field[2], cax=cbar_im1a_ax, format = ticker.FuncFormatter(fmt))
        cbar_im1a.set_label('Carrier densities (m$^{-3}$)', rotation = 90)
        cbar_im1a.ax.tick_params(labelsize = 12) 

        cbar_im1a_ax = fig.add_axes([0.92, bbox_ax.y0, 0.02, bbox_ax.y1-bbox_ax.y0])
        cbar_im1a = plt.colorbar(field[3], cax=cbar_im1a_ax, format = ticker.FuncFormatter(fmt))
        cbar_im1a.ax.tick_params(labelsize = 12) 

        cbar_im1a.set_label('Electric field magnitude (V m$^{-1}$)', rotation = 90)
        
        shutil.rmtree("frames", ignore_errors = True)
        os.mkdir("frames")

        solution = self.reader.getNextStep("frames/frame0")

        j = 0
        while solution != False:

            if j % self.each == 0:

                fig.suptitle(fr"Time in simulation = {solution[0] * 1E6:.2f} $\mu$s")

                maximum = np.max([solution[1][0], solution[1][1], solution[1][2]])
                
                for i in range(nSpecies):
                    field[i].set_array(solution[1][i])
                    field[i].set_clim(0, maximum)

                Emod = np.sqrt(solution[2][:, 0]**2 + solution[2][:, 1]**2 +
                               solution[2][:, 2]**2)

                field[3].set_array(Emod)
                field[3].set_clim(0, np.max(Emod))

                fig.tight_layout()
                plt.pause(1E-3)
                
            solution = self.reader.getNextStep(f"frames/frame{j}")

            j += 1

        self.reader.close()