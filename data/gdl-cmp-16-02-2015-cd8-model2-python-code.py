# == Presentation of the CD8 model (model 2) ==

import numpy 
from pylab import plot, grid, legend, ylabel, yticks, yscale, ylim, xlim, xticks, xlabel, savefig
import math

lambda1 = 0.10
lambda2 = 0.5
kappa1 = 1050000.0
kappa2 = 460123.0
mu3 = 0.027
xi1 = 0.09
xi2 = 0.4


def dX_dt(X, t=0):
    """ Return the growth rate of cd8 T cell populations """
    return numpy.array([lambda1*X[0]*(1.0-X[0]/kappa1) - xi1*X[0], lambda2*X[1]*(1.0-X[1]/kappa2) - xi2*X[1] + xi1*X[0], -mu3*X[2] + xi2*X[1]])

#
# Now we will use the scipy.integrate module to integrate the ODEs.
# This module offers a method named odeint, very easy to use to integrate ODEs:
#

from scipy import integrate

t = numpy.linspace(0.0, 60.0, 5000)              # time
X0 = numpy.array([100000.0, 0.0, 0.0])   # IC for Robey's manuscript 10^5 initial cells

X, infodict = integrate.odeint(dX_dt, X0, t, full_output=True)
infodict['message']                     # >>> 'Integration successful.'

#
# `infodict` is optional, and you can omit the `full_output` argument if you don't want it.
# Type "info(odeint)" if you want more information about odeint inputs and outputs.
# 
# We can now use Matplotlib to plot the evolution of the populations:
#

stem, intermediate, terminal = X.T

plot(t, stem, 'g-', label='T$_{\\rm mem}$ cells', linewidth=2.0)
plot(t, intermediate, 'b-', label='T$_{\\rm int}$ cells', linewidth=2.0)
plot(t, terminal, 'r-', label='T$_{\\rm eff}$ cells', linewidth=2.0)
yscale('log')
xlim((0, 60.0))
ylim((1, 20000000.0))
grid(which='major')

savefig('cd8modelsteady05022015-cells-initial-5.pdf', dpi=1200)
