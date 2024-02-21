import numpy as np
import random
import matplotlib.pyplot as plt
try:
    from scipy.special import comb
except:
    from scipy.misc import comb

from PIL import Image, ImageOps
import matplotlib.pyplot as plt


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.
       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000
        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals


def nonlinear_transformation(raw):
    slices = norm(raw.copy())

    points_1 = [[-1, -1], [-1, -1], [1, 1], [1, 1]]
    xvals_1, yvals_1 = bezier_curve(points_1, nTimes=100000)
    xvals_1 = np.sort(xvals_1)

    points_2 = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
    xvals_2, yvals_2 = bezier_curve(points_2, nTimes=100000)
    xvals_2 = np.sort(xvals_2)
    yvals_2 = np.sort(yvals_2)

    points_3 = [[-1, -1], [-0.5, 0.5], [0.5, -0.5], [1, 1]]
    xvals_3, yvals_3 = bezier_curve(points_3, nTimes=100000)
    xvals_3 = np.sort(xvals_3)

    points_4 = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
    xvals_4, yvals_4 = bezier_curve(points_4, nTimes=100000)
    xvals_4 = np.sort(xvals_4)
    yvals_4 = np.sort(yvals_4)

    points_5 = [[-1, -1], [-0.75, 0.75], [0.75, -0.75], [1, 1]]
    xvals_5, yvals_5 = bezier_curve(points_5, nTimes=100000)
    xvals_5 = np.sort(xvals_5)

    """
    slices, nonlinear_slices_2, nonlinear_slices_4 are source-similar images
    nonlinear_slices_1, nonlinear_slices_3, nonlinear_slices_5 are source-dissimilar images
    """
    nonlinear_slices_1 = np.interp(slices, xvals_1, yvals_1)
    nonlinear_slices_1[nonlinear_slices_1 == 1] = -1
    
    nonlinear_slices_2 = np.interp(slices, xvals_2, yvals_2)

    nonlinear_slices_3 = np.interp(slices, xvals_3, yvals_3)
    nonlinear_slices_3[nonlinear_slices_3 == 1] = -1

    nonlinear_slices_4 = np.interp(slices, xvals_4, yvals_4)

    nonlinear_slices_5 = np.interp(slices, xvals_5, yvals_5)
    nonlinear_slices_5[nonlinear_slices_5 == 1] = -1

    return renorm(slices), renorm(nonlinear_slices_1), renorm(nonlinear_slices_2), \
           renorm(nonlinear_slices_3), renorm(nonlinear_slices_4), renorm(nonlinear_slices_5)


def norm(slices):    
    return (slices * 2) - 1

def renorm(slices):
    return (slices + 1) / 2


