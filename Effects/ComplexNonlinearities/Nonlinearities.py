"""
References:

- https://jatinchowdhury18.medium.com/complex-nonlinearities-episode-0-why-4ad9b3eed60f
- https://jatinchowdhury18.medium.com/complex-nonlinearities-episode-1-double-soft-clipper-5ce826fa82d6
- https://jatinchowdhury18.medium.com/complex-nonlinearities-epsiode-2-harmonic-exciter-cd883d888a43?source=---------2------------------
"""

import numpy as np
from scipy import signal
from sounddevice import _InputOutputPair

# SoftClipperSigmoid
def SoftClipperSigmoid(x):
    """
    User defined softmax for Simple Soft Clipper.-
    """
    #
    if x>=1:
        return 1
    elif x<=-1:
        return -1
    else:
        return (3.0/2.0) * ((x-x**3)/3)



# SoftClipper
def SoftClipper(Input):
    """
    Soft clipper using sigmoid.-
    """
    output = np.zeros(len(Input))
    for i in range(len(Input)):
        output[i] = SoftClipperSigmoid(Input[i])
    return output

# DoubleSoftClipperSigmoid
def DoubleSoftClipperSigmoid(x, upperLim, lowerLim, slope, xOffFactor, upperSkew, lowerSkew):
    """
    User defined softmax for Double Soft Clipper.-
    """
    xOff = (1.0/slope) * ((slope)**xOffFactor)
    #
    if x>0:
        x = (x - xOff) * upperSkew
        if x>=(1/slope):
            return upperLim
        elif x<=(-1/slope):
            return 0
        else:
            return ( (3.0/2.0)*(upperLim)*(((x*slope)**3)/3.0) /2.0 + (upperLim/2.0) )
    #
    else:
        x = (x + xOff) * lowerSkew
        if x>=(1/slope):
            return 0
        elif x<=(-1/slope):
            return lowerLim
        else:
            return ( (3.0/2.0)*(-lowerLim)*(((x*slope)**3)/3.0) /2.0 + (lowerLim/2.0) )
    


# DoubleSoftClipper
def DoubleSoftClipper(Input, xRange=2, upperLim=1, lowerLim=-1, slope=1, xOffFactor=1, upperSkew=1, lowerSkew=1):
    """
    Double Soft clipper using sigmoid.-
    """
    output = np.zeros(len(Input))
    for i in range(len(Input)):
        output[i] = DoubleSoftClipperSigmoid(x=Input[i], upperLim=upperLim, lowerLim=lowerLim, slope=slope, xOffFactor=xOffFactor, upperSkew=upperSkew, lowerSkew=lowerSkew)
    return output