import numpy as np
from matplotlib import pyplot as plt


# FadeIn
def LinearFadeIn(input, time_mS, sampleRate):
    """
    A “fade in” gradually increases the amplitude of the signal from 0 to 1 (unity gain). 
    """
    # Generate a linear fade in
    samples = int(( time_mS * sampleRate ) / 1000)
    linFadeIn = np.linspace(0, 1, samples)

    # Fade in
    output = input
    output[:len(linFadeIn)] = np.multiply(linFadeIn, output[:len(linFadeIn)])
    return output
 
# FadeOut
def LinearFadeOut(input, time_mS, sampleRate):

    """
    A “fade out” gradually decreases the gain of a signal from 1 (unity gain) to 0.
    """
    # Generate a linear fade out
    samples = int(( time_mS * sampleRate ) / 1000)
    linFadeOut = np.linspace(1, 0, samples)

    # Fade out
    output = input
    output[(len(input)-len(linFadeOut)):] = np.multiply(linFadeOut, output[(len(input)-len(linFadeOut)):] )
    return output

# FadeIn
def QuadraticFadeIn(input, time_mS, sampleRate):
    """
    A “fade in” gradually increases the amplitude of the signal from 0 to 1 (unity gain). 
    """
    # Generate a linear fade in
    samples = int(( time_mS * sampleRate ) / 1000)
    quadFadeIn = np.linspace(0, 1, int(samples/2.0))
    quadFadeIn = np.square(quadFadeIn)

    # Fade in
    output = input
    output[:len(quadFadeIn)] = np.multiply(quadFadeIn, output[:len(quadFadeIn)])
    return output
 
# FadeOut
def QuadraticFadeOut(input, time_mS, sampleRate):

    """
    A “fade out” gradually decreases the gain of a signal from 1 (unity gain) to 0.
    """
    # Generate a linear fade out
    samples = int(( time_mS * sampleRate ) / 1000)
    quadFadeOut = np.linspace(1, 0, samples)
    quadFadeOut = np.square(quadFadeOut)

    # Fade out
    output = input
    output[(len(input)-len(quadFadeOut)):] = np.multiply(quadFadeOut, output[(len(input)-len(quadFadeOut)):] )
    return output

# FadeIn
def ConcaveFadeIn(input, time_mS, sampleRate):
    """
    A “fade in” gradually increases the amplitude of the signal from 0 to 1 (unity gain). 
    """
    # Generate a linear fade in
    samples = int(( time_mS * sampleRate ) / 1000)
    quadFadeIn = np.linspace(0, 1, samples)
    quadFadeIn = np.power(quadFadeIn, 3)

    # Fade in
    output = input
    output[:len(quadFadeIn)] = np.multiply(quadFadeIn, output[:len(quadFadeIn)])
    return output
 
# FadeOut
def ConcaveFadeOut(input, time_mS, sampleRate):

    """
    A “fade out” gradually decreases the gain of a signal from 1 (unity gain) to 0.
    """
    # Generate a linear fade out
    samples = int(( time_mS * sampleRate ) / 1000)
    quadFadeOut = np.linspace(1, 0, int(samples/2.0))
    quadFadeOut = np.power(quadFadeOut, 3)

    # Fade out
    output = input
    output[(len(input)-len(quadFadeOut)):] = np.multiply(quadFadeOut, output[(len(input)-len(quadFadeOut)):] )
    return output