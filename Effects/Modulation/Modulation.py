import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

# AM
def AM(carrier, modFreq, sampleRate):
    """
    Modulate a signal and return it.-
    """

    # t
    t = np.linspace(0, (len(carrier)/sampleRate), len(carrier))

    # Synthesize LFO
    # lfo = 0.5 * (signal.square(2 * np.pi * modFreq * t) + 1)
    # lfo = 0.5 * (signal.sawtooth(2 * np.pi * modFreq * t) + 1)
    # lfo = 0.5 * (signal.sawtooth(2 * np.pi * modFreq * t) + 1)
    # lfo = lfo[::-1]
    lfo = 0.5 * (np.sin(2 * np.pi * modFreq * t) + 1)

    # AM
    output = np.multiply(carrier, lfo)

    return output
        

# Tremolo
def Tremolo(Input, rate, depth, sampleRate, lfo='Sine'):
    """
    Modulate a signal and return it.-
    """

    # t
    t = np.linspace(0, (len(Input)/sampleRate), len(Input))


    # Synthesize LFO 
    if lfo == 'Square':      
        _lfo = 0.5 * (signal.square(2 * np.pi * rate * t) + 1)
    if lfo == 'Sawtooth': 
        _lfo = 0.5 * (signal.sawtooth(2 * np.pi * rate * t) + 1)
    if lfo == 'Sawtooth_Inv': 
        _lfo = 0.5 * (signal.sawtooth(2 * np.pi * rate * t) + 1)
        _lfo = lfo[::-1]
    if lfo == 'Sine': 
        _lfo = 0.5 * (np.sin(2 * np.pi * rate * t) + 1)
    else:
        _lfo = 0.5 * (np.sin(2 * np.pi * rate * t) + 1)
    
    # AM
    output = depth * np.multiply(Input, _lfo)

    return output