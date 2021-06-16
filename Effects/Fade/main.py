#
# Imports
#
from scipy import signal
from matplotlib import pyplot as plt
from matplotlib import style
from scipy import signal
import numpy as np
import math
import pyaudio
from scipy.io import wavfile
import sounddevice as sd
import pickle
import Plugin as plugin
import Fade as fade



#
# Constants
#
ECHO_DELAY = 10000

#
# Private functions
#

# Plot PlotFreqResponse
def PlotFreqResponse(x, title):
    #
    # Plot fequency response
    #
    # fig = plt.figure()
    [freq, response] = signal.freqz(x)
    fig, pltArr = plt.subplots(2, sharex=True) 
    fig.suptitle(title)
    #Magnitude
    pltArr[0].plot((freq/math.pi), 20 * np.log10(np.abs(response)+1e-6))
    pltArr[0].set_title("Magnitude of Frequency Response")
    pltArr[0].set_xlabel('Normalized Frequency (xPi [rad/sample])')
    pltArr[0].set_ylabel("Magnitude [dB]")
    # Phase
    angles = np.unwrap(np.angle(response))
    pltArr[1].plot((freq/math.pi), angles)
    pltArr[1].set_title("Phase of Frequency Response")
    pltArr[1].set_xlabel('Normalized Frequency (xPi [rad/sample])')
    pltArr[1].set_ylabel("Angle [xPi [rad/sample]]")

# PyAudioPlay
def PyAudioPlay(input_array, rate, channels):
    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()

    # open stream (2), 2 is size in bytes of int16
    stream = p.open(format=p.get_format_from_width(2),
                    channels=channels,
                    rate=rate,
                    output=True)

    # play stream (3), blocking call
    stream.write(input_array)

    # stop stream (4)
    stream.stop_stream()
    stream.close()

    # close PyAudio (5)
    p.terminate()


#
#
# Main
#
#
if __name__ == "__main__":

    # 
    # Signals
    #
    
    # Sax
    SaxPath = "Effects\sax.wav"
    fs_Sax, SaxSignalFullRange = wavfile.read(SaxPath)
    SaxSignal_length = SaxSignalFullRange.shape[0] / fs_Sax
    print("Input info: ")
    print(f"    Sax fs: {fs_Sax}")
    print(f"    Samples: {SaxSignalFullRange.shape[0]}")
    print(f"    Length = {SaxSignal_length}[s]") 
    print()
    # Play
    # sd.play(SaxSignalFullRange, fs_Sax)
    # sd.wait()


    # Normalize
    x_max = np.abs(SaxSignalFullRange).max()  
    SaxSignalNorm = SaxSignalFullRange / x_max

    # Stereo
    SaxSignalNormStereo = np.array([SaxSignalNorm, SaxSignalNorm])
    SaxSignalNormStereo = np.transpose(SaxSignalNormStereo)

    #
    # Linear fade
    #

    # Fade In
    SaxSignalNormFadedIn = fade.LinearFadeIn(SaxSignalNorm, 2000, fs_Sax)

    # Fade Out
    SaxSignalNormFadedOut = fade.LinearFadeOut(SaxSignalNorm, 2000, fs_Sax)

    # Fade InOut
    SaxSignalNormFadedInOut_aux = fade.LinearFadeIn(SaxSignalNorm, 2000, fs_Sax)
    SaxSignalNormFadedInOut = fade.LinearFadeOut(SaxSignalNormFadedInOut_aux, 2000, fs_Sax)
    
    #
    # Quadratic fade
    #

    # Fade In
    SaxSignalNormFadedIn_quad = fade.QuadraticFadeIn(SaxSignalNorm, 2000, fs_Sax)

    # Fade Out
    SaxSignalNormFadedOut_quad = fade.QuadraticFadeOut(SaxSignalNorm, 2000, fs_Sax)

    # Fade InOut
    SaxSignalNormFadedInOut_quad_aux = fade.QuadraticFadeIn(SaxSignalNorm, 2000, fs_Sax)
    SaxSignalNormFadedInOut_quad = fade.QuadraticFadeOut(SaxSignalNormFadedInOut_quad_aux, 2000, fs_Sax)
    

    #
    # Concave fade
    #

    # Fade In
    SaxSignalNormFadedIn_conc = fade.ConcaveFadeIn(SaxSignalNorm, 2000, fs_Sax)

    # Fade Out
    SaxSignalNormFadedOut_conc = fade.LinearFadeOut(SaxSignalNorm, 2000, fs_Sax)

    # Fade InOut
    SaxSignalNormFadedInOut_conc_aux = fade.LinearFadeIn(SaxSignalNorm, 2000, fs_Sax)
    SaxSignalNormFadedInOut_conc = fade.LinearFadeOut(SaxSignalNormFadedInOut_conc_aux, 2000, fs_Sax)
    
    #
    # Play
    #
    print()
    print("Playing...")
    sd.play(SaxSignalNormFadedInOut_conc, fs_Sax)
    sd.wait()

    # Plot
    fig, pltArr = plt.subplots(5, sharex=True) 
    fig.suptitle("Linear Fade")
    pltArr[0].plot(SaxSignalFullRange)
    pltArr[1].plot(SaxSignalNorm)
    pltArr[2].plot(SaxSignalNormFadedIn)
    pltArr[3].plot(SaxSignalNormFadedOut)
    pltArr[4].plot(SaxSignalNormFadedInOut)

    # Plot
    fig, pltArr = plt.subplots(5, sharex=True) 
    fig.suptitle("Quadratic Fade")
    pltArr[0].plot(SaxSignalFullRange)
    pltArr[1].plot(SaxSignalNorm)
    pltArr[2].plot(SaxSignalNormFadedIn_quad)
    pltArr[3].plot(SaxSignalNormFadedOut_quad)
    pltArr[4].plot(SaxSignalNormFadedInOut_quad)

    # Plot
    fig, pltArr = plt.subplots(5, sharex=True) 
    fig.suptitle("Concave Fade")
    pltArr[0].plot(SaxSignalFullRange)
    pltArr[1].plot(SaxSignalNorm)
    pltArr[2].plot(SaxSignalNormFadedIn_conc)
    pltArr[3].plot(SaxSignalNormFadedOut_conc)
    pltArr[4].plot(SaxSignalNormFadedInOut_conc)
    


    #
    plt.show()

    #Exit
    print("Exiting program...")
