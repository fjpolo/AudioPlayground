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
import Nonlinearities as nonlin



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
    # Softclipper
    #
    SaxSignalNorm_x2 = 5 * SaxSignalNorm
    SaxSignalNormSoftClip = nonlin.SoftClipper(SaxSignalNorm)
    # Impulse
    x = np.zeros(2 * fs_Sax)
    x[0] = 1
    # Impulse Response Softclipper
    # x_IR = nonlin.SoftClipper(x)
    # SavePath = "Effects\Plugins\Softclipper_IR.wav"
    # print("Saving Softclipper_IR.wav")
    # plugin.SavePluginAsWav(x_IR, SavePath)
    # print("Save successful!!")
    # plt.figure()
    # plt.plot(x_IR)
    # PlotFreqResponse(x_IR, "Soft Clipper Softmax Frequency Response")
    # Sine
    t = np.linspace(0, (len(SaxSignalNorm)/fs_Sax), len(SaxSignalNorm))
    x_sin = 5 * np.sin(2 * np.pi * 100 * t)
    # THD@100Hz Softclipper
    x_THD100Hz = nonlin.SoftClipper(x_sin)
    # plt.figure()
    # plt.plot(x_IR)
    PlotFreqResponse(x_THD100Hz, "Soft Clipper Softmax THD@100Hz")


    #
    # DoubleSoftclipper
    #
    SaxSignalNormSoftClipDouble = nonlin.DoubleSoftClipper(SaxSignalNorm)
    # Impulse
    x = np.zeros(2 * fs_Sax)
    x[0] = 1
    # Impulse Response Softclipper
    # x_IR = nonlin.DoubleSoftclipper(x)
    # SavePath = "Effects\Plugins\DoubleSoftclipper_IR.wav"
    # print("Saving DoubleSoftclipper.wav")
    # plugin.SavePluginAsWav(x_IR, SavePath)
    # print("Save successful!!")
    # plt.figure()
    # plt.plot(x_IR)
    # PlotFreqResponse(x_IR, "Soft Clipper Softmax Frequency Response")
    # Sine
    t = np.linspace(0, (len(SaxSignalNorm)/fs_Sax), len(SaxSignalNorm))
    x_sin = 5 * np.sin(2 * np.pi * 100 * t)
    # THD@100Hz Softclipper
    x_THD100Hz = nonlin.DoubleSoftClipper(x_sin)
    # plt.figure()
    # plt.plot(x_IR)
    PlotFreqResponse(x_THD100Hz, "Double Soft Clipper Softmax THD@100Hz")

    #
    # Play
    #
    print()
    print("Playing...")
    sd.play(SaxSignalNormSoftClipDouble, fs_Sax)
    sd.wait()

    # Plot
    fig, pltArr = plt.subplots(4, sharex=True) 
    fig.suptitle("Linear Fade")
    pltArr[0].plot(SaxSignalFullRange)
    pltArr[1].plot(SaxSignalNorm)
    pltArr[2].plot(SaxSignalNormSoftClip)
    pltArr[3].plot(SaxSignalNormSoftClipDouble)



    #
    plt.show()

    #Exit
    print("Exiting program...")
