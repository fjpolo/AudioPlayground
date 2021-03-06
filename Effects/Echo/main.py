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
import Echo as echo
import Plugin as plugin

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

    # Impulse
    x = np.zeros(2 * fs_Sax)
    x[0] = 1

    # FeedForwardEcho
    x_IR = echo.FeedForwardEcho(x, 300, fs_Sax, 0.6)
    SavePath = "Effects\Plugins\FeedForwardEcho_IR.wav"
    print("Saving FeedForwardEcho_IR.wav")
    plugin.SavePluginAsWav(x_IR, SavePath)
    print("Save successful!!")

    # TempoSyncFeedForwardEcho
    x_IR = echo.TempoSyncFeedForwardEcho(x, bpm=60, SampleRate=fs_Sax, delayGain=0.2)
    SavePath = "Effects\Plugins\TempoSyncFeedForwardEcho_IR.wav"
    print("Saving TempoSyncFeedForwardEcho_IR.wav")
    plugin.SavePluginAsWav(x_IR, SavePath)
    print("Save successful!!")

    # TempoSyncFeedbackEcho
    x_IR = echo.TempoSyncFeedbackEcho(x, bpm=60, SampleRate=fs_Sax, delayGain=0.2)
    SavePath = "Effects\Plugins\TempoSyncFeedbackEcho_IR.wav"
    print("Saving TempoSyncFeedbackEcho_IR.wav")
    plugin.SavePluginAsWav(x_IR, SavePath)
    print("Save successful!!")

    # MultitapTempoSyncFeedbackEcho
    x_IR = echo.MultitapTempoSyncFeedbackEcho(
        Input=x, 
        bpm=60,
        noteDurations=[1, 0.5],
        SampleRate=fs_Sax,
        delayGains=[1, 0.7, 0.5],
        taps=2
        )    
    SavePath = "Effects\Plugins\MultitapTempoSyncFeedbackEcho_IR.wav"
    print("Saving MultitapTempoSyncFeedbackEcho_IR.wav")
    plugin.SavePluginAsWav(x_IR, SavePath)
    print("Save successful!!")

    # StereoTempoSyncFeedbackEcho
    x_IR = echo.StereoTempoSyncFeedbackEcho(
        Input=x,
        bpm=60,
        SampleRate=fs_Sax,
        delayGains = [0.7, 0.5]
    )
    SavePath = "Effects\Plugins\StereoTempoSyncFeedbackEcho_IR.wav"
    print("Saving StereoTempoSyncFeedbackEcho_IR.wav")
    plugin.SavePluginAsWav(x_IR, SavePath)
    print("Save successful!!")




    # Load plugin
    LoadPath = "Effects\Plugins\FeedForwardEcho_IR.wav"
    print()
    print("Loading FeedForwardEcho_IR.wav...")
    x_ir = plugin.LoadPluginFromWav(LoadPath)
    
    # Convolve
    print()
    print("Convolving input and TempoSyncFeedbackEcho_IR...")
    SaxSignalEcho = np.convolve(SaxSignalNorm, x_ir)
    # plt.figure()
    # plt.plot(x_ir)
    
    # # Play
    print()
    print("Playing...")
    sd.play(SaxSignalEcho, fs_Sax)
    sd.wait()

    # Plot
    fig, pltArr = plt.subplots(3, sharex=True) 
    pltArr[0].plot(SaxSignalFullRange)
    pltArr[1].plot(SaxSignalNorm)
    pltArr[2].plot(SaxSignalEcho)


    #
    plt.show()

    #Exit
    print("Exiting program...")
