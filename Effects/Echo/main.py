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

# Echo
def Echo(signal, nrOfEchoes, delay_mS, SampleRate, Gain=1):
    """
    Echo.-
    """
    delay_samples = (SampleRate * delay_mS) / 1000
    delay = int(nrOfEchoes * delay_samples)
    Echo = np.zeros(len(signal) + delay)
    outputSignal = np.zeros(len(signal) + delay)
    # outputSignal[:-int(nrOfEchoes * delay_samples)] = signal
    plt.figure()
    plt.plot(signal)
    for it in range(1, nrOfEchoes+1):
        temp_delay = int(it * delay_samples)
        temp_echo = np.zeros(len(signal) + delay)
        temp_echo[temp_delay:temp_delay+len(signal)] = signal
        Echo += temp_echo
        plt.plot(temp_echo)
    outputSignal[:len(signal)] = signal
    outputSignal += Echo / nrOfEchoes
    outputSignal *= Gain
    return outputSignal

# FeedForwardEcho
def FeedForwardEcho(Input, delay_mS, SampleRate, delayGain=0.2, arrayIndexing=True):
    """
    Feedworward Echo.-
    """
    delaySamples = int((SampleRate * delay_mS) / 1000)


    #
    if not arrayIndexing:
        dryPath = np.zeros(len(Input) + delaySamples)
        dryPath[:-delaySamples] = Input
        wetPath = np.zeros(len(Input) + delaySamples)
        wetPath[delaySamples:] = Input
        #
        output = np.zeros(len(Input) + delaySamples)
        output = dryPath + delayGain*wetPath

    # Array Indexing
    output = np.zeros(len(Input))
    for i in range(len(Input)):
        if(i<delaySamples):
            output[i] = Input[i]
        else:
            output[i] = Input[i] + delayGain * Input[i-delaySamples]

    return output

# TempoSyncFeedForwardEcho
def TempoSyncFeedForwardEcho(Input, bpm, SampleRate, delayGain=0.2):
    """
    Tempo syncronized Feedforward Echo.-
    """
    # Delay length
    bps = bpm / 60.0
    spb = 1 / float(bps)

    # Note duration
    noteDuration = 0.5

    # samples
    samplesOfDelay = int((noteDuration * spb) * SampleRate)


    #
    output = np.zeros(len(Input))
    for i in range(len(Input)):
        if(i<samplesOfDelay):
            output[i] = Input[i]
        else:
            output[i] = Input[i] + delayGain * Input[i-samplesOfDelay] 
    return output

# TempoSyncFeedbackEcho
def TempoSyncFeedbackEcho(Input, bpm, SampleRate, delayGain=0.2):
    """
    Tempo syncronized Feedback Echo.-
    """
    # Delay length
    bps = bpm / 60.0
    spb = 1 / float(bps)

    # Note duration
    noteDuration = 0.5

    # samples
    samplesOfDelay = int((noteDuration * spb) * SampleRate)

    #
    auxInput = np.zeros(3*len(Input))
    auxInput[:len(Input)] = Input

    #
    output = np.zeros(len(Input))
    for i in range(len(Input)):
        if(i<samplesOfDelay):
            output[i] = Input[i]
        else:
            output[i] = Input[i] + delayGain * output[i-samplesOfDelay] 
    return output

# TempoSyncFeedbackEcho
def MultitapTempoSyncFeedbackEcho(Input, bpm, noteDurations, SampleRate, delayGains, taps=1):
    """
    Multitap Tempo syncronized Feedback Echo.-
    """
    # Delay length
    bps = bpm / 60.0
    spb = 1 / float(bps)
    # samples
    samplesOfDelay = []
    buffers = []
    for i in range(taps):
        temp = int((noteDurations[i] * spb) * SampleRate)
        samplesOfDelay.append(temp)
        tempBuff = np.zeros(temp)
        buffers.append(tempBuff)

    # 
    output = np.zeros(len(Input))
    for i in range(len(Input)):
        output[i] = delayGains[0] * Input[i] 
        for j in range(1, taps+1):
            tempArr = buffers[j-1]
            output[i] += delayGains[j] * tempArr[-1]
        for buf in buffers:
            buf[1:] = buf[:-1]
            buf[0] = Input[i]
    return output

# StereoTempoSyncFeedbackEcho
def StereoTempoSyncFeedbackEcho(Input, bpm, SampleRate, delayGains):
    """
    Stereo Tempo syncronized Feedback Echo.-
    """
    # Delay length
    bps = bpm / 60.0
    spb = 1 / float(bps)

    # Note duration
    noteDuration = 0.5

    # samples
    samplesOfDelay1 = int((noteDuration * spb) * SampleRate)
    samplesOfDelay2 = int((2 * noteDuration * spb) * SampleRate)

    #
    auxInput = np.zeros(int(1.5 * len(Input)))
    auxInput[:len(Input)] = Input

    #
    output = np.array([np.zeros(len(Input)), np.zeros(len(Input))])
    output = np.transpose(output)

    #
    for i in range(len(Input)):
        if (i-samplesOfDelay1) < 1:
            output[:, 0][i] = delayGains[0] * Input[i]
            output[:, 1][i] = delayGains[1] * Input[i]

        elif (i-samplesOfDelay2) < 1:
            output[:, 0][i] = delayGains[0] * Input[i] + delayGains[0] * Input[i-samplesOfDelay2]
            output[:, 1][i] = delayGains[1] * Input[i]
        else:
            output[:, 0][i] = Input[i] + delayGains[0] * Input[i-samplesOfDelay1] 
            output[:, 1][i] = Input[i] + delayGains[1] * Input[i-samplesOfDelay2] 
    return output

# # ImpulseResponse
# def ImpulseResponse(SampleRate):
#     """
#     Returns the signal's impulse response.-
#     """
#     x = np.zeros(2 * SampleRate)
#     x[0] = 1
#     plt.figure()
#     plt.plot(x)

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
    print(f"    Sax fs: {fs_Sax}")
    print(f"    Samples: {SaxSignalFullRange.shape[0]}")
    print(f"    Length = {SaxSignal_length}[s]") 
    # Play
    # sd.play(SaxSignalFullRange, fs_Sax)
    # sd.wait()


    # Normalize
    x_max = np.abs(SaxSignalFullRange).max()  
    SaxSignalNorm = SaxSignalFullRange / x_max

    # Stereo
    SaxSignalNormStereo = np.array([SaxSignalNorm, SaxSignalNorm])
    SaxSignalNormStereo = np.transpose(SaxSignalNormStereo)

    # Echo
    # # # # # # # # Echo = np.zeros(len(SaxSignalNorm) + ECHO_DELAY)
    # # # # # # # # Echo[ECHO_DELAY:] = SaxSignalNorm
    # # # # # # # # SaxSignalNormEcho = np.zeros(len(SaxSignalFullRange))
    # # # # # # # # SaxSignalNormEcho = (SaxSignalNorm + Echo[:-ECHO_DELAY] / 2)
    # # # # # # # SaxSignalEcho = Echo(SaxSignalNorm, 3, 300, fs_Sax, 2)
    # # # # # SaxSignalEcho = FeedForwardEcho(SaxSignalNorm, 300, fs_Sax, 0.6)
    # # # # SaxSignalEcho = TempoSyncFeedForwardEcho(SaxSignalNorm, bpm=60, SampleRate=fs_Sax, delayGain=0.2)
    # # # SaxSignalEcho = TempoSyncFeedbackEcho(SaxSignalNorm, bpm=60, SampleRate=fs_Sax, delayGain=0.2)
    # # SaxSignalEcho = MultitapTempoSyncFeedbackEcho(
    # #     Input=SaxSignalNorm, 
    # #     bpm=60,
    # #     noteDurations=[1, 0.5],
    # #     SampleRate=fs_Sax,
    # #     delayGains=[1, 0.7, 0.5],
    # #     taps=2
    # #     )
    # SaxSignalEcho = StereoTempoSyncFeedbackEcho(
    #     Input=SaxSignalNorm,
    #     bpm=60,
    #     SampleRate=fs_Sax,
    #     delayGains = [0.7, 0.5]
    # )

    x = np.zeros(2 * fs_Sax)
    x[0] = 1
    # plt.figure()
    # plt.plot(x)
    x_IR = MultitapTempoSyncFeedbackEcho(
        Input=x, 
        bpm=60,
        noteDurations=[1, 0.5],
        SampleRate=fs_Sax,
        delayGains=[1, 0.7, 0.5],
        taps=2
        )
    # plt.figure()
    # plt.plot(x_IR)

    # Save to wav
    SavePath = "Effects\Plugins\MultitapTempoSyncFeedbackEcho_IR.wav"
    with open(SavePath,'wb') as f: pickle.dump(x_IR, f)

    # Load plugin
    LoadPath = "Effects\Plugins\MultitapTempoSyncFeedbackEcho_IR.wav"
    with open(LoadPath,'rb') as f: x_ir = pickle.load(f)

    # Convolve
    SaxSignalEcho = np.convolve(SaxSignalNorm, x_ir)
    # plt.figure()
    # plt.plot(x_ir)
    
    # # Play
    sd.play(SaxSignalEcho, fs_Sax)
    sd.wait()

    # Plot
    fig, pltArr = plt.subplots(3, sharex=True) 
    pltArr[0].plot(SaxSignalFullRange)
    pltArr[1].plot(SaxSignalNorm)
    pltArr[2].plot(SaxSignalEcho)


    #
    plt.show()