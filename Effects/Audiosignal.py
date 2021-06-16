import numpy as np
import pickle
from scipy import signal


#
# Channels
#
def MonoToStereo(Input):
    """
    Converts a mono input to stereo.-
    """
    output = np.array([Input, Input])
    output = np.transpose(Input)
    return output




#
# Plugins
#

# SavePluginAsWav
def SavePluginAsWav(signal, path):
    """
    Saves signal as .wav file.-
    """
    # Save
    with open(path,'wb') as f: pickle.dump(signal, f)

# LoadPluginFromWav
def LoadPluginFromWav(path):
    """
    Loads a wav file and returns a signal.-
    """
    # Load from pickle
    with open(path,'rb') as f: array = pickle.load(f)
    return array


#
# Amplitude stuff
#

# Normalize
def Normalize(input, R_dB):
    """
    Normalize a signal RMS scale.-
    """
    #Convert level to inear scale
    R = 10**(R_dB/20.0)
    # Determine scale factior
    a = np.sqrt((len(input)*R**2) / np.sum(input**2))
    # Scale amplitude of output signal
    output = input * a
    # Square each individual element
    inputSquare = np.square(output)
    # Take the mean
    meanSquare = np.sum(inputSquare / len(input))
    # Take the square of the root mean square
    rootMeanSquare = np.sqrt(meanSquare)
    # Convert results to dB scale
    rootMeanSquare_dB = 20*np.log10(rootMeanSquare/1.0)

    return output, rootMeanSquare_dB

# Sum
def Sum(input1, input2):
    """
    Returns the sum of two signals.-
    """
    if len(input1) == len(input2):
        return (input1+input2)
    else:
        return []

# Substract
def Substract(input1, input2):
    """
    Returns the substraction of two signals.-
    """
    if len(input1) != len(input2):
        return []
    return (input1-input2)

# NullTest
def NullTest(input1, input2):
    """
    Null Test between two signals: Returns False if didn't pass, True if it passes.-
    """
    if len(input1) != len(input2):
        return False
    for i in range(len(input1)):
        if (abs(abs(input1)-abs(input2))) != 0:
            return False

# Multiply
def Multiply(input1, input2):
    """
    Return the multiplication of two signals.-
    """
    if len(input1) != len(input2):
        return []
    return (np.multiply(input1, input2))



#
# Echoes
# 

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

#
# Modulation
#

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



#
# Complex nonlinearities
#

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