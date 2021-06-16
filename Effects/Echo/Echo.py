import numpy as np


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
