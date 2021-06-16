import pickle

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