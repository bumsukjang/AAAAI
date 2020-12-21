import numpy as np
import librosa

#%%
def stVAD(signal, sr):

    VoiceInterval = np.zeros([4,2])
    VoiceInterval[0,:]  = [1000,  int(sr*10)]
    VoiceInterval[1,:]  = [int(sr*10)-1000,  int(sr*20)]
    VoiceInterval[2,:]   = [int(sr*20)-1000,  int(sr*30)]
    VoiceInterval[3,:]  = [int(sr*30)-1000,  len(signal)-1000]

    return  VoiceInterval
#%%
if __name__ == '__main__':
    # path_wav = os.path.join(os.getcwd() , "Test_DB_pilot/t2_0003.wav" )
    path_wav = 'Test_DB_wav/t2_0003 - 복사본.wav'
    sig, sr = librosa.load(path_wav,  mono=False, sr=16000, duration=None, dtype=np.float) 
    signal = np.mean(sig, 0)
        
    VoiceInterval = stVAD(signal, sr)


