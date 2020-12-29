import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
#%% 
def moving_average(a, n=3): ### "n=3" indicates the default value
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret/=n
    ### Masking initial a few values to prevent weird values 
    ret[:n-1]=ret[n-1]
    return ret
#%%
def stft(x, n_fft=512, win_length=400, hop_length=160): 
    window = np.hamming(win_length)
    return np.array([np.fft.rfft(window*x[i:i+win_length],n_fft,axis=0) for i in range(0, len(x)-win_length, hop_length)])
#%%

def signalSegmentation(signal, sr):

    VoiceInterval = np.zeros([4,2])
    VoiceInterval[0,:]  = [1000,  int(sr*10)]
    VoiceInterval[1,:]  = [int(sr*10)-1000,  int(sr*20)]
    VoiceInterval[2,:]   = [int(sr*20)-1000,  int(sr*30)]
    VoiceInterval[3,:]  = [int(sr*30)-1000,  len(signal)-1000]

    return  VoiceInterval

def stVAD(signal, sr, nFFT=512, win_length=0.032, hop_length=0.008):

    signal=signal.astype('float')

    win_length_sample = round(win_length*sr)
    hop_length_sample = round(hop_length*sr)    

    # the variance of the speech; lambda_x(k)
    _stft = stft(signal, n_fft=nFFT, win_length=win_length_sample, hop_length=hop_length_sample)
    pSpectrum = np.abs(_stft) ** 2  + 0.0000000001                   

    (nFrames,nFFT2) = pSpectrum.shape 
#%%               
    All_estHngVAD=np.zeros((nFrames,1))
    estHngVAD = 0
    # logGamma_frame=0 
    Epo_count = 0     
    EhangOver = 0      
    
    N_var = pSpectrum[0,:] 
    pre_N_var = pSpectrum[0,:] 
    S_var = 0.0001 * pSpectrum[0,:] 
    list_GRatio=[]
#%%              
    for Frm_i in range(nFrames):          
        current_frame_power = pSpectrum[Frm_i,:]   
        aPosterioriSNR_frame = current_frame_power / (N_var + 1e-9)
        aPosterioriSNR_frame[aPosterioriSNR_frame < 0.008**2] = 0.008**2
        
        smoothFactorDD = 0.95
        oper=aPosterioriSNR_frame-1
        oper[oper < 0] = 0 
        smoothed_a_priori_SNR = smoothFactorDD * (S_var/ (pre_N_var+1e-9)) + (1.0 - smoothFactorDD)*oper                
#%%
        term1 = 1/(1+smoothed_a_priori_SNR)
        tmp = aPosterioriSNR_frame *smoothed_a_priori_SNR/(smoothed_a_priori_SNR + 1)
        tmp[tmp>8]=8

        term2=np.exp( tmp )
        
        LRatio=term1*term2
        logLRatio=np.log(LRatio+1e-9)+1e-9
                
        logLRatio = np.sort(logLRatio)[::-1]
        probRatio = np.mean(logLRatio[:196])  
        
        if Frm_i ==0:
            pre_probRatio = probRatio

        probRatio = 0.8*pre_probRatio + 0.2*probRatio
        
        pre_probRatio = probRatio
                
        gain = smoothed_a_priori_SNR/(smoothed_a_priori_SNR + 1)   # wiener filter;
    
        gain[gain<0.1]=0.1
        S_var = (gain**1) * current_frame_power
                
        if Frm_i <= 20:
            list_GRatio.append(probRatio)
        
        if Frm_i <= 20:
            Gthreshold_0=np.mean(list_GRatio)       
            Gthreshold=(Gthreshold_0+1e-9)*1.5
#%%        
        if Frm_i <= 20:
            estHngVAD = 0
            estVAD = 0
        elif Frm_i > (nFrames-3):
            estHngVAD = 0
            estVAD = 0
        else:            
            if probRatio >= Gthreshold: 
                estVAD = 1
            else:
                estVAD = 0
            
            if estVAD == 1:
                estHngVAD = 1
                Epo_count = Epo_count + 1
            else:
                if Epo_count > 3:
                    EhangOver = 10
                    Epo_count = 0
                
                if EhangOver != 0:
                    EhangOver = EhangOver - 1
                    estHngVAD = 1
                else:
                    estHngVAD = 0

        estHngVAD = estVAD        
        pre_N_var = N_var
#%%        
        lambda_Nvar_DD = 0.9
        if estHngVAD == 0:
            N_var = lambda_Nvar_DD*N_var + (1-lambda_Nvar_DD)*current_frame_power

        All_estHngVAD[Frm_i] = estHngVAD
        
#%%    
    All_estHngVAD = moving_average(All_estHngVAD, n=10)
    
    All_estHngVAD[All_estHngVAD>=0.3] = 1    
    All_estHngVAD[All_estHngVAD<0.3] = 0   
       
    All_estHngVAD[:20]=0
    All_estHngVAD[-2:]=0
    
    pre_Vad = 0
    start_fi=[0]
    end_fi=[0]
    
    for vi in range(len(All_estHngVAD)):
        cVad = All_estHngVAD[vi]
        if (cVad - pre_Vad)==1:
            start_fi.append(vi)
        elif (cVad - pre_Vad)==-1:
            end_fi.append(vi) 
                    
        pre_Vad = cVad
        
    start_pt = np.array(start_fi)*int(hop_length*sr)    
    end_pt   = np.array(end_fi)*int(hop_length*sr)    
    
    A= np.abs(end_pt-sr*10)
    B= np.abs(end_pt-sr*20)
    C= np.abs(end_pt-sr*30)

    VoiceInterval = np.zeros([4,2])
    VoiceInterval[0,:]  = [start_pt[0],                                       end_pt[np.where(np.min(A) == A)][0,].tolist()]
    VoiceInterval[1,:]  = [end_pt[np.where(np.min(A) == A)][0,].tolist()+800, end_pt[np.where(np.min(B) == B)][0,].tolist()]
    VoiceInterval[2,:]   = [end_pt[np.where(np.min(B) == B)][0,].tolist()+800, end_pt[np.where(np.min(C) == C)][0,].tolist()]
    VoiceInterval[3,:]  = [end_pt[np.where(np.min(C) == C)][0,].tolist()+800, end_pt[-1]]

    return All_estHngVAD, VoiceInterval
#%%
if __name__ == '__main__':
    path_wav = os.path.join(os.getcwd() , "Test_DB_pilot/t2_0003.wav" )
    
    sig, sr = librosa.load(path_wav,  mono=False, sr=16000, duration=None, dtype=np.float) 
    signal = np.mean(sig, 0)
        
    signal = signal[9000:]
    (vad, VoiceInterval) = stVAD(signal, sr, nFFT=512, win_length=0.032, hop_length=0.008)

    plt.subplot(2, 1, 1)
    plt.plot(signal)
    plt.title('Time Signal')
    
    plt.subplot(2, 1, 2)
    plt.plot(vad)
    plt.xlabel('frame')
    plt.ylabel('Prob')

    plt.tight_layout()
    plt.show()
    

