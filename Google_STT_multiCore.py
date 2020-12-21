import asyncio

import time
import multiprocessing
num_cores = multiprocessing.cpu_count() # 12
num_cores = 1


import speech_recognition as sr
import soundfile as sf
import librosa
import numpy as np
import os
import argparse
from tqdm import tqdm
from hanspell import spell_checker
from chatspace import ChatSpace
from statistical_VAD import stVAD, signalSegmentation
import parmap
import shutil
#%%
def all2one_txt_file(result_STT_textfile):  
    #%% Data Read and preprocess
    tag_label = 0
    result_STT_textfile_list = result_STT_textfile.replace(".txt","_list.txt")

    if os.path.exists(result_STT_textfile_list):
        os.remove(result_STT_textfile_list)
    wflist = open(result_STT_textfile_list, encoding="utf-8", mode="a")

    if os.path.exists(result_STT_textfile):
        os.remove(result_STT_textfile)
    wf = open(result_STT_textfile, encoding="utf-8", mode="a")
    print("all2one : ", result_STT_textfile)
    DBpath = os.path.dirname(os.path.realpath(__file__))+'/tmp_ASR'
    txtfiles = os.listdir(DBpath)   
    txtfiles.sort()

    for filename in tqdm(txtfiles):
        if os.path.splitext(filename)[1].lower() == '.txt':
            read_path = "{}\{}".format(DBpath, filename).replace('\\', '/')
            
            rf = open(read_path, encoding="utf-8", mode="r")
            text = rf.readline()
            text = text.strip()
            filename=filename.split('.txt')[0]
            #text = filename + ':' + text + ' ' + str(tag_label)
            text = text + ' ' + str(tag_label)
            print(filename, file=wflist)
            print(text, file=wf)
#%%
def gSTT_file(file_path):
    r = sr.Recognizer()
    sapcer = ChatSpace()
    sig, fs = librosa.load(file_path,  mono=False, sr=16000, duration=None, dtype=np.float) 
    #sig = np.mean(sig, 0)
    sig = sig[0, 9000:]          
    filename=file_path.split('/')[-1]
    sf.write(filename+'tmp_noise_file.wav', sig[:int(fs*0.7)], samplerate=16000, subtype='PCM_16')
    noise_sig = sr.AudioFile(filename+'tmp_noise_file.wav')
    with noise_sig as source:
        r.adjust_for_ambient_noise(source, duration=0.2)            
                
    os.remove(filename+'tmp_noise_file.wav')
    print(file_path)
    #(vad, VoiceInterval) = stVAD(sig, fs, nFFT=512, win_length=0.032, hop_length=0.008)
    VoiceInterval = signalSegmentation(sig, fs)

    Out_file_name_txt = "{}/{}.txt".format(os.path.dirname(os.path.realpath(__file__))+'/tmp_ASR', filename)
    if os.path.exists(Out_file_name_txt):
        os.remove(Out_file_name_txt)

    f = open(Out_file_name_txt, encoding="utf-8", mode="a")
    All_texts = []
    for seg_i in range(len(VoiceInterval)):
        strt_i = int(VoiceInterval[seg_i][0])
        end_i  = int(VoiceInterval[seg_i][1])
        
        seg_sig = sig[strt_i:end_i,]

        sf.write(filename+'tmp_file.wav', seg_sig, samplerate=16000, subtype='PCM_16')

        seg_sig = sr.AudioFile(filename+'tmp_file.wav')

        with seg_sig as source:
            audio = r.record(source)
        os.remove(filename+'tmp_file.wav')
        try:
            ASR_result = r.recognize_google(audio, language="ko-KR", show_all=True )
        except:
            ASR_result = None
        
        if not ASR_result:
            ASR_result = ' '
        else:
            ASR_result_text = ASR_result['alternative'][0]['transcript']
            
            ASR_result_text = spell_checker.check(ASR_result_text)
            ASR_result_text = ASR_result_text.as_dict() ['checked']
                                                
            ASR_result_text=(' '.join(str(x) for x in ASR_result_text))    
            ASR_result_text = ASR_result_text.replace("          ", "")  
            ASR_result_text = ASR_result_text.replace("         ", "")     
            ASR_result_text = ASR_result_text.replace("        ", "") 
            ASR_result_text = ASR_result_text.replace("       ", "")  
            ASR_result_text = ASR_result_text.replace("      ", "")   
            ASR_result_text = ASR_result_text.replace("     ", "")  
            ASR_result_text = ASR_result_text.replace("    ", "")  
            ASR_result_text = ASR_result_text.replace("   ", "") 
            ASR_result_text = ASR_result_text.replace("  ", "")       
            ASR_result_text = ASR_result_text.replace(" ", "")
                       
                  
               
                
               
            
            
            
            ASR_result_text = ASR_result_text.replace("          ", "")     

            ASR_result_text = sapcer.space(ASR_result_text, batch_size=1) 
            All_texts.append(ASR_result_text)

    One_text = (' '.join(str(x) for x in All_texts))+ ' ' + str(0)
    print(One_text, file=f)  

async def gSTT_file_async(file_path):
    r = sr.Recognizer()
    sapcer = ChatSpace()
    sig, fs = librosa.load(file_path,  mono=False, sr=16000, duration=None, dtype=np.float) 
    #sig = np.mean(sig, 0)
    sig = sig[0, 9000:]          
    filename=file_path.split('/')[-1]
    sf.write(filename+'tmp_noise_file.wav', sig[:int(fs*0.7)], samplerate=16000, subtype='PCM_16')
    noise_sig = sr.AudioFile(filename+'tmp_noise_file.wav')
    with noise_sig as source:
        r.adjust_for_ambient_noise(source, duration=0.2)            
                
    os.remove(filename+'tmp_noise_file.wav')
    
    (vad, VoiceInterval) = stVAD(sig, fs, nFFT=512, win_length=0.032, hop_length=0.008)

    Out_file_name_txt = "{}/{}.txt".format(os.path.dirname(os.path.realpath(__file__))+'/tmp_ASR', filename)
    if os.path.exists(Out_file_name_txt):
        os.remove(Out_file_name_txt)

    f = open(Out_file_name_txt, encoding="utf-8", mode="a")
    All_texts = []
    for seg_i in range(len(VoiceInterval)):
        strt_i = int(VoiceInterval[seg_i][0])
        end_i  = int(VoiceInterval[seg_i][1])
        
        seg_sig = sig[strt_i:end_i,]

        sf.write(filename+'tmp_file.wav', seg_sig, samplerate=16000, subtype='PCM_16')

        seg_sig = sr.AudioFile(filename+'tmp_file.wav')

        with seg_sig as source:
            audio = r.record(source)
        os.remove(filename+'tmp_file.wav')
        
        #비동기 요청 start
        loop = asyncio.get_event_loop()        
        # run_in_executor으로 코루틴 변환
        # ASR_result = r.recognize_google(audio, language="ko-KR", show_all=True )
        def do_req():
            return r.recognize_google(audio, language="ko-KR", show_all=True )
        ASR_result = await loop.run_in_executor(None, do_req)
        #비동기 요청 end
        
        if not ASR_result:
            ASR_result = ' '
        else:
            ASR_result_text = ASR_result['alternative'][0]['transcript']
            
            ASR_result_text = spell_checker.check(ASR_result_text)
            ASR_result_text = ASR_result_text.as_dict() ['checked']
                                                
            ASR_result_text=(' '.join(str(x) for x in ASR_result_text))    
            ASR_result_text = ASR_result_text.replace(" ", "")
            ASR_result_text = ASR_result_text.replace("  ", "")            
            ASR_result_text = ASR_result_text.replace("   ", "")        
            ASR_result_text = ASR_result_text.replace("    ", "")     
            ASR_result_text = ASR_result_text.replace("     ", "")     
            ASR_result_text = ASR_result_text.replace("      ", "")     
            ASR_result_text = ASR_result_text.replace("       ", "")     
            ASR_result_text = ASR_result_text.replace("        ", "")     
            ASR_result_text = ASR_result_text.replace("         ", "")     
            ASR_result_text = ASR_result_text.replace("          ", "")     

            ASR_result_text = sapcer.space(ASR_result_text, batch_size=1) 
            All_texts.append(ASR_result_text)

    One_text = (' '.join(str(x) for x in All_texts))+ ' ' + str(0)
    print(One_text, file=f)  
    
async def gSTT_async(full_path_list):       
    ##file list에 있는 file을 대상으로 비동기 그룹 생성    
    futures = [asyncio.ensure_future(gSTT_file_async(file_path)) for file_path in full_path_list]
    await asyncio.gather(*futures)
    all2one_txt_file(os.path.dirname(os.path.realpath(__file__))+'/text/asr_result.txt')
#%%

def gSTT(full_path_list):       
    for file_path in tqdm(full_path_list):
        print(file_path)
        gSTT_file(file_path)
    all2one_txt_file(os.path.dirname(os.path.realpath(__file__))+'/text/asr_result.txt')

def GoogleSTT(wavFolder):       
    if os.path.exists(os.path.dirname(os.path.realpath(__file__))+'/tmp_ASR'):
        shutil.rmtree(os.path.dirname(os.path.realpath(__file__))+'/tmp_ASR')
    os.makedirs(os.path.dirname(os.path.realpath(__file__))+'/tmp_ASR')
    #print(wavFolder)
    files = os.listdir(wavFolder)   
        
    full_path_list = []
    for filename in files:
        if os.path.splitext(filename)[1].lower() == '.wav':
            #fullPath = "{}\{}".format(wavFolder, filename).replace('\\', '/')
            full_path_list.append(wavFolder+"/"+filename)
            #print(wavFolder+filename)
    start = time.time()

    #cpu기반의 데이터 할당(by KSM)
    #splited_data = np.array_split(full_path_list, num_cores)
    #splited_data = [x.tolist() for x in splited_data]    
    #parmap.map(gSTT, splited_data, pm_pbar=True, pm_processes=num_cores)

    #비동기 코드 start    
    #loop = asyncio.get_event_loop()
    #loop.run_until_complete(gSTT_async(full_path_list))        
    #비동기 코드 end

    #일반코드
    gSTT(full_path_list)

    end = time.time()
    print(f'time taken: {end-start}')    
    
    return 1
#%%    
def main(args):    
    GoogleSTT(args.wavFolder)
    
    if os.path.exists(args.result_STT_Folder):
        shutil.rmtree(args.result_STT_Folder)
        os.makedirs(args.result_STT_Folder)

    result_STT_textfile = os.path.join(args.result_STT_Folder, args.result_STT_textfile_name )
    all2one_txt_file(result_STT_textfile)          

#%%			
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--result_STT_Folder', default = './Test_DB_txt',     type=str,  dest='result_STT_Folder')
    parser.add_argument('--wavFolder', default = './Test_DB', type=str,  dest='wavFolder')
    parser.add_argument('--result_STT_textfile_name', default = 'STT_result.txt', type=str,  dest='result_STT_textfile_name')
    
    args = parser.parse_args()
    print(args)

    start_time = time.time()
    main(args)
    print('elapsed time:', time.time() - start_time)

