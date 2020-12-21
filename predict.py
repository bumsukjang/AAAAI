import sys
import time
import os

#Multi Thread
from Google_STT_multiThread import GoogleSTT

#Single Thread
#from Google_STT_multiCore import GoogleSTT

from koelectraModel import pred_sm
from robetaModel import pred_cj
from to_json import write_json

#%%    
if __name__ == '__main__':
    sys.path.append('/aichallenge')
    os.chdir("/aichallenge")
    
    dataset_foler = os.path.dirname(os.path.realpath(__file__))+'/../error_dataset'
    output_file_path = os.path.dirname(os.path.realpath(__file__))+'/t2_res_U0000000272.json'


    if(len(sys.argv) > 1):
        dataset_foler = sys.argv[1]

    cli_args = lambda: None
    setattr(cli_args, "out_infer_file", os.path.dirname(os.path.realpath(__file__))+"/infer_result/infer_results.txt")
    setattr(cli_args, "in_wav_folder", dataset_foler)
    setattr(cli_args, "batch_size", 2)

    start_time = time.time()

    #wav to text
    #folder name : tmp_ASR
    GoogleSTT(cli_args.in_wav_folder)

    #KoElectra Model
    result_sm = pred_sm(cli_args)
    print(result_sm)
    sm_val = result_sm[1]

    #print("result(sm) : ", result_sm)

    #Roberta Model
    #result_cj = pred_cj(cli_args)  
    #cj_val = result_cj[1]
    #print("result(cj) : ", result_cj)

    
    DBpath = os.path.dirname(os.path.realpath(__file__))+'/tmp_ASR'
    print(DBpath)
    txtfiles = os.listdir(DBpath)
    txtfiles.sort()
    print(txtfiles)
    #print("filelist : ",txtfiles)
    
    merge_val = []
    for idx in range(0, len(txtfiles)):     
        #both   
        #val = "000001" if cj_val[idx]==0 else "000001" if sm_val[idx]=="0" else "020819" if sm_val[idx]=="1" else "020811" if sm_val[idx]=="2" else "02051" if sm_val[idx]=="3" else "020121"
        
        #Koelectra        
        val = "000001" if sm_val[idx]=="0" else "020819" if sm_val[idx]=="1" else "020811" if sm_val[idx]=="2" else "02051" if sm_val[idx]=="3" else "020121"
        
        #roberta
        #val = "000001" if cj_val[idx]==0 else "020819" if cj_val[idx]==1 else "020811" if cj_val[idx]==2 else "02051" if cj_val[idx]==3 else "020121"
        
        #dummy
        #val = "000001" 
        merge_val.append(val)
    write_json(output_file_path, txtfiles, merge_val, len(txtfiles))


    #main(cli_args)
    print('elapsed time:', time.time() - start_time)

# %%
