import sys
import time
import os

#Multi Thread
from Google_STT_multiThread import GoogleSTT

#Single Thread
#from Google_STT_multiCore import GoogleSTT

from koelectraModel import pred_sm, pred_sm_raw, pred_sm_h, pred_sm_with_raw
from robetaModel import pred_cj
from to_json import write_json

import numpy as np

import sklearn.metrics as metrics

from statistics import mode
from collections import Counter

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
    setattr(cli_args, "ko_model_name", "s2_checkpoint-7300")
    setattr(cli_args, "batch_size", 2)

    
    start_time = time.time()

    #wav to text
    #folder name : tmp_ASR
    GoogleSTT(cli_args.in_wav_folder)

    #DB Path Setting
    DBpath = os.path.dirname(os.path.realpath(__file__))+'/tmp_ASR'
    txtfiles = os.listdir(DBpath)
    txtfiles.sort()

    #KoElectra Model(ENSEMBLE - sum)    
    """ models = ["s2_checkpoint-7300","s4_checkpoint-4200","s3_checkpoint-2400","s5_checkpoint-5100"]
    results_sm = None
    sm_val = None
    for idx in range(0, len(models)):
        setattr(cli_args, "ko_model_name", models[idx])
        result_sm = pred_sm_raw(cli_args)
        if results_sm is None:
            results_sm = np.array(result_sm)
        else :
            results_sm += np.array(result_sm)
        if sm_val is None:
            sm_val = np.array(result_sm[1])
        else : 
            sm_val += np.array(result_sm[1])
    
    for idx in range(0, len(results_sm)):
        print(results_sm[idx])
    sm_val = np.argmax(np.array(sm_val), axis=1)
    print(sm_val) """ 

    #KoElectra Model(ENSEMBLE - Voting)    
    #models = ["s2_checkpoint-7300","s4_checkpoint-4200","s3_checkpoint-2400","s5_checkpoint-5100"]
    """ models = ["s2_checkpoint-7300","s4_checkpoint-4200","s3_checkpoint-2400", "s10_checkpoint-6300"]
    sm_val = None
    for idx in range(0, len(models)):
        setattr(cli_args, "ko_model_name", models[idx])
        result_sm = pred_sm(cli_args)
        print("result(sm) : ", result_sm)
        if sm_val is None:
            sm_val = result_sm[1]
        else : 
            sm_val += result_sm[1]
    
    #sm_val = np.argmax(np.array(sm_val), axis=1)
    print(sm_val)
    results = np.reshape(sm_val,(len(models),len(txtfiles)))
    print(results)
    results_trans = np.transpose(results)
    print(results_trans)
    ensenble_vote_result = list(map(lambda x:  Counter(x).most_common(2)[0][0], results_trans))
    #result_val = list(map(lambda x:  print(x, multimode(x)), results_trans)) #this is for python 3.8
    print(np.transpose(ensenble_vote_result))
    result_for_metrics = ensenble_vote_result """

    #KoElectra Model(ENSEMBLE - expert)        
    TotalClassNum = 5
    models = ["checkpoint-4500_class0","checkpoint-5100_class1","checkpoint-4700_class2","checkpoint-900_class3","checkpoint-3800_class4","s2_checkpoint-7300"]
    #models = ["s1_checkpoint-2500","s2_checkpoint-7300","s3_checkpoint-2400","s4_checkpoint-4200","s5_checkpoint-5100","s6_checkpoint-2900","s7_checkpoint-4500","s8_checkpoint-2100","s9_checkpoint-1300","s10_checkpoint-6300"]
    #models = ["s1_checkpoint-2500","s2_checkpoint-7300","s3_checkpoint-2400","s4_checkpoint-4200","s5_checkpoint-5100"]
    sm_val = None
    for idx in range(0, len(models)):
        setattr(cli_args, "ko_model_name", models[idx])
        result_sm = pred_sm_raw(cli_args)
        #print("result(sm) : ", result_sm)
        if sm_val is None:
            sm_val = result_sm[1]
        else : 
            sm_val += result_sm[1]
    
    #sm_val = np.argmax(np.array(sm_val), axis=1)
    print("sm_val")
    print(sm_val)
    #sm_val_classification = list(map(lambda x : print(x), sm_val))
    sm_val_classification = np.argmax(sm_val, axis=1)
    #sm_val_classification = list(map(lambda x : np.argmax(x, axis=1), sm_val))
    print("sm_val_classification")
    print(sm_val_classification)
    results = np.reshape(sm_val,(len(models),len(txtfiles), TotalClassNum))    
    print("results")
    print(results)

    results_classfication = np.reshape(sm_val_classification,(len(models),len(txtfiles)))    
    print("results_classfication")
    print(results_classfication)
    
    results_trans = np.transpose(results)
    print("results_trans")
    print(results_trans)

    results_classfication_trans = np.transpose(results_classfication)
    print("results_classfication_trans")
    print(results_classfication_trans)

    #ensenble_vote_result = list(map(lambda x:  Counter(x).most_common(2)[0][0], results_trans))
    #result_val = list(map(lambda x:  print(x, multimode(x)), results_trans)) #this is for python 3.8
    def vote(val):
        i = val[0]
        x = val[1]
        experts_pred = []  
        #다수결
        """ votes = []
        for modelNum in range(0, len(models)):
            vote = results_classfication[int(modelNum)][i]
            print("model " + str(modelNum) + " vote to " + str(vote))
            votes.append(vote)
        if len(Counter(votes).most_common(1)[0]) == 1:
            decision = str(Counter(votes).most_common(1)[0][0])
            print(str(i+1) + " sample is decided to " + str(decision))
            return decision
        else :
            decision = str(Counter(votes).most_common(1)[0][0])
            print(str(i+1) + " sample is decided to " + str(decision))
            return decision """
                
        #calculate experts
        for classNum in range(0, TotalClassNum):
            ## start code for expert ensuring
            th = float(results[int(classNum)][i][int(classNum)])
            if th > 0.01 and classNum == 1:
                x[classNum] = classNum
            ## end code for expert ensuring
            if x[classNum] == classNum:
                experts_pred += str(classNum)


        print((i+1), experts_pred)

        #experts more than one
        #decision by expert's predction value
        """ if len(experts_pred) > 1:
            th_prev = 0
            th_class = ""
            for expert_pred in experts_pred :
                th = float(results[int(expert_pred)][i][int(expert_pred)])
                if th > th_prev :
                    print(th, expert_pred)
                    th_prev = th
                    th_class = expert_pred
            return th_class
        elif len(experts_pred) == 1:
            return experts_pred[0] """
        
        
        
        #decision by other's vote with result
        """ if(len(experts_pred) > 1):
            votes = []
            for modelNum in range(0, len(models)):
                vote = results_classfication[int(modelNum)][i]
                print("model " + str(modelNum) + " vote to " + str(vote))
                votes.append(vote)
            decision = str(Counter(votes).most_common(1)[0][0])
            print(str(i+1) + " sample is decided to " + str(decision))
            return decision
        else :
            return experts_pred[0] """
        
        #decision with s2 model
        """ if len(experts_pred) == 1:
            return experts_pred[0]
        print(str(results_classfication[len(models)-1][i]))
        return str(results_classfication[len(models)-1][i]) """

        #decision(0-semi) with s2 model
        """ if len(experts_pred) == 1:
            return experts_pred[0]
        elif len(experts_pred) == 2:
            if "0" in experts_pred:
                return "0"
            if "2" in experts_pred and "3" in experts_pred:
                return "3"          
        print(str(results_classfication[len(models)-1][i]))
        return str(results_classfication[len(models)-1][i]) """

        #decision(0) and c2-3 with s2 model
        if "0" in experts_pred:
            return "0"
        if len(experts_pred) == 1:
            return experts_pred[0]
        elif len(experts_pred) == 2:
            if "2" in experts_pred and "3" in experts_pred:
                return "3"          
        print(str(results_classfication[len(models)-1][i]))
        return str(results_classfication[len(models)-1][i])

        #decision with priority
        """ if len(experts_pred) == 1:
            return experts_pred[0]
        else:
            decision = ""
            if "0" in experts_pred:
                decision = "0"
            elif "1" in experts_pred:
                decision = "1"
            elif "4" in experts_pred:
                decision = "4"
            elif "3" in experts_pred:
                decision = "3"
            else :
                decision = "2"
        print(decision)
        return decision """

    ensenble_vote_result = list(map(vote, enumerate(results_classfication_trans)))
    print(ensenble_vote_result)
    result_for_metrics = ensenble_vote_result

    #KoElectra Model
    #result_sm = pred_sm(cli_args)  
    #sm_val = result_sm[1]
    #print("result(sm) : ", result_sm)

    #KoElectra Model heuristic
    """ result_sm = pred_sm_h(cli_args)  
    sm_val = result_sm[1]
    print("result(sm) : ", result_sm) """

    #Roberta Model
    #result_cj = pred_cj(cli_args)  
    #cj_val = result_cj[1]
    #print("result(cj) : ", result_cj)

    
    
    
    merge_val = []
    for idx in range(0, len(txtfiles)):     
        #both   
        #val = "000001" if cj_val[idx]==0 else "000001" if sm_val[idx]=="0" else "020819" if sm_val[idx]=="1" else "020811" if sm_val[idx]=="2" else "02051" if sm_val[idx]=="3" else "020121"
        
        #Koelectra        
        #val = "000001" if sm_val[idx]=="0" else "020819" if sm_val[idx]=="1" else "020811" if sm_val[idx]=="2" else "02051" if sm_val[idx]=="3" else "020121"
        
        #Koelectra(Ensemble)
        val = "000001" if result_for_metrics[idx]=="0" else "020819" if result_for_metrics[idx]=="1" else "020811" if result_for_metrics[idx]=="2" else "02051" if result_for_metrics[idx]=="3" else "020121"

        #roberta
        #val = "000001" if cj_val[idx]==0 else "020819" if cj_val[idx]==1 else "020811" if cj_val[idx]==2 else "02051" if cj_val[idx]==3 else "020121"
        
        #dummy
        #val = "000001" 
        merge_val.append(val)
    write_json(output_file_path, txtfiles, merge_val, len(txtfiles))
    
    #metrics
    """ print(result_for_metrics)
    #y = np.array([4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1]) #GT
    y = np.array(["4", "4", "4", "4", "4", "4", "4", "4", "0", "0", "3", "3", "3", "3", "3", "3", "3", "3", "3", "3", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "1", "1", "1", "1", "1", "1"]) #GT
    p = np.array(result_for_metrics) #prediction

    # sklearn 
    print('accuracy', metrics.accuracy_score(y,p) )
    print('precision', metrics.precision_score(y,p, average='micro') )
    print('recall', metrics.recall_score(y,p, average='micro') )
    print('f1', metrics.f1_score(y,p, average='micro') )

    print(metrics.classification_report(y,p))
    print(metrics.confusion_matrix(y,p))
 """
    #main(cli_args)
    print('elapsed time:', time.time() - start_time)

# %%
