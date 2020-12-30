import numpy as np
import json

def write_json(write_path, y_pred_loc, y_pred_cls, total_len=800):
    #y_pred_loc = y_pred_loc.tolist()
    #y_pred_cls = y_pred_cls.tolist()
    print(y_pred_loc)
    print(y_pred_cls)
    data = []   
    for i in range(0, total_len):
        data.append({
            'file_name': str(y_pred_loc[i]).replace('.txt',''),
            'class code': str(y_pred_cls[i])
        })
    output = {
        "annotations" : data
    }

    #txt_path = write_path.replace('.json', '.txt')
    #with open(txt_path, 'w') as outfile:
    #    json.dump(output, outfile, ensure_ascii=False, indent="    ")

    with open(write_path, 'w') as outfile:
        json.dump(output, outfile, ensure_ascii=False, indent="    ")

    print('txt & json file saved.')


if __name__ == "__main__":
    filepath = 'output.json'
    y_pred_loc = np.random.randint(10, size=(1000, 1))
    y_pred_cls = np.random.randint(10, size=(1000, 1))

    write_json(filepath, y_pred_loc.tolist(), y_pred_cls.tolist())

