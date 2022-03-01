import json
import os
import pandas as pd
def readData(path):
    with open(path, encoding='utf8') as json_file:
        content = json_file.read().replace('\n','')
        data = json.loads(content)
        dfResult = pd.json_normalize(data)
        return dfResult

if __name__ == '__main__':
    pathFolder = r'C:\Users\HuyBin\Downloads\laodong-kinhte-20220301T060054Z-001\laodong-kinhte'
    df = pd.DataFrame()
    for file in os.listdir(pathFolder):
        data = readData(pathFolder+'\\'+file)
        df = pd.concat([pd.DataFrame(data), df], ignore_index=True)
    df = pd.DataFrame(df, columns=['title','paragraph','tags'])
    print(df)