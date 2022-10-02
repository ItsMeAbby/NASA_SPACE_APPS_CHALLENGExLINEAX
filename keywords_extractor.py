from uuid import uuid4
import yake
import json
import fitz
import os
import re
numOfKeywords = 10
output_folder="output_folder"
output_json={
    
}


def clean_data(raw_data):
    # ----------data pre-processing

    # 1- case folding - lower case
    raw_data = raw_data.casefold()

    # 2- cleaning - removing punctuations
    raw_data = re.sub(r'[^\s\w]', " ",raw_data) 
    raw_data = re.sub(r"[^a-zA-Z0-9]+", " ",raw_data) 
    
    raw_data=re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', raw_data)
    return raw_data


for file in os.listdir(os.path.join(os.curdir,"nasa_dataset")):
    print("working on file: ",file)
    file_data=""
    doc = fitz.open(os.path.join(os.curdir,"nasa_dataset",file))
    for page in doc:  # iterate the document pages
        file_data += page.get_text().encode("utf8").decode('utf-8',errors="ignore")
    
    text=clean_data(file_data.replace("\n"," "))
    # print(text)
    language = "en"
    max_ngram_size = 2
    deduplication_threshold = 0.75
    deduplication_algo = 'seqm'
    windowSize = 6
    

    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)

    temp_list={}
    for kw in keywords:
        temp_list.update({kw[0]:text.count(kw[0])})
        
          
    output_json.update({
        os.path.join(os.curdir,"nasa_dataset",file):temp_list
                        })
    
try:
    os.makedirs(os.path.join(os.curdir,output_folder))
except:
    pass
with open(os.path.join(os.curdir,output_folder,"key_words_"+str(uuid4().hex) +".json"), "w") as outfile:
    json.dump(output_json, outfile,indent=4)
    
    