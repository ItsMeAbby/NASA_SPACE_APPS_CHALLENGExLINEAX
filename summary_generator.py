import nltk
nltk.download('stopwords')
import fitz
import nltk
from fuzzywuzzy import process 
import numpy as np
from collections import Counter 
snow = nltk.stem.SnowballStemmer('english')
import pandas as pd
from nltk.corpus import stopwords
from nltk import *
import re
from uuid import uuid4
from pysummarization.nlp_base import NlpBase
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.similarityfilter.tfidf_cosine import TfIdfCosine
from sentence_sem_sim import sen_sem_sim
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


stopwords = set(nltk.corpus.stopwords.words('english') + ['reuter', '\x03'])
#stoplist = stopwords.words('english')
#stops = set(stopwords.words('english'))

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()
input_folder= "summary_dataset"
OutputFolder="output_folder"
print(list(stop))

stopwords = list(stop)+['a', 'about', 'above', 'across', 'after', 'afterwards','The','the','In','in','A','while','when','what','say','said']
stopwords += ['again', 'against', 'all', 'almost', 'alone', 'along','isbn','early','formal','normally','later']
stopwords += ['already', 'also', 'although', 'always', 'am', 'among']
stopwords += ['amongst', 'amoungst', 'amount', 'an', 'and', 'another']
stopwords += ['any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere']
stopwords += ['are', 'around', 'as', 'at', 'back', 'be', 'became']
stopwords += ['because', 'become', 'becomes', 'becoming', 'been']
stopwords += ['before', 'beforehand', 'behind', 'being', 'below']
stopwords += ['beside', 'besides', 'between', 'beyond', 'bill', 'both']
stopwords += ['bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant']
stopwords += ['co', 'con', 'could', 'couldnt', 'cry', 'de']
stopwords += ['describe', 'detail', 'did', 'do', 'done', 'down', 'due']
stopwords += ['during', 'each', 'eg', 'eight', 'either', 'eleven', 'else']
stopwords += ['elsewhere', 'empty', 'enough', 'etc', 'even', 'ever']
stopwords += ['every', 'everyone', 'everything', 'everywhere', 'except']
stopwords += ['few', 'fifteen', 'fifty', 'fill', 'find', 'fire', 'first']
stopwords += ['five', 'for', 'former', 'formerly', 'forty', 'found']
stopwords += ['four', 'from', 'front', 'full', 'further', 'get', 'give']
stopwords += ['go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her']
stopwords += ['here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers']
stopwords += ['herself', 'him', 'himself', 'his', 'how', 'however']
stopwords += ['hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed']
stopwords += ['interest', 'into', 'is', 'it', 'its', 'itself', 'keep']
stopwords += ['last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made']
stopwords += ['many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine']
stopwords += ['more', 'moreover', 'most', 'mostly', 'move', 'much']
stopwords += ['must', 'my', 'myself', 'name', 'namely', 'neither', 'never']
stopwords += ['nevertheless', 'next', 'nine', 'no', 'nobody', 'none']
stopwords += ['noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of']
stopwords += ['off', 'often', 'on','once', 'one', 'only', 'onto', 'or']
stopwords += ['other', 'others', 'otherwise', 'our', 'ours', 'ourselves']
stopwords += ['out', 'over', 'own', 'part', 'per', 'perhaps', 'please']
stopwords += ['put', 'rather', 're', 's', 'same', 'see', 'seem', 'seemed']
stopwords += ['seeming', 'seems', 'serious', 'several', 'she', 'should']
stopwords += ['show', 'side', 'since', 'sincere', 'six', 'sixty', 'so']
stopwords += ['some', 'somehow', 'someone', 'something', 'sometime']
stopwords += ['sometimes', 'somewhere', 'still', 'such', 'system', 'take']
stopwords += ['ten', 'than', 'that', 'the', 'their', 'them', 'themselves']
stopwords += ['then', 'thence', 'there', 'thereafter', 'thereby']
stopwords += ['therefore', 'therein', 'thereupon', 'these', 'they']
stopwords += ['thick', 'thin', 'third', 'this', 'those', 'though', 'three']
stopwords += ['three', 'through', 'throughout', 'thru', 'thus', 'to']
stopwords += ['together', 'too', 'top', 'toward', 'towards', 'twelve']
stopwords += ['twenty', 'two', 'un', 'under', 'until', 'up', 'upon']
stopwords += ['us', 'very', 'via', 'was', 'we', 'well', 'were', 'what']
stopwords += ['whatever', 'when', 'whence', 'whenever', 'where']
stopwords += ['whereafter', 'whereas', 'whereby', 'wherein', 'whereupon']
stopwords += ['wherever', 'whether', 'which', 'while', 'whither', 'who']
stopwords += ['whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with']
stopwords += ['within', 'without', 'would', 'yet', 'you', 'your']
stopwords += ['yours', 'yourself', 'yourselves']

print(stopwords)

def remove_url(txt):
        return " ".join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", txt).split())

def process_texts(texts):
    texts = [[word for word in line if word not in stopwords] for line in texts]
    texts = [bigram[line] for line in texts]  
    return texts
#contains your dataset of CSV files


def counter(comment_clear):
    cnt = Counter()
    for words in comment_clear:
        if words not in stopwords:
            cnt[words] += 1
    return cnt

def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,10), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer



def clean_data(raw_data):
    # ----------data pre-processing

    # 1- case folding - lower case
    raw_data = raw_data.casefold()

    # 2- cleaning - removing punctuations
    raw_data = re.sub(r'[^\s\w]', " ",raw_data) 
    raw_data = re.sub(r"[^a-zA-Z0-9]+", " ",raw_data)
    raw_data = re.sub(r"\b[a-zA-Z]\b", " ",raw_data)
    
    raw_data=re.sub('(\\b[A-Za-z] \\b|\\b [A-Za-z]\\b)', '', raw_data)
    return raw_data


#working on first csv only, 
# name=NameOfCSV[0] #first element of list of csv files 



# text=clean_data(file_data.replace("\n"," "))



totalWords=0
SummarizedWordsM1=0
SummarizedWordsM2=0
SummarizedWordsM3=0
documentWordsCount=0

nltk.download('omw-1.4')

import matplotlib.pyplot as plt
for file in os.listdir(os.path.join(os.curdir,input_folder)):
    print("working on file: ",file)
    file_data=""
    doc = fitz.open(os.path.join(os.curdir,input_folder,file))
    for page in doc:  # iterate the document pages
        file_data += page.get_text().encode("utf8").decode('utf-8',errors="ignore")
    
    text_original=clean_data(file_data.replace("\n"," "))
    text_original=clean(text_original)
    errorOccured=False

    plt.clf()
    
    print('''
    
    Working on {}
    
    '''.format(file))

    similarity_filter = TfIdfCosine()
    # The object of the NLP.
    nlp_base = NlpBase()
    # Set tokenizer. This is japanese tokenizer with MeCab.
    nlp_base.tokenizable_doc = SimpleTokenizer()
    # The object of `Similarity Filter`. 
    # The similarity obs erved by this object is so-called cosine similarity of Tf-Idf vectors.
    similarity_filter =  TfIdfCosine()

    # Set the object of NLP.
    similarity_filter.nlp_base = nlp_base

    # If the similarity exceeds this value, the sentence will be cut off.
    similarity_filter.similarity_limit = 1.0
    # The object of automatic sumamrization.
    auto_abstractor = AutoAbstractor()
    # Set tokenizer. This is japanese tokenizer with MeCab.
    auto_abstractor.tokenizable_doc = SimpleTokenizer()
    auto_abstractor.delimiter_list = [".", "\n"]
    # Object of abstracting and filtering document.
    abstractable_doc = TopNRankAbstractor()
    # Delegate the objects and execute summarization.
    content = auto_abstractor.summarize(text_original, abstractable_doc, similarity_filter)
    
    contents = sent_tokenize(str(content))
    contents = str(content).split('.')
    cont = []
    for i in range(0,(len(contents)-1)): #technique to exclude last scoring data tag last element of array
        cont.append(contents[i])
    articles = [article.lower() for article in cont]

    # Strip all punctuation from each article
    # This uses str.translate to map all punctuation to the empty string
    table = str.maketrans('', '', string.punctuation)
    articles = [article.translate(table) for article in articles]

    # Convert all numbers in the article to the word 'num' using regular expressions
    articles = [re.sub(r'\d+', ' ', article) for article in articles]

    # Print the first article as a running example
    articles = [[word for word in article.split() if word not in stopwords] for article in articles]
    text =str(articles).lower()
    #Define the count vectorizer that will be used to process the data
    count_vectorizer = CountVectorizer()
    #Apply this vectorizer to text to get a sparse matrix of counts
    count_matrix = count_vectorizer.fit_transform([text])
    #Get the names of the features
    features = count_vectorizer.get_feature_names()
    #Create a series from the sparse matrix
    content = " "
    content= pd.Series(count_matrix.toarray().flatten(), 
                  index = features).sort_values(ascending=False)
    
    contnt = word_tokenize(remove_url(str(cont)))
    cnt = counter(contnt)
    most_token = cnt.most_common(50)
    com_token = []
    for i in range(len(most_token)):
        com_token.append(most_token[i][0])   
    
    lemmatizer = WordNetLemmatizer() 
    stems = [lemmatizer.lemmatize(com_token) for com_token in com_token]
    filt_token = []
    token = nltk.pos_tag(stems)

    for i in range(len(token)):
        if token[i][1] in ['NN']:
            filt_token.append(token[i][0]) 
    unq_token = np.unique(filt_token)
    unq_token =remove_url(str(unq_token))
    unq_token = str(unq_token).split()

    sum_token = []
    for i in range(len(unq_token)):
        for j in range(len(unq_token)):
            s1 = str(unq_token[i])
            s2 = str(unq_token[j])
            if sen_sem_sim.word_similarity(s1,s2) > 0.2 and sen_sem_sim.word_similarity(s1,s2) <= 0.9:
                if s1 != s2:
                    sum_token.append(s1)
    sum_token = np.unique(sum_token)

    abs_sum = []
    sent = sent_tokenize(str(text_original))
    first_sent = sent_tokenize(str(sent[0]))

    highest = process.extractOne("test",first_sent)
    abs_sum.append(highest[0])
    for i in range(len(sum_token)):
        if i == 0:
            highest = process.extractOne(sum_token[i],sent)
            abs_sum.append(highest[0])
        else:
            highest = process.extractOne(sum_token[i],sent)
            abs_sum.append(highest[0])

    ext_date = []
    from date_extractor import extract_dates
    dates = extract_dates(str(cont))
    for i in range(len(dates)):
        ext_date.append(str(dates[i]).split(" "))
    ext_year = []
    for i in range(len(ext_date)):
        year = (str(ext_date[i][0]).split('-'))
        ext_year.append(year[0])

    ext_year = np.unique(ext_year)

    sents = sent_tokenize(str(text_original))
    ext_sents = []
    for i in range(len(ext_year)):
        for j in range(len(sents)):
            if (ext_year[i] in sents[j]) == True:
                #print(str(ext_year[i]) + "\n\n")
                if len(sents[j]) > 50:
                    #print(str(sents[j]))
                    ext_sents.append(str(sents[j]))
                    break;
    indexes = np.unique(ext_sents, return_index=True)[1]
    sorted_sents = [ext_sents[index] for index in sorted(indexes)]

    for i in range(len(sorted_sents)):
      abs_sum.append(sorted_sents[i])
    indexes = np.unique(abs_sum, return_index=True)[1]
    sorted_sents = [abs_sum[index] for index in sorted(indexes)]

    s = ' '
    for i in range(len(sorted_sents )):
        s = s + sorted_sents[i] 

    s = word_tokenize(s)
    ext_s = ' '
    #s = word_tokenize(s)
    for i in range(len(s)):
        ext_s = ext_s + ' ' + s[i]

    ext_s = remove_url(str(ext_s))

    # Convert each article to all lower case
    articles = [article.lower() for article in sorted_sents]

    # Strip all punctuation from each article
    # This uses str.translate to map all punctuation to the empty string
    table = str.maketrans('', '', string.punctuation)
    articles = [article.translate(table) for article in articles]

    # Convert all numbers in the article to the word 'num' using regular expressions
    articles = [re.sub(r'\d+', ' ', article) for article in articles]

    # Print the first article as a running example
    articles = [[word for word in article.split() if word not in stopwords] for article in articles]

    #Extract text for a particular person
    text =str(articles).lower()
    #Define the count vectorizer that will be used to process the data
    count_vectorizer = CountVectorizer()
    #Apply this vectorizer to text to get a sparse matrix of counts
    count_matrix2 = count_vectorizer.fit_transform([text])
    #Get the names of the features
    features2 = count_vectorizer.get_feature_names()

    summary = pd.Series(count_matrix2.toarray().flatten(),index = features2).sort_values(ascending=False)
    
    method1=100-(len(ext_s)/len(text_original) * 100)
    method1Cwords=len(ext_s)
    with open(f"{OutputFolder}/summary_{uuid4().hex}{file}.txt", 'w+') as f:
      content = f"Summary: \n{ext_s}"
      f.write(f"{content}")