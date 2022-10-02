from fileinput import filename
import json  # for json dump dictionary
import math  # for log, sqrt, pow
import os  # provides portable way of using functionalities that are dependent on our os
import tkinter as tk
from tkinter import *  # for ui
from tkinter import scrolledtext, messagebox
import sys
from xml.dom import NAMESPACE_ERR
import nltk  # for tokenization (text to words)
import numpy  # for indices sorting function
import numpy as np
from PIL import ImageTk, Image
from nltk.stem import WordNetLemmatizer  # for lemmetization
import re
import fitz

alpha = 0.05
stopWords = []
# lemmatization objects which will be used in through out code for query and data set
lemmatizer = WordNetLemmatizer()
DATASET_FOLDER="nasa_dataset"
filenames = os.listdir(os.path.join(os.curdir, DATASET_FOLDER))
len_folder=len(filenames)
def close_vsm():
    print("Shutting down VSM Search engine.")
    sys.exit(1)


def VectorSpaceModel(event=None):
    result_box.delete('1.0', END)  # every time new result is sent to UI so deleting previous

    with open(os.path.join(os.curdir,"dictionary_vsm.json"), 'r') as dictionary:  # already made dictionary
        f1 = dictionary.read()
    Vsm_dictionary = json.loads(f1)
    Vsm_dictionary = dict(sorted(Vsm_dictionary.items(), key=lambda w: w[0]))  # list view
    user_query = query_input.get()

    user_query = clean_data(user_query)

    # 5-lemmatization - converting word to its root form
    for word in user_query:
        lemmatizer.lemmatize(word)

    query_dict = dict()
    for word in user_query:
        if word not in stopWords:  # if word not in stopwords
            if word not in query_dict:  # also not in query dict // 1st time here
                query_dict2 = {word: 1}
                query_dict.update(query_dict2)
            else:
                query_dict[word] += 1  # if is already in dictionary then ++ presence only

    # tf, idf, tf-idf, normalization of query
    normalized_query = 0
    for w in query_dict.keys():
        if w in Vsm_dictionary:
            query_dict[w] = 1 + math.log10(query_dict[w])  # ---tf !
            query_dict[w] *= Vsm_dictionary[w]["df"] / len_folder  # -----tf*idf !
            normalized_query += math.pow(query_dict[w], 2)
        else:
            messagebox.showerror('Not Found', w + " not found in dictionary.")
    normalized_query = math.sqrt(normalized_query)

    similarity = [0] * len_folder
    if normalized_query != 0:
        for j in range(0, len_folder):
            if Vsm_dictionary["normalized-values"][j] != 0:
                dot_product = 0
                for w in query_dict.keys():
                    if w in Vsm_dictionary:
                        # --------------------->    a.b
                        dot_product += (query_dict[w]) * (Vsm_dictionary[w]["postings"][j])
                # ----->   |a||b|
                similarity[j] = dot_product / (normalized_query * Vsm_dictionary["normalized-values"][j])

        sorted_doc_ids = numpy.argsort(similarity)
        # print(sorted_doc_ids)

        Length = 0

        # RANKED RETRIEVAL
        result_box.insert(END, "         Document_id                Similarity_Score\n\n")
        for i in range(len_folder-1, -1, -1):
            if similarity[sorted_doc_ids[i]] > float(alpha_input.get()):
                Length += 1
                result_box.insert(END, "     ")  # output on UI
                if (sorted_doc_ids[i] + 1) < 10:
                    result_box.insert(END, str(0))
                result_box.insert(END, str(filenames[sorted_doc_ids[i]]))
                result_box.insert(END, str("                   "))
                result_box.insert(END, str(np.round(similarity[sorted_doc_ids[i]], 6)))
                result_box.insert(END, str("\n"))

        retrieve_docs_count.config(text=f'{Length}')


def clean_data(raw_data):
    # ----------data pre-processing

    # 1- case folding - lower case
    raw_data = raw_data.casefold()

    # 2- cleaning - removing punctuations
    raw_data = re.sub(r'[^\s\w]', " ",
                      raw_data)  # replace everything with space, except word and spaces

    # 3- tokenization - breaking text into words
    raw_data = nltk.word_tokenize(raw_data)

    # 4-removing stopwords
    raw_data = [word for word in raw_data if word not in stopWords]

    return raw_data


def build_vsm_dictionary():
    # tokenizing stopwords list to remove them from data set or query
    stopWordFile = open(os.path.join(os.curdir,"Stopword-List.txt"), "r")
    for i in stopWordFile:
        stopWords.extend(nltk.word_tokenize(i))

    if not (os.path.isfile(os.path.join(os.curdir,'dictionary_vsm.json'))):
        Vsm_dict = dict()
        vocab_count = 0
        
        for i,filename in enumerate(os.listdir(os.path.join(os.curdir, DATASET_FOLDER))):
            doc = fitz.open(os.path.join(os.curdir, DATASET_FOLDER, filename))
            file_data=""
            for page in doc:  # iterate the document pages
                file_data += page.get_text().encode("utf8").decode('utf-8',errors="ignore")
            raw_data = file_data.replace('\n', ' ')

            raw_data = clean_data(raw_data)

            # 5-lemmatization - converting word to its root form
            for word in raw_data:
                lemmatizer.lemmatize(word)

            # dictionary formation and term frequency count
            for word in raw_data:
                if word not in stopWords:  # putting in dictionary only if it is valid: (pre-processed) and (not a stop word)
                    if word not in Vsm_dict:  # checking if word is already in dictionary: if not, then only add
                        tempVsm_dictionary = {
                            word: {"postings": [0], "df": 1}}  # generate a list there and make its doc freq 1
                        for y in range(0, len_folder):  #
                            tempVsm_dictionary[word]["postings"].append(
                                0)  # and as it is a new word, it will be 0 from other documents too
                        Vsm_dict.update(tempVsm_dictionary)  # new word added in dictionary
                        Vsm_dict[word]["postings"][i] += 1  # and its tf
                        vocab_count += 1
                    else:
                        Vsm_dict[word]["postings"][i] += 1  # if repeated then increment freq
                        if Vsm_dict[word]["postings"][i] == 1:
                            Vsm_dict[word]["df"] += 1

        # tf = 1 + log(tf)
        # idf = log(df) / N
        # tf-idf = tf * idf
        # normalizing using cosine similarity
        normalize = [0] * len_folder  # list for normalization values
        for word in Vsm_dict.keys():
            Vsm_dict[word]["df"] = (math.log10(Vsm_dict[word]["df"])) / len_folder  # idf ready!
            for tf in range(0, len_folder):
                if Vsm_dict[word]["postings"][tf] != 0:
                    Vsm_dict[word]["postings"][tf] = 1 + math.log10(Vsm_dict[word]["postings"][tf])  # tf ready!
                    Vsm_dict[word]["postings"][tf] *= Vsm_dict[word]["df"]  # tfidf = tf * idf
                    normalize[tf] += math.pow(Vsm_dict[word]["postings"][tf], 2)  # a^2 + b^2 + ....
        for i in range(0, len_folder):
            normalize[i] = math.sqrt(normalize[i])  # under_root(a^2 + b^2+...)
        Vsm_dict.update({"normalized-values": normalize})  # for all len_folder docs
        del normalize

        with open(os.path.join(os.curdir,"dictionary_vsm.json"), 'w+') as dictionary:
            dictionary.write(json.dumps(Vsm_dict,indent=2))
        del Vsm_dict


if __name__ == "__main__":
    # if not existing dictionary then create dictionary
    build_vsm_dictionary()

    # UI part starts here
    window = tk.Tk()
    window.title("LINEAX Vector Space Model")
    window.geometry("800x500")
    window.maxsize(800, 500)
    window.minsize(800, 500)
    window.configure(background='grey')

    backgroundImage = "Icons/background2.jpg"
    search_icon = PhotoImage(file=r"Icons/search_icon (2).png")
    search_icon = search_icon.subsample(3, 3)

    close_icon = PhotoImage(file=r"Icons/icons8-close-window-100-3.png")
    close_icon = close_icon.subsample(1, 1)

    img = ImageTk.PhotoImage(Image.open(backgroundImage))
    panel = tk.Label(window, image=img)
    panel.place(x=0, y=0, relwidth=1, relheight=1)

    window.iconphoto(False, tk.PhotoImage(file='Icons/search_icon (2).png'))

    query_label = Label(window, text="Query", bd=3, relief="sunken", font="Helvetica 12")
    query_label.place(x=20, y=20, relheight=0.08, relwidth=0.08)
    query_input = Entry(window, bd=3, relief="sunken", font="Helvetica 11 bold")
    query_input.place(x=90, y=20, relheight=0.08, relwidth=0.36)

    alpha_label = Label(window, text="Alpha ", bd=3, relief="sunken", font="Helvetica 12")
    alpha_label.place(x=390, y=20, relheight=0.08, relwidth=0.07)

    alpha_input = Entry(window, bd=3, relief="sunken", font="Helvetica 11 bold")
    alpha_input.place(x=450, y=20, relheight=0.08, relwidth=0.08)
    #
    search_icon_button = Button(window, image=search_icon, command=VectorSpaceModel)
    search_icon_button.place(x=530, y=20, relheight=0.18, relwidth=0.15)
    close_icon_button = Button(window, image=close_icon, command=close_vsm)
    close_icon_button.place(x=660, y=20, relheight=0.18, relwidth=0.15)
    #
    retrieve_docs_label = Label(window, text="Retrieved Documents", bd=3, relief="sunken", font="Helvetica 12")
    retrieve_docs_label.place(x=20, y=70, relheight=0.08, relwidth=0.3)
    retrieve_docs_count = Label(window, text="0", bd=3, relief="sunken", font="Helvetica 12")
    retrieve_docs_count.place(x=280, y=70, relheight=0.08, relwidth=0.1)

    result_box = scrolledtext.ScrolledText(window)
    result_box.place(x=20, y=120, relheight=0.75, relwidth=0.95)

    # query_input.insert(0, 'query')
    alpha_input.insert(0, alpha)
    query_input.bind('<Return>', VectorSpaceModel)
    alpha_input.bind('<Return>', VectorSpaceModel)
    window.mainloop()
