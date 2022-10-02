# NASA_SPACE_APPS_CHALLENGExLINEAX
Searching through millions of words in thousands of files. The project uses a combination of three Natural Language Processing algorithms; Summarization, Key Words identification and relevance based search. The input data, of no matter what size is vectorized and searched with respect to the query of the user.  
```
  git clone https://github.com/ItsMeAbby/NASA_SPACE_APPS_CHALLENGExLINEAX.git
  cd NASA_SPACE_APPS_CHALLENGExLINEAX
  pip install -r requirements.txt
  pip install fitz
  pip install PyMuPDF
```
#INPUT
  Input for all sultion is already defined, u can replace files in two folders to chnage dataset
  1. summary_dataset
  2. nasa_dataset

For Relevency Based Solution
input is nasa_dataset
```
  python relevency_based_search.py
```


For Keywords Extraction
input is nasa_dataset
```
  python keywords_extractor.py
```


For summarization
input is summary_dataset
```
  python summary_generator.py
```
