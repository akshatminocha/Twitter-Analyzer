
global summaryArticle, summaryTweet

def getNewsArticleSummary(keyword):
    import nltk
    import heapq
    from newsapi import NewsApiClient
    api = NewsApiClient(api_key='') # Note: enter your news api key
    results = api.get_everything(q=keyword, language = 'en')
    iter = results['totalResults']
    article = ''
    for i in range(0, len(results['articles'])):
        article = article + results['articles'][i]['title'] + '. '
        if(results['articles'][i]['description'][0]!='?'):
            article = article + results['articles'][i]['description'] + '. '
        
    sentence_list = nltk.sent_tokenize(article)
    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(article):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(10,sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    print(summary)
    summaryArticle = summary
    return summary

def getTwitterSummary(keyword):
    import tweepy
    import re
    import nltk
    import heapq
    consumer_key = '' # Note: enter your consumer key
    consumer_secret = '' # Note: enter your consumer secret
    access_token = '' # Note: enter your access token
    access_token_secret = '' # Note: enter your access token secret
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth,wait_on_rate_limit=True)
    tweets = ''
    new_search = keyword + " -filter:retweets"
    for tweet in tweepy.Cursor(api.search,q=new_search,
                            lang="en",truncated=False,tweet_mode='extended',include_entities=True).items(100):

        if(tweet.retweeted):
            current = tweet.retweeted.full_text[:]
            regex_remove = "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^RT|http.+?"
            stripped_text = [re.sub(regex_remove, '',current).strip()]
            current = '. '.join(stripped_text)
            tweets = tweets + current + '. '
        else:
            current = tweet.full_text[:]
            regex_remove = "(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^RT|http.+?"
            stripped_text = [re.sub(regex_remove, '',current).strip()]
            current = '. '.join(stripped_text)
            tweets = tweets + current + '. ' 
            
    sentence_list = nltk.sent_tokenize(tweets)
    stopwords = nltk.corpus.stopwords.words('english')

    word_frequencies = {}
    for word in nltk.word_tokenize(tweets):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word]/maximum_frequncy)

    sentence_scores = {}
    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(10,sentence_scores, key=sentence_scores.get)

    summary = ' '.join(summary_sentences)
    print(summary)
    summaryTweet = summary
    return summary
    
def similarity(summary1, summary2):
    from nltk.corpus import stopwords 
    from nltk.tokenize import word_tokenize 
    articleSummary_list = word_tokenize(summary1)  
    tweetSummary_list = word_tokenize(summary2) 

    # sw contains the list of stopwords 
    sw = stopwords.words('english')  
    l1 =[];l2 =[]  

    # remove stop words from string 
    articleSummary_set = {w for w in articleSummary_list if not w in sw}  
    tweetSummary_set = {w for w in tweetSummary_list if not w in sw} 

    # form a set containing keywords of both strings  
    rvector = articleSummary_set.union(tweetSummary_set)  
    for w in rvector: 
        if w in articleSummary_set: l1.append(1) # create a vector 
        else: l1.append(0) 
        if w in tweetSummary_set: l2.append(1) 
        else: l2.append(0) 
    c = 0
    # cosine formula  
    for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    return cosine

def vennDiagram(sim):
    import matplotlib.pyplot as plt

    ## output of your first part
    cosine = [[ 1., sim]]

    ## set constants
    r = 1
    d = 2 * r * (1 - cosine[0][1])

    ## draw circles
    circle1=plt.Circle((0, 0), r, alpha=.5)
    circle2=plt.Circle((d, 0), r, alpha=.5)
    ## set axis limits
    plt.ylim([-1.1, 1.1])
    plt.xlim([-1.1, 1.1 + d])
    fig = plt.gcf()
    fig.gca().add_artist(circle1)
    fig.gca().add_artist(circle2)
    fig.savefig('venn_diagramm.png')
    from PIL import Image, ImageTk

    image = Image.open("venn_diagramm.png")
    photo = ImageTk.PhotoImage(image)
    venn.config(image=photo)
    venn.image = photo # keep a reference!

def summarize():
    articleSummary = getNewsArticleSummary(Hashtag.get())
    txt1.insert(END, articleSummary)
    twitterSummary = getTwitterSummary(Hashtag.get())
    txt2.insert(END, twitterSummary)
    resultLabel2.config(text = "Similarity is {0}".format(similarity(articleSummary, twitterSummary)))
    vennDiagram(float(similarity(articleSummary, twitterSummary)))




def predictor(url_count, followers_count, friends_count, user_created_at,user_favourites_count, geo_enabled, user_statuses_count, cs):
    import pickle
    import xgboost
    import pandas as pd
    
    list_pickle_path = 'final_prediction.pkl'
    list_unpickle = open(list_pickle_path, 'rb')
    model = pickle.load(list_unpickle)
    
    df = pd.DataFrame({'url_count':[url_count], 'followers_count': [followers_count],'friends_count':[friends_count], 'user_created_at': [user_created_at], 'user_favourites_count':[user_favourites_count], 'geo_enabled':[geo_enabled], 'user_statuses_count':[user_statuses_count], 'cs':[cs]}, dtype=int)

    prediction = model.predict(df) 
    
    if(prediction[0]==1):
        resultLabel.config(text = 'The user might follow back')
    else:
        resultLabel.config(text = 'The user might not follow back')
        
def submit():
    predictor(url_count.get(), followers_count.get(), friends_count.get(), user_created_at.get(),user_favourites_count.get(), geo_enabled.get(), user_statuses_count.get(), cs.get())
    




from tkinter import *
import tkinter as tk
import tkinter.ttk as ttk

# creating root
root = tk.Tk()
root.geometry("1500x700")
rows = 0

scrollbar = Scrollbar(root)
scrollbar.grid(sticky = 'ns')

# creating a frame for storing all widgets
AppFrame = tk.Frame(root)
AppFrame.grid(row=0, column=0,sticky='news')

# Creating label for title
AppTitle = tk.Label(AppFrame, text="Twitter Analyzer", font=("Times New Roman", 30))
AppTitle.grid(row=0, column=0)
AppTitle2 = tk.Label(AppFrame, text="", font=("Arial", 5))
AppTitle2.grid(row=0, column=1)

notebook = ttk.Notebook(AppFrame)
notebook.grid(row=2, column=0, sticky="W", rowspan=100, columnspan=230)

page1 = ttk.Frame(AppFrame)
notebook.add(page1, text='Follow-Back')

url_countL = ttk.Label(page1, text="url_count", padding=8)
url_countL.grid(row=3, column=0)
url_count = ttk.Entry(page1)
url_count.grid(row=3, column=1)

followers_countL = ttk.Label(page1, text="followers_count", padding=8)
followers_countL.grid(row=4, column=0)
followers_count = ttk.Entry(page1)
followers_count.grid(row=4, column=1)

friends_countL = ttk.Label(page1, text="friends_count", padding=8)
friends_countL.grid(row=5, column=0)
friends_count = ttk.Entry(page1)
friends_count.grid(row=5, column=1)

user_created_atL = ttk.Label(page1, text="user_created_at", padding=8)
user_created_atL.grid(row=6, column=0)
user_created_at = ttk.Entry(page1)
user_created_at.grid(row=6, column=1)

user_favourites_countL = ttk.Label(page1, text="user_favourites_count", padding=8)
user_favourites_countL.grid(row=7, column=0)
user_favourites_count = ttk.Entry(page1)
user_favourites_count.grid(row=7, column=1)

geo_enabledL = ttk.Label(page1, text="geo_enabled", padding=8)
geo_enabledL.grid(row=8, column=0)
geo_enabled = ttk.Entry(page1)
geo_enabled.grid(row=8, column=1)

user_statuses_countL = ttk.Label(page1, text="user_statuses_count", padding=8)
user_statuses_countL.grid(row=9, column=0)
user_statuses_count = ttk.Entry(page1)
user_statuses_count.grid(row=9, column=1)

verifiedL = ttk.Label(page1, text="verified", padding=8)
verifiedL.grid(row=10, column=0)
verified = ttk.Entry(page1)
verified.grid(row=10, column=1)

contributors_enabledL = ttk.Label(page1, text="contributors_enabled", padding=8)
contributors_enabledL.grid(row=11, column=0)
contributors_enabled = ttk.Entry(page1)
contributors_enabled.grid(row=11, column=1)

translation_enabledL = ttk.Label(page1, text="translation_enabled", padding=8)
translation_enabledL.grid(row=12, column=0)
translation_enabled = ttk.Entry(page1)
translation_enabled.grid(row=12, column=1)

user_is_friendL = ttk.Label(page1, text="user_is_friend", padding=8)
user_is_friendL.grid(row=3, column=2)
user_is_friend = ttk.Entry(page1)
user_is_friend.grid(row=3, column=3)

csL = ttk.Label(page1, text="cs", padding=8)
csL.grid(row=4, column=2)
cs = ttk.Entry(page1)
cs.grid(row=4, column=3)

pythonL = ttk.Label(page1, text="python", padding=8)
pythonL.grid(row=5, column=2)
python = ttk.Entry(page1)
python.grid(row=5, column=3)

machine_learningL = ttk.Label(page1, text="machine learning", padding=8)
machine_learningL.grid(row=6, column=2)
machine_learning = ttk.Entry(page1)
machine_learning.grid(row=6, column=3)

deep_learningL = ttk.Label(page1, text="deep learning", padding=8)
deep_learningL.grid(row=7, column=2)
deep_learning = ttk.Entry(page1)
deep_learning.grid(row=7, column=3)

engineerL = ttk.Label(page1, text="engineer", padding=8)
engineerL.grid(row=8, column=2)
engineer = ttk.Entry(page1)
engineer.grid(row=8, column=3)

data_scienceL = ttk.Label(page1, text="data science", padding=8)
data_scienceL.grid(row=9, column=2)
data_science = ttk.Entry(page1)
data_science.grid(row=9, column=3)

artificial_intelligenceL = ttk.Label(page1, text="artificial intelligence", padding=8)
artificial_intelligenceL.grid(row=10, column=2)
artificial_intelligence = ttk.Entry(page1)
artificial_intelligence.grid(row=10, column=3)

nlpL = ttk.Label(page1, text="nlp", padding=8)
nlpL.grid(row=11, column=2)
nlp = ttk.Entry(page1)
nlp.grid(row=11, column=3)

computersL = ttk.Label(page1, text="computers", padding=8)
computersL.grid(row=12, column=2)
computers = ttk.Entry(page1)
computers.grid(row=12, column=3)

# Button for applying pre-processing and prediction
processButton = ttk.Button(page1, text="Apply Prediction", width=40, command=submit)
processButton.grid(row=30, column=2)

# Label for showcasing the result.
resultLabel = ttk.Label(page1, text="\"Click Apply to see result\"",
                        font=("Arial", 20))
resultLabel.grid(row=32, column=2)


page2 = ttk.Frame(AppFrame)
notebook.add(page2, text='Summarizer')

HLabel = ttk.Label(page2, text="Enter Keyword/Hashtag")
HLabel.grid(row=0, column=0, sticky="E")

Hashtag = ttk.Entry(page2)
Hashtag.grid(row=0, column=1, sticky="W")

submitButton = ttk.Button(page2, text="Summarize", width=20, padding=10, command = summarize)
submitButton.grid(row=0, column=2, columnspan=8)

scr1=Scrollbar(page2)
scr1.grid(row=1,column=1,rowspan=10)
txt1=Text(page2,height=20,width=50)
txt1.grid(row=1,rowspan=10,column=0,columnspan=2)
txt1.configure(yscrollcommand=scr1.set)
scr1.configure(command=txt1.yview)

scr2=Scrollbar(page2)
scr2.grid(row=1,column=2,rowspan=10)
txt2=Text(page2,height=20,width=50)
txt2.grid(row=1,rowspan=10,column=5,columnspan=2)
txt2.configure(yscrollcommand=scr2.set)
scr2.configure(command=txt2.yview)

venn = ttk.Label(page2)
venn.grid(column=10, row = 3, ipadx = 10)


resultLabel2 = ttk.Label(page2, text="", font=("Arial", 15), padding=7)
resultLabel2.grid(row=13, column=0, columnspan = 2)

# main loop the root window
root.mainloop()




