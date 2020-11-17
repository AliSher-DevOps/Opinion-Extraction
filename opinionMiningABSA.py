
import nltk
from nltk.corpus import stopwords
import stanfordnlp
import pandas as pd


def aspect_base_sentiment_analysis_opinion_mining(input_text, stop_words, nlp):
    
    input_text = input_text.lower() #convert input_text into lowerCase
    sentenceList = nltk.sent_tokenize(input_text) # Splitting the input_text into sentences
    fcluster = []
    totalfeatureList = []
    finalcluster = []
    dic = {}

    for line in sentenceList:
        newtaggedList = []
        txt_list = nltk.word_tokenize(line) # splitting up into words
        taggedList = nltk.pos_tag(txt_list) # doing Part-of-Speech Tagging to each word
        newwordList = []
        flag = 0
        for i in range(0,len(taggedList)-1):
            if(taggedList[i][1]=="NN" and taggedList[i+1][1]=="NN"): # if 2 consecutive words are Nouns then joined together
                newwordList.append(taggedList[i][0]+taggedList[i+1][0])
                flag=1
            else:
                if(flag==1):
                    flag=0
                    continue
                newwordList.append(taggedList[i][0])
                if(i==len(taggedList)-2):
                    newwordList.append(taggedList[i+1][0])

        finaltxt = ' '.join(word for word in newwordList) 
        new_txt_list = nltk.word_tokenize(finaltxt)
        wordsList = [w for w in new_txt_list if not w in stop_words]
        taggedList = nltk.pos_tag(wordsList)

        doc = nlp(finaltxt) #  stanford NLP Pipeleine Object 
        
        # getting the dependency relations betwwen the words
        dep_node = []
        for dep_edge in doc.sentences[0].dependencies:
            dep_node.append([dep_edge[2].text, dep_edge[0].index, dep_edge[1]])

        # coverting and format
        for i in range(0, len(dep_node)):
            if (int(dep_node[i][1]) != 0):
                dep_node[i][1] = newwordList[(int(dep_node[i][1]) - 1)]

        featureList = []
        categories = []
        for i in taggedList:
            if(i[1]=='JJ' or i[1]=='NN' or i[1]=='JJR' or i[1]=='NNS' or i[1]=='RB'):
                featureList.append(list(i)) # For features for each sentence
                totalfeatureList.append(list(i)) # Stores the features of all the sentences in the text
                categories.append(i[0])

        for i in featureList:
            filist = []
            for j in dep_node:
                if((j[0]==i[0] or j[1]==i[0]) and (j[2] in ["nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"])):
                    if(j[0]==i[0]):
                        filist.append(j[1])
                    else:
                        filist.append(j[0])
            fcluster.append([i[0], filist])
            
    for i in totalfeatureList:
        dic[i[0]] = i[1]
    
    for i in fcluster:
        if(dic[i[0]]=="NN"):
            finalcluster.append(i)
        
    return(finalcluster)

nlp = stanfordnlp.Pipeline()
stop_words = set(stopwords.words('english'))
df_coc = pd.read_csv("./reviews.csv", encoding="utf-8")
input_text = "sound quality is good but price is high"

print(input_text)
print(aspect_base_sentiment_analysis_opinion_mining(input_text, stop_words, nlp))
print("\n")