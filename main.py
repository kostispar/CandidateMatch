from selenium import webdriver
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import textdistance
from nltk.corpus import stopwords
import PyPDF2
import argparse
import sys
from pynput.keyboard import Key, Controller
from selenium.webdriver.firefox.options import Options
import matplotlib.pyplot as plt
import re
import numpy as np
import os
import string
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
sys.setrecursionlimit(10**6)

def similarity(cleaned_lines_ad, cleaned_lines_cv):
    total_score = 0
    i = 0
    for x in range(len(cleaned_lines_ad)):
        for y in range(len(cleaned_lines_cv)):
            lst_strings = [cleaned_lines_ad[x],cleaned_lines_cv[y]]
            vectorizer = CountVectorizer().fit_transform(lst_strings)
            vectors = vectorizer.toarray()
            score_jac = get_jaccard_sim(cleaned_lines_ad[x],cleaned_lines_cv[y])
            score_cosine = cosine_sim_vectors(vectors[0],vectors[1])
            score_rand = randcliff(cleaned_lines_ad[x],cleaned_lines_cv[y])
            score_jaro = jaro(cleaned_lines_ad[x],cleaned_lines_cv[y])
            score_lev = lev(cleaned_lines_ad[x],cleaned_lines_cv[y])
            score_seq = seq_sim(cleaned_lines_ad[x],cleaned_lines_cv[y])
            total_line = (score_lev + score_rand + score_cosine + score_jac + score_seq + score_jaro)/6
            i += 1
            total_score += total_line
    print("TOTAL SCORE IS: ", (total_score/i))
    return(total_score/i)


def cosine_sim_vectors(vec1,vec2):
    vec1 = vec1.reshape(1,-1)
    vec2 = vec2.reshape(1, -1)
    return cosine_similarity(vec1,vec2)[0][0]

def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def randcliff(string1,string2):
    return textdistance.ratcliff_obershelp(string1, string2)

def jaro(string1,string2):
    return textdistance.jaro_winkler.normalized_similarity(string1, string2)

def lev(string1, string2):
    return textdistance.levenshtein.normalized_similarity(string1, string2)

def seq_sim(string1,string2):
    return textdistance.lcsstr.normalized_similarity(string1,string2)

def clean(doc):
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    final = ''.join([i for i in normalized if not i.isdigit()])
    return final


def tokenize(text_file):
    lst_lines = text_file.split('\n')
    cleaned_lines = []
    tokenized_lines = []
    for line in lst_lines:
        cleaned_lines.append(clean(line))
        tokenized_lines.append(nltk.tokenize.word_tokenize(clean(line)))
    keywords = nltk.tokenize.word_tokenize(text_file)
    cleaned_lines = [x for x in cleaned_lines if x != '']
    tokenized_lines = [x for x in tokenized_lines if x != []]
    tokenized_cleaned_lines = []
    for list in tokenized_lines:
        text = ""
        for word in list:
            text += " "
            text += word
        tokenized_cleaned_lines.append(text)
    return (cleaned_lines)


def read_pdf(path):
    pdfFileObj = open(path, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    no_pages = pdfReader.numPages
    text = ""
    for page in range(no_pages):
        pageObj = pdfReader.getPage(page)
        text += (pageObj.extractText())
    return(text)

def scrape(url,cv_path):
    #job ad scrape
    options = Options()
    options.headless = True
    keyboard = Controller()
    driver = webdriver.Firefox(executable_path = "E:\\strikes\\geckodriver.exe", options=options)
    driver.get(url)
    location = driver.find_element_by_xpath("/html/body/div[3]/div/div/div[1]/div[1]/div[2]/div/div/div[2]/div[1]/div[1]/div[2]/div/div/div[3]")
    location = (location.text)
    title = driver.find_element_by_xpath("/html/body/div[3]/div/div/div[1]/div[1]/div[2]/div/div/div[2]/div[1]/div[1]/div[2]/div/div/div[2]")
    title = title.text
    company = driver.find_element_by_xpath("/html/body/div[3]/div/div/div[1]/div[1]/div[2]/div/div/div[2]/div[1]/div[1]/div[2]/div/div/div[1]")
    company = company.text
    sep ="\n"
    company = company.split(sep, 1)[0]
    text_ad = driver.find_element_by_xpath("/html/body/div[3]/div/div/div[1]/div[4]/div/div/div/div/ul[2]")
    text_ad   = text_ad.text
    cleaned_lines_ad = tokenize(text_ad)
    #cv scrape
    lst_results = []
    lst_names = []
    remove_words = ['\.pdf','CV','_','\d+']
    for cv in os.listdir(cv_path):
        text_cv = read_pdf(os.path.join(cv_path, cv))
        title = re.sub('(\.pdf|\d+|CV|Resume|-)', '', cv)
        title = re.sub('(_)', ' ', title)
        lst_names.append(title)
        cleaned_lines_cv = tokenize(text_cv)
        score = similarity(cleaned_lines_ad,cleaned_lines_cv)
        lst_results.append(round(score*100,2))
    fig, ax = plt.subplots()
    rects = ax.bar(range(len(lst_names)),lst_results,width = 0.5)
    ax.set_ylabel('Similarity')
    ax.set_title('Candidate scores')
    x = np.arange(len(lst_names))
    ax.set_xticks(x)
    ax.set_xticklabels(lst_names)
    ax.set_ylim([0, 50])
    ax.tick_params(axis='x', which='major', labelsize=8)
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height)+ '%',
                ha='center', va='bottom')
    plt.show()






def parseArguments():
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-u' , '--url', nargs=None,required=True,help="Url of job post")
    parser.add_argument('-c', '--cv_folder', nargs=None, required=True, help="path to CVs folder")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parseArguments()
    url = args.url
    cv_path = args.cv_folder
    scrape(url,cv_path)
















