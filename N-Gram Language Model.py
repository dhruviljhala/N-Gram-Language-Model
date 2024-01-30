#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 17:19:39 2024

@author: dhruviljhala
"""

import nltk
from nltk.corpus import reuters
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk import ngrams
from collections import defaultdict
from tkinter import *
from functools import reduce


nltk.download('reuters')
nltk.download('punkt')
nltk.download('stopwords')

documents = reuters.fileids()

# Combining all documents
corpus = ''
for doc_id in documents:
    text = reuters.raw(doc_id)
    corpus += ' ' + text

# Convert to lower case
corpus = corpus.lower()

# Cleaning the corpus by removing numbers, symbols and space
cleaned_corpus = ''
for char in corpus:
    if char.isalpha() or char.isspace():
        cleaned_corpus += char

tokens = word_tokenize(cleaned_corpus)

all_words = set(tokens)


# Generate n-grams
def generate_ngrams(tokens, n):
    return list(ngrams(tokens, n))


# Function to get frequency given the ngram tuple 
def get_frequency_n_gram(ngrams):
    n_gram_freq = defaultdict(list)

    for ngm in ngrams:
        if ngm in n_gram_freq:
            n_gram_freq[ngm] += 1
        else:
            n_gram_freq[ngm] = 1

    return n_gram_freq


# Function to calculate the probability using the frequency of n and n-1 gram
def calculate_probability(prefix, all_words, ngram_freq, ngram_minus_one_freq):
  
    next_probability = defaultdict(list)
    for word in all_words:
        new_text = prefix + (str(word),)

        if not ngram_freq[new_text]:
            numerator = 0
        else:
            numerator = ngram_freq[new_text]
        
        if not ngram_minus_one_freq[prefix]:
            denominator = 0
        else:
            denominator = ngram_minus_one_freq[prefix]

        prob = (numerator + 1) / (denominator + len(all_words))
        #print(word, prefix, prob)
        next_probability[word] = prob

    
    return next_probability
    

# Function to get next word based on the previous n-1 gram
def predict_next_word(prefix, all_words, ngram_freq, ngram_minus_one_freq):
    probabilities = calculate_probability(prefix, all_words, ngram_freq, ngram_minus_one_freq)
    next_word = max(probabilities, key=lambda k: probabilities[k])
    
    return next_word


# Function to generate a sentence of length  given based on prefix
def generate_sentence(prefix, length, ngram_size):

    ngram_size_minus_one = ngram_size - 1

    # Generate n-grams
    ngram = generate_ngrams(tokens, ngram_size)
    ngram_minus_one = generate_ngrams(tokens, ngram_size_minus_one)

    # Generate Freeq for N-Gram
    ngram_freq = get_frequency_n_gram(ngram)
    ngram_minus_one_freq = get_frequency_n_gram(ngram_minus_one)

    # Print the result
    print(len(ngram), ngram[:10])
    print(len(ngram_minus_one), ngram_minus_one[:10])

    for i in range(length - len(prefix)):
        next_word = predict_next_word(tuple(prefix[-(ngram_size-1):]), all_words, ngram_freq, ngram_minus_one_freq)
        if next_word is not None:
            prefix += (next_word,)
        else:
            break
    return ' '.join(prefix)




def main():
    root = Tk()
 
    root.title("N-Gram Language Model")
    root.geometry('700x400')
     
    # adding a label to root
    lbl = Label(root, text = "Enter prefix with comma seperator: ")
    lbl.grid()
     
    # adding data entry box to enter prefix
    prefix_input = Entry(root, width=50)
    prefix_input.grid(column =1, row =0)
    
    
    lbl2 = Label(root, text = "Enter the length of sentence ")
    lbl2.grid()
      
    length_input = Entry(root, width=50)
    length_input.grid(column =1, row =1)
    
    
    lbl4 = Label(root, text = "Enter the n-gram value ")
    lbl4.grid()
      
    ngram_input = Entry(root, width=50)
    ngram_input.grid(column =1, row =2)
    
    
    lbl3 = Label(root, text = "")
    lbl3.grid(column =1, row =5)
    
    
    # function to run the sentence generation function when the button is clicked 
    def clicked():
        prefix_list = prefix_input.get().split(', ')
        sentence_len = int(length_input.get())
        ngram_val = int(ngram_input.get())
        prefix_tuple = tuple(prefix_list)
        
        sentence = generate_sentence(prefix_tuple, sentence_len, ngram_val)
        
        lbl3.configure(text = sentence)
     
    # button widget with red color text inside
    btn = Button(root, text = "Generate Sentence" , fg = "red", command=clicked)
    btn.grid(column=1, row=4)
     
    root.mainloop()

    return 

main() 
