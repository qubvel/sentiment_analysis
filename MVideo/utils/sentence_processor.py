import pickle
import gensim
import pymorphy2
import requests
import numpy as np
from nltk.tokenize import RegexpTokenizer


class SentenceProcessor(object):
    def __init__(self, w2v_model_path, stop_list=[], tokenizer_regexp=u'[а-яА-Яa-zA-Z]+'):
        self.w2v = self._load_w2v(w2v_model_path)
        self.morph = pymorphy2.MorphAnalyzer()
        self.tokenizer = RegexpTokenizer(tokenizer_regexp)
        self.stop_list = []
        
    def _load_w2v(self, w2v_model_path):
        w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_model_path, binary=True, unicode_errors='ignore')
        w2v.init_sims(replace=True)
        return w2v
    
    def _make_bag_of_words(self, sample):
        if type(sample) is list:
            pass
        elif type(sample) is str:
            sample = sample.split()
        else:
            raise Exception('Sample should be string or list of words')
        return sample
        
    def tokenize(self, sample):
        '''make tokenization, return bag of words'''
        return self.tokenizer.tokenize(sample)
    
    def correction(self, sample):
        bag_of_words = self._make_bag_of_words(sample)
        corrected_bag = []
        for word in bag_of_words:
            if word not in self.w2v.vocab:
                word = self.correct_word(word)
            corrected_bag.append(word)
        return corrected_bag

    def normalize(self, sample):
        """make words normalization"""
        bag_of_words = self._make_bag_of_words(sample)
        return [self.morph.parse(word)[0].normal_form for word in bag_of_words]

    def correct_word(self, word):
        try:
            r = requests.get("http://speller.yandex.net/services/spellservice.json/checkText?text={}".format(word), timeout=10.)
            correct_word = r.json()[0]['s'][0]
            return correct_word
        except (KeyError, IndexError):
            return word
    
    def delete_stop_words(self, sample, stop_list=[]):
        """delete all garbage words from sample"""
        if not stop_list:
            stop_list = self.stop_list
        
        bag_of_words = self._make_bag_of_words(sample)

        for word in bag_of_words:
            if word.lower() in stop_list:
                bag_of_words.remove(word)

        return sample
    
    def process(self, sample, tokenize=True, correction=True, normalize=True, delete_stop_words=True):
        
        sample = sample.lower()
        
        if tokenize:
            sample = self.tokenize(sample)

        if correction:
            sample = self.correction(sample)
            
        if normalize:
            sample = self.normalize(sample)
            
        if delete_stop_words:
            sample = self.delete_stop_words(sample)
        
        return sample
    
    def cut_or_add(self, sample, sample_len):
        
        while len(sample) < sample_len and len(sample) != 0:
            sample.append(np.zeros_like(sample[0], dtype=np.float32))
        
        if len(sample) > sample_len:
            sample = sample[:sample_len]
            
        return sample
    
    def convert2matrix(self, sample, sample_len=None):

        bag_of_words = self._make_bag_of_words(sample)

        bag_of_vectors = []
        for word in bag_of_words:
            try:
                bag_of_vectors.append(self.w2v.word_vec(word))
            except KeyError:
                pass

        if sample_len:
            bag_of_vectors = self.cut_or_add(bag_of_vectors, sample_len)

        matrix = np.array(bag_of_vectors)
        
        return matrix