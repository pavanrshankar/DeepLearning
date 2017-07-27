import nltk
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
import numpy as np 
import random
from collections import Counter

lemmatizer = WordNetLemmatizer()
			
def create_dictionary(pos,neg):
	total_words = []
	for fi in [pos, neg]:
		fp = open(fi, 'r')
		lines = fp.readlines()
		for l in lines:
			words = word_tokenize(l.lower())
			total_words += words

	total_words = [lemmatizer.lemmatize(i) for i in total_words]
	
	final_word_list = []
	word_count = Counter(total_words)
	for w in word_count:
		if 30 < word_count[w] < 1000:
			final_word_list.append(w)

	return final_word_list		

def sample_file_handling(file, final_words_list, classification):
	featureset = []
	fp = open(file, 'r')
	lines = fp.readlines()
	for l in lines:
		features = np.zeros(len(final_words_list))
		word_list = word_tokenize(l.lower())	
		word_list = [lemmatizer.lemmatize(i) for i in word_list]
		for w in word_list:
			if w in final_words_list:
				idx = final_words_list.index(w)
				features[idx] = features[idx] + 1
		featureset.append([features, classification])

	return featureset

def create_feature_sets_and_labels(pos,neg,test_size=0.1):
	total_word_list = create_dictionary(pos,neg)
	features = sample_file_handling(pos,total_word_list,[1,0])
	features += sample_file_handling(neg,total_word_list,[0,1])
	random.shuffle(features)

	testing_size = int(test_size * len(features))
	training_size = len(features) - testing_size
		
	#converted to array to enable slicing operations	
	features = np.array(features)	
	#converted to list for comma separated array elements
	train_x = list(features[:,0][:training_size]) 
	train_y = list(features[:,1][:training_size]) 
		
	test_x = list(features[:,0][training_size:])
	test_y = list(features[:,1][training_size:])

	return train_x, train_y, test_x, test_y

if __name__ == '__main__':
	create_feature_sets_and_labels('pos.txt', 'neg.txt')	