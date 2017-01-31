from CSVFile import CSVFile
import nltk
# import Embedding as emb
import numpy as np

#Part-of-speech tagger
def tagPOS(sentence):
	text = nltk.word_tokenize(sentence)
	# print text
	print (nltk.pos_tag(text))

tagWeight = {'NN':2, 'NNP':2, 'VBG':1}

class ConceptItem(object):
	"""docstring for ConceptItem"""
	def __init__(self, arg):
		super(ConceptItem, self).__init__()
		self.concept = arg[0]
		self.description = arg[1]
		self.category = arg[2]
		self.found = True
		self.lowemb = []

	def conceptName(self):
		return self.concept

	def getCategory(self):
		return self.category

	def tagBag(self, sentence):
		sentence = sentence.replace("/"," ")
		tagBag = tagWeight = {'NN':[], 'NNP':[], 'VBG':[]}
		text = nltk.word_tokenize(sentence)
		posList = nltk.pos_tag(text)
		for pos in posList:
			try:
				tagBag[pos[1]].append(pos[0])
			except Exception as e:
				pass
		return tagBag

	def conceptBag(self):
		return self.tagBag(self.concept)

	# def itemVector(self):
	# 	itemVec = np.zeros(128)
	# 	weightSum = 0.0
	# 	bag = self.tagBag(self.concept)
	# 	for tag in bag:
	# 		for word in bag[tag]:
	# 			try:
	# 				itemVec = np.add(itemVec,emb.wordVec(word.lower())) * tagWeight[tag]
	# 				weightSum = weightSum + tagWeight[tag]
	# 			except Exception as e:
	# 				print (word + ' not found')
	# 	if weightSum==0:
	# 		self.found = False
	# 		return np.zeros(128)
	# 	return (itemVec/weightSum)

	def isFound(self):
		return self.found

	def setLowEmb(self,lowemb):
		self.lowemb = lowemb

	def lowEmb(self):
		return self.lowemb

	def fullConcept(self):
		return (self.concept.replace('-',' ') + " " +self.description.replace('-',' ')).split()

if __name__ == '__main__':
	file = CSVFile()
	conceptlist = file.getContent()[0:5]
	for item in conceptlist:
		conceptItem = ConceptItem(item)
		print (conceptItem.fullConcept())

		