import csv
import string

class CSVFile(object):
    """docstring for CSVFile"""
    def __init__(self, fileanme="dataset/ConceptTeam1.csv"):
        super(CSVFile, self).__init__()
        self.fileanme = fileanme

    def getContent(self):
        conceptlist = list()

        with open(self.fileanme,'r') as csvfile:
            firstline = True
            reader = csv.reader(csvfile)
            for row in reader:
                if firstline:
                    firstline = False
                    continue
                def trim_str(str, exclude): return ''.join(ch for ch in str if ch not in exclude)
                ex = string.punctuation
                conceptlist.append([trim_str(row[1],ex),trim_str(row[2],ex),trim_str(row[3],ex)])
        return conceptlist

if __name__ == '__main__':
    file = CSVFile("dataset/ConceptTeam1.csv")
    print( file.getContent())