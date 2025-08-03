import sys
from conll_reader import conll_reader
from collections import defaultdict

def get_vocabularies(conll_reader):
    word_set = defaultdict(int)
    for dtree in conll_reader:
        for ident, node in dtree.deprels.items():
            if node.pos != "CD" and node.pos!="NNP":
                word_set[node.word.lower()] += 1

    word_set = set(x for x in word_set if word_set[x] > 1)

    word_list = ["<CD>","<NNP>","<UNK>","<ROOT>","<NULL>"] + list(word_set)

    return word_list

if __name__ == "__main__":
    with open(sys.argv[1],'r') as in_file, open(sys.argv[2],'w') as word_file:
        word_list = get_vocabularies(conll_reader(in_file))
        print("Writing word word indices...")
        for index, word in enumerate(word_list): 
            word_file.write("{}\t{}\n".format(word, index))
        
        


