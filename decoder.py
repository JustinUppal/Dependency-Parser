import sys
import copy

import numpy as np
import torch

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
from train_model import DependencyModel

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.extractor = extractor

        # Create a new model and load the parameters
        self.model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])
    def get_model_predictions(self, features):
        with torch.no_grad():
            model_output = self.model(features)
            probabilities = torch.nn.functional.softmax(model_output, dim=1).squeeze(0)
        return probabilities
    def get_transition(self, probs, state):
        sorted_idx = torch.argsort(probs, descending=True)
        for action_idx in sorted_idx:
            transition, label = self.output_labels[action_idx.item()]

            if transition == "shift":
                if not (len(state.buffer) == 1 and len(state.stack) > 0):
                    return transition, label
                
            elif transition == "left_arc":
                if len(state.stack) > 0 and state.stack[-1] != 0:
                    return transition, label
                
            elif transition == "right_arc":
                if len(state.stack) > 0:
                    return transition, label
        return None, None

    def parse_sentence(self, words:list, pos:list) -> DependencyStructure:

        state = State(range(1,len(words)))
        state.stack.append(0)

        # TODO: Write the body of this loop for part 5

        while state.buffer:
            
            features_np = self.extractor.get_input_representation(words, pos, state)
            features = torch.from_numpy(features_np).unsqueeze(0)

            probs = self.get_model_predictions(features)

            transition, label = self.get_transition(probs,state)
            if transition == "shift":
                state.shift()
            elif transition == "left_arc":
                state.left_arc(label)
            elif transition == "right_arc":
                state.right_arc(label)
            elif transition is None:
                break


        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))

        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
