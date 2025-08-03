import sys
import numpy as np
import torch

from torch.nn import Module, Linear, Embedding, CrossEntropyLoss
from torch.nn.functional import relu, log_softmax
from torch.utils.data import Dataset, DataLoader 

from extract_training_data import FeatureExtractor

class DependencyDataset(Dataset):

  def __init__(self, inputs_filename, output_filename):
    self.inputs = np.load(inputs_filename)
    self.outputs = np.load(output_filename)

  def __len__(self): 
    return self.inputs.shape[0]

  def __getitem__(self, k): 
    return (self.inputs[k], self.outputs[k])


class DependencyModel(Module): 

  def __init__(self, word_types, outputs):
    # TODO: complete for part 3
    super(DependencyModel, self).__init__()
    self.embedding = Embedding(word_types,128)
    self.hidden_lay = Linear(128 * 6,128)
    self.output_lay = Linear(128,outputs)

  def forward(self, inputs):

    # TODO: complete for part 3
    embedded = self.embedding(inputs)
    flattened = embedded.view(embedded.shape[0], -1)
    hidden= self.hidden_lay(flattened)
    relu_hidden = relu(hidden)
    res = self.output_lay(relu_hidden)
    return res

def train(model, loader, device): 

  loss_function = CrossEntropyLoss(reduction='mean')

  LEARNING_RATE = 0.01 
  optimizer = torch.optim.Adagrad(params=model.parameters(), lr=LEARNING_RATE)

  tr_loss = 0 
  tr_steps = 0

  model.train()
 

  correct = 0 
  total =  0 
  for idx, batch in enumerate(loader):
 
    inputs, targets = batch

    inputs = inputs.long().to(device)
    targets = torch.argmax(targets, dim=1).to(device)


    predictions = model(inputs)

    loss = loss_function(predictions, targets)
    tr_loss += loss.item()

    #print("Batch loss: ", loss.item()) # Helpful for debugging, maybe 

    tr_steps += 1
    
    if idx % 1000==0:
      curr_avg_loss = tr_loss / tr_steps
      print(f"Current average loss: {curr_avg_loss}")


    predicted_labels = torch.argmax(predictions, dim=1)
    correct += (predicted_labels == targets).sum().item()
    total += targets.size(0)
      
    # Run the backward pass to update parameters 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


  epoch_loss = tr_loss / tr_steps
  acc = correct / total
  print(f"Training loss epoch: {epoch_loss},   Accuracy: {acc}")


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    if torch.backends.mps.is_available():
      device = torch.device("mps")
      print("Using MPS")
    else:
        device = torch.device("cpu")
        print("MPS not available")


    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary file {}.".format(WORD_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f)


    model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
    model.to(device)

    dataset = DependencyDataset(sys.argv[1], sys.argv[2])
    loader = DataLoader(dataset, batch_size = 16, shuffle = True)

    print("Done loading data")

    # Now train the model
    for i in range(5): 
      train(model, loader,device)


    torch.save(model.state_dict(), sys.argv[3]) 
