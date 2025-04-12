import torch
import torch.nn as nn
import re
import random
from torch.amp import autocast, GradScaler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd
from math import floor
from sklearn.model_selection import train_test_split
import pickle
from random import randint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

def load_glove_embeddings(glove_path, word_to_index, embedding_dim=100):
    embeddings_index = {}

    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = torch.tensor([float(x) for x in values[1:]], dtype=torch.float)
            embeddings_index[word] = vector

    embedding_matrix = torch.randn(len(word_to_index), embedding_dim)

    for word, idx in word_to_index.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]

    return embedding_matrix

data = pd.read_csv("interactions_25000.csv")

data = data.iloc[:20000]

data['source_text'] = "<user> " + data['source_text'].str.lower() + " </user> "

data['target_text'] = " <bot> " + data['target_text'] + " </bot> "

tokens_set = " ".join(data["source_text"].astype(str) + data["target_text"].astype(str))

pattern = r'(<bot>|</bot>|<user>|</user>|[^a-zA-Z0-9<>?]+|[?])'

values_set = list(set(re.split(pattern, tokens_set)))

word_to_index = {value: values_set.index(value) for value in values_set}

word_to_index["<pad>"] = len(word_to_index)

padding_token = word_to_index['<pad>']

print(f"Paddin token = {padding_token}")

embedding_dim = 100
embedding_matrix = load_glove_embeddings("glove.6B.100d.txt", word_to_index, embedding_dim)
embedding_matrix = embedding_matrix.to(device)

index_to_word = {idx: value for value, idx in word_to_index.items()}

with open("word_to_index.pkl", "wb") as file:
  pickle.dump(word_to_index, file)

with open("index_to_word.pkl", "wb") as file:
  pickle.dump(index_to_word, file)

sos_token_idx = word_to_index["<bot>"]

end_token_id = word_to_index["</bot>"]

num_words = len(index_to_word)

print(f"num_words = {num_words}")

inputs = data["source_text"].astype(str)

targets = data["target_text"].astype(str)


def tokenization(text):

    pattern = r'(<bot>|</bot>|<user>|</user>|[^a-zA-Z0-9<>?]+|[?])'
    text = re.split(pattern, text)

    text = [word for word in text if not word.isspace()]

    text = [word for word in text if word != ""]

    words_numeric_list = [word_to_index[word] if word in word_to_index else 0 for word in text]

    return words_numeric_list

class Encoder(nn.Module):
  def __init__(self,embedding_matrix, hidden_size, num_layers):
    super().__init__()
    self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = False, padding_idx = padding_token)
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,num_layers=num_layers, batch_first=True)

  def forward(self, x, x_lenghts):
    x = self.embedding(x).float()
    x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lenghts.cpu().int(), batch_first = True, enforce_sorted=False)
    x, (hn, cn) = self.lstm(x)

    return x, hn, cn

class Decoder(nn.Module):
  def __init__(self, embedding_matrix, hidden_size, num_layers):
    super().__init__()
    self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = False, padding_idx = padding_token)
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size,num_layers=num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, num_words)

  def forward(self, x, hn, cn):

    x = self.embedding(x).float()

    x, (hn, cn) = self.lstm(x, (hn, cn))

    x = self.fc(x)

    return x, hn, cn

class seq2seq(nn.Module):

  def __init__(self, encoder, decoder, teach_forcing_ratio, temperature):
    super().__init__()

    self.encoder = encoder
    self.decoder = decoder
    self.teach_forcing_ratio = teach_forcing_ratio
    self.temperature = temperature

  def forward(self, src, src_len, tgt, tgt_len, epoch, inference, max_len_targets = None):

    if inference:
        return self.greedy_decode(src, src_len, max_len_targets)

    self.teach_forcing_ratio = max(0.1, (0.964 ** epoch))

    #print(f"teach forcing ratio = {self.teach_forcing_ratio}")

    max_len = tgt.shape[1]

    batch_size = src.shape[0]

    tgt_len = tgt_len.to(device)

    outputs = torch.zeros(batch_size, max_len, num_words)
    encoder_outputs, hn, cn = self.encoder(src, src_len)

    input = tgt[:, 0]

    finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
    finished_sequences = finished_sequences.to(device)

    for t in range(1, max_len):


      output, hn, cn = self.decoder(input.view(-1, 1), hn, cn)

      output = output.squeeze(1)

      outputs[:, t,:] = output

      teach_forcing = random.random() < self.teach_forcing_ratio

      output = output / self.temperature

      output = torch.nn.functional.softmax(output, dim =1)

      next_input_generated = torch.multinomial(output, 1).squeeze(1)



      next_input = tgt[:, t] if teach_forcing else next_input_generated

      finished_sequences |= (tgt_len <= t) | (next_input_generated == end_token_id)

      input = torch.where(finished_sequences, torch.tensor(0, device=src.device), next_input)

      if finished_sequences.all():
              break

    return outputs

  def greedy_decode(self, inputs, input_lengths, max_len):

      batch_size = inputs.size(0)
      _, hn, cn = self.encoder(inputs, input_lengths)
      max_len = max_words_targets
      input_token = torch.tensor([[sos_token_idx]] * batch_size).to(inputs.device) 
      outputs = torch.zeros(batch_size, max_len, num_words)
      generated_sentence = []
      for t in range(max_len):

          output, hn, cn = self.decoder(input_token.view(-1, 1), hn, cn)
          output = output.squeeze(1)

          predicted_token = output.argmax(1)  
          generated_sentence.append(predicted_token)
          outputs[:, t,:] = output

          input_token = predicted_token  


      return outputs, generated_sentence

inputs = inputs.apply(tokenization)

targets = targets.apply(tokenization)

inputs = inputs.tolist()

max_words_inputs = max(len(x) for x in inputs)

max_words_targets = max(len(x) for x in targets)

targets = targets.tolist()

encoder = Encoder(embedding_matrix, 512, 1)

encoder = encoder.to(device)

decoder = Decoder(embedding_matrix, 512, 1)

decoder = decoder.to(device)

model = seq2seq(encoder = encoder, decoder = decoder, teach_forcing_ratio=0.5, temperature=1.2)

model = model.to(device)

scaler = GradScaler(enabled=True)

# Creating training data

lenght_inputs = []

lenght_targets = []

for idx in range(len(inputs)):
    lenght_inputs.append(len(inputs[idx]))
    lenght_targets.append(len(targets[idx]))

    padding_length = max_words_inputs - len(inputs[idx])

    for k in range(padding_length):
         inputs[idx].append(word_to_index["<pad>"])

for idx in range(len(targets)):
    padding_length = max_words_targets - len(targets[idx])

    for k in range(padding_length):
         targets[idx].append(word_to_index["<pad>"])

for idx in range(len(inputs)):
  inputs[idx] = [inputs[idx], lenght_inputs[idx]]

for idx in range(len(targets)):
  targets[idx] = [targets[idx], lenght_targets[idx]]

x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.05, random_state=45)

train_inputs = []

train_inputs_lenghts = []

train_targets = []

train_targets_lenghts = []

validation_inputs = []

validation_targets = []

validation_inputs_lenghts = []

validation_targets_lenghts = []

for idx in range(len(x_train)):
  train_inputs.append(x_train[idx][0])
  train_targets.append(y_train[idx][0])

  train_inputs_lenghts.append(x_train[idx][1])
  train_targets_lenghts.append(y_train[idx][1])


for idx in range(len(x_test)):
  validation_inputs.append(x_test[idx][0])
  validation_targets.append(y_test[idx][0])

  validation_inputs_lenghts.append(x_test[idx][1])
  validation_targets_lenghts.append(y_test[idx][1])

save_or_load_data = False  # False: Load, True: Save 

if save_or_load_data:

    training_data = {
        "train_inputs": train_inputs,
        "train_targets": train_targets,
        "train_inputs_lenghts": train_inputs_lenghts,
        "train_targets_lenghts": train_targets_lenghts
    }

    validation_data = {
        "validation_inputs": validation_inputs,
        "validation_targets": validation_targets,
        "validation_inputs_lenghts": validation_inputs_lenghts,
        "validation_targets_lenghts": validation_targets_lenghts
    }

    with open('train_data.pkl', 'wb') as file:
        pickle.dump(training_data, file)

    with open('validation_data.pkl', 'wb') as file:
        pickle.dump(validation_data, file)

    print("Training and validation data saved successfully!")

else:

    with open('train_data.pkl', 'rb') as file:
        loaded_train_data = pickle.load(file)

    with open('validation_data.pkl', 'rb') as file:
        loaded_validation_data = pickle.load(file)

    print("Training data loaded:")
    print(loaded_train_data['train_inputs'])
    
    print("Validation data loaded:")
    print(loaded_validation_data['validation_inputs'])


validation_targets_lenghts = torch.tensor(validation_targets_lenghts)

validation_targets_lenghts = validation_targets_lenghts.to(device)

validation_inputs_lenghts = torch.tensor(validation_inputs_lenghts)

validation_inputs_lenghts = validation_inputs_lenghts.to(device)

validation_inputs = torch.tensor(validation_inputs)

validation_inputs = validation_inputs.to(device)

validation_targets = torch.tensor(validation_targets)

validation_targets = validation_targets.to(device)

# Loading the main training data

train_inputs = torch.tensor(train_inputs).long()

train_inputs = train_inputs.to(device)

train_inputs_lenghts = torch.tensor(train_inputs_lenghts)

train_inputs_lenghts = train_inputs_lenghts.to(device)

train_targets_lenghts = torch.tensor(train_targets_lenghts)

train_targets_lenghts = train_targets_lenghts.to(device)

train_targets = torch.tensor(train_targets).long()

train_targets = train_targets.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

criterion = nn.CrossEntropyLoss(ignore_index=padding_token)

print(f"Max_words_inputs = {max_words_inputs}")

print(f"Max_words_targets = {max_words_targets}")

n_epochs = 100
batch_size = 32

checkpoint_path = "model_3_freezed.pth"

checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
inital_epoch = int(checkpoint_path[6])

model.train()

for epoch in range(inital_epoch, n_epochs):

  for idx in range(0, len(train_inputs), batch_size):

      batch_inputs = train_inputs[idx: idx + batch_size]

      batch_targets = train_targets[idx: idx + batch_size]


      batch_input_lenghts = train_inputs_lenghts[idx:idx + batch_size]
      batch_target_lenghts = train_targets_lenghts[idx:idx + batch_size]


      batch_inputs = batch_inputs.to(device)

      batch_targets = batch_targets.to(device)

      optimizer.zero_grad()
      with autocast(device_type=device.type):

        predicted = model(batch_inputs, batch_input_lenghts, batch_targets, batch_target_lenghts, epoch, False)

        loss = criterion(predicted.view(-1, num_words).to(device), batch_targets.view(-1).to(device))

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        print(f"---------------------- Epoch {epoch} / Loss {loss.item()} ", end="")

        if (idx % 15 == 0):
          with torch.no_grad():
              predicted_validation, sentence = model(
                  validation_inputs, 
                  validation_inputs_lenghts, 
                  validation_targets, 
                  validation_targets_lenghts, 
                  epoch, 
                  True, 
                  max_words_targets
              )

              validation_loss = criterion(
                  predicted_validation.view(-1, num_words).to(device), 
                  validation_targets.view(-1).to(device)
              )
              print(f"/ Validation Loss: {validation_loss.item()}")

              rand_index = randint(0, len(validation_inputs) - 1)

              sample = validation_inputs[rand_index].unsqueeze(0)
              sample_len = validation_inputs_lenghts[rand_index].unsqueeze(0)

              _, output_seq = model(sample, sample_len, None, None, epoch, True, max_words_targets)

              output_seq = [token.item() for token in output_seq]

              if end_token_id in output_seq:
                  output_seq = output_seq[:output_seq.index(end_token_id)]

              response = [index_to_word[i] for i in output_seq if i in index_to_word and i != padding_token]

              print("Input:", " ".join(index_to_word[i.item()] for i in sample[0] if i.item() != padding_token))

              print("Bot:", " ".join(response))

        else:
          print("")

        if (idx == 0 and epoch != inital_epoch and epoch != 0):
          torch.save(
              {'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict()},
              f"model_{epoch}_freezed.pth")