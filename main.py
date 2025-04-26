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
from torch.utils.data import DataLoader, Dataset, TensorDataset
import gc
import torch
import pandas as pd
import os

CHUNK_SIZE = 250
TOTAL_EXAMPLES = 10200

using_tpu = False

first_train = True

word_to_index = {}
index_to_word = {}
embedding_matrix = None
sos_token_idx = None
end_token_id = None
num_words = 100000
padding_token = None
inputs = None
targets = None
max_words_inputs = None
max_words_targets = None
embedding_dim = 100
encoder = None
decoder = None
model = None

if using_tpu:
  import torch_xla
  import torch_xla.core.xla_model as xm
  import torch_xla.distributed.parallel_loader as pl
  from torch_xla.distributed import xla_multiprocessing as xmp


def load_glove_embeddings(glove_path, word_to_index, embedding_dim=100):
    embeddings_index = {}

    with open(glove_path, encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = torch.tensor([float(x) for x in values[1:]], dtype=torch.float)
            embeddings_index[word] = vector

    embedding_matrix = torch.randn(num_words, embedding_dim)

    for word, idx in word_to_index.items():
        if word in embeddings_index:
            embedding_matrix[idx] = embeddings_index[word]

    return embedding_matrix

class Encoder(nn.Module):
  def __init__(self,embedding_matrix, hidden_size, num_layers):
    super().__init__()
    self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = False, padding_idx = padding_token)
    self.lstm = nn.GRU(input_size=embedding_dim, hidden_size=hidden_size,num_layers=num_layers, batch_first=True)
    self.dropout = nn.Dropout(0.20)

  def forward(self, x, x_lenghts):
    x = self.embedding(x).float()
    x = self.dropout(x)
    x = torch.nn.utils.rnn.pack_padded_sequence(x, x_lenghts.cpu().int(), batch_first = True, enforce_sorted=False)
    hn = None
    x, hn = self.lstm(x)

    return x, hn


class Attention(nn.Module):
  def __init__(self, hidden_size):

    super().__init__()
    self.attn = nn.Linear(hidden_size  * 2, hidden_size)
    self.v = nn.Linear(hidden_size, 1, bias = False)


  def forward(self, hidden, encoder_outputs, src_len, mask = None):

    hidden = hidden.unsqueeze(1)
    hidden = hidden.repeat(1, src_len, 1)
    energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))
    attention = self.v(energy).squeeze(2)

    if mask is not None:
            attention = attention.masked_fill(mask == 0, float('-inf'))

    return torch.softmax(attention, dim = 1)


class Decoder(nn.Module):
  def __init__(self, embedding_matrix, hidden_size, num_layers):
    super().__init__()
    self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze = False, padding_idx = padding_token)
    self.lstm = nn.GRU(input_size=embedding_dim + hidden_size, hidden_size=hidden_size,num_layers=num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size * 2, num_words)

    self.atention = Attention(hidden_size)
    self.dropout = nn.Dropout(0.20)
  def forward(self, x, hn, encoder_outputs, src_len):

    mask = torch.all(encoder_outputs != 0, dim=2, keepdim=True).float()
    mask = mask.squeeze(-1)
    x = self.embedding(x).float()
    x = self.dropout(x)
    decoder_output = hn[-1]
    attn_weights = self.atention(decoder_output, encoder_outputs, src_len, mask).unsqueeze(1)
    context = torch.bmm(attn_weights, encoder_outputs)
    
    rnn_input = torch.cat((x, context), dim=2)
    x, hn  = self.lstm(rnn_input, hn)
    x = self.fc(torch.cat((x, context), dim = 2))

    return x, hn

class seq2seq(nn.Module):

  def __init__(self, encoder, decoder, teach_forcing_ratio, temperature):

    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    self.teach_forcing_ratio = teach_forcing_ratio
    self.temperature = temperature

  def forward(self, src, src_len, tgt, tgt_len, epoch, inference, device, max_len_targets = None):

    if inference:
        return self.greedy_decode(src, src_len, max_len_targets)

    self.teach_forcing_ratio = max(0.1, (0.97 ** epoch))

    max_len = tgt.shape[1]

    batch_size = src.shape[0]

    tgt_len = tgt_len.to(device)

    outputs = torch.zeros(batch_size, max_len, num_words, device = src.device)
    encoder_outputs, hn = self.encoder(src, src_len)

    padded_encoder_outputs, _= pad_packed_sequence(encoder_outputs, batch_first=True)
    src_len = padded_encoder_outputs.shape[1]

    input = tgt[:, 0]

    finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
    finished_sequences = finished_sequences.to(device)

    for t in range(1, max_len):

      output, hn = self.decoder(input.view(-1, 1), hn, padded_encoder_outputs, src_len)

      output = output.squeeze(1)

      # Batch size, seq len, input size

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
      encoder_outputs_greedy_decode, hn = self.encoder(inputs, input_lengths)

      padded_encoder_outputs_greedy_decode, _= pad_packed_sequence(encoder_outputs_greedy_decode, batch_first=True)
      src_len_greedy_decode = padded_encoder_outputs_greedy_decode.shape[1]

      input_token = torch.tensor([[sos_token_idx]] * batch_size).to(inputs.device)
      outputs = torch.zeros(batch_size, max_len, num_words, device = inputs.device)
      generated_sentence = []
      for t in range(max_len):

          output, hn= self.decoder(input_token.view(-1, 1), hn, padded_encoder_outputs_greedy_decode, src_len_greedy_decode)
          output = output.squeeze(1)

          predicted_token = output.argmax(1)
          generated_sentence.append(predicted_token)
          outputs[:, t,:] = output

          input_token = predicted_token

      return outputs, generated_sentence


device = xm.xla_device() if using_tpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_and_save_chunk(start_idx, end_idx, step, first_train=False):

  global word_to_index, index_to_word, embedding_matrix, sos_token_idx, end_token_id, num_words, padding_token
  global inputs, targets, max_words_inputs, max_words_targets, encoder, decoder, model

  data = pd.read_csv("export.csv", sep=",", quotechar='"', skipinitialspace=True)
  data = data.iloc[start_idx:end_idx]
  data = data.sample(frac=1).reset_index(drop=True)

  print(data)

  data['source_text'] = "<user> " + data['source_text'].str.lower() + " </user> "
  data['target_text'] = " <bot> " + data['target_text'].str.lower() + " </bot> "

  inputs = data["source_text"].astype(str)

  targets = data["target_text"].astype(str)

  tokens_set = " ".join(data["source_text"].astype(str) + data["target_text"].astype(str))

  pattern = r'(<bot>|</bot>|<user>|</user>|[^a-zA-Z0-9<>?]+|[?])'

  values_set = list(set(re.split(pattern, tokens_set)))

  if not first_train:

      with open("word_to_index.pkl", "rb") as file:
        word_to_index = pickle.load(file)

      for word in values_set:
        if word not in word_to_index:
          word_to_index[word] = len(word_to_index)

      index_to_word = {idx: value for value, idx in word_to_index.items()}

  else:

    word_to_index = {value: values_set.index(value) for value in values_set}

    if "<pad>" not in word_to_index:
        word_to_index["<pad>"] = len(word_to_index)

    padding_token = word_to_index['<pad>']

    print(f"Paddin token = {padding_token}")

    index_to_word = {idx: value for value, idx in word_to_index.items()}


  with open("word_to_index.pkl", "wb") as file:
    pickle.dump(word_to_index, file)

  with open("index_to_word.pkl", "wb") as file:
    pickle.dump(index_to_word, file)

  if first_train:

    print(f"word_to_index = {word_to_index}")
    print(f"index_to_word = {index_to_word}")

    embedding_matrix = load_glove_embeddings("glove.6B.100d.txt", word_to_index, embedding_dim)
    encoder = Encoder(embedding_matrix, 512, 1).to(device)
    decoder = Decoder(embedding_matrix, 512, 1).to(device)
    model = seq2seq(encoder=encoder, decoder=decoder, teach_forcing_ratio=0.5, temperature=1.0).to(device)

  sos_token_idx = word_to_index["<bot>"]

  end_token_id = word_to_index["</bot>"]


  print("-------------------num_words = " + str(num_words))

  inputs = inputs.apply(tokenization)

  targets = targets.apply(tokenization)

  inputs = inputs.tolist()

  max_words_inputs = max(len(x) for x in inputs)

  max_words_targets = max(len(x) for x in targets)

  targets = targets.tolist()


  # Create training data

  print(f"len_targets = {len(targets)}")

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


  x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.1, random_state=45)

  train_inputs = []

  train_inputs_lenghts = []

  train_targets = []

  train_targets_lenghts = []

  validation_inputs = []

  validation_targets = []

  validation_inputs_lenghts = []

  validation_targets_lenghts = []


  # Preparing the lenghts and actual tensors

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

  # Loading the validation data

  validation_targets_lenghts = torch.tensor(validation_targets_lenghts)

  validation_inputs_lenghts = torch.tensor(validation_inputs_lenghts)

  validation_inputs = torch.tensor(validation_inputs)

  validation_targets = torch.tensor(validation_targets)

  # Loading the main training data

  train_inputs = torch.tensor(train_inputs).long()

  train_inputs_lenghts = torch.tensor(train_inputs_lenghts)

  train_targets_lenghts = torch.tensor(train_targets_lenghts)

  train_targets = torch.tensor(train_targets).long()

  # Some prints

  print(f"Inputs shape = {train_inputs.shape}")

  print(f"targets_shape = {train_targets.shape}")

  print(f"Validation inputs shape = {validation_inputs.shape}")

  print(f"Max_words_inputs = {max_words_inputs}")

  print(f"Max_words_targets = {max_words_targets}")

  save_data = True

  if (save_data == True):

    torch.save(embedding_matrix, 'embedding_matrix.pt')

    torch.save(train_inputs, 'train_inputs.pt')
    torch.save(train_targets, 'train_targets.pt')
    torch.save(train_inputs_lenghts, 'train_inputs_lenghts.pt')
    torch.save(train_targets_lenghts, 'train_targets_lenghts.pt')
    torch.save(embedding_matrix, 'embedding_matrix.pt')

    torch.save(validation_inputs, 'validation_inputs.pt')
    torch.save(validation_targets, 'validation_targets.pt')
    torch.save(validation_inputs_lenghts, 'validation_inputs_lenghts.pt')
    torch.save(validation_targets_lenghts, 'validation_targets_lenghts.pt')


  train(step)

  torch.cuda.empty_cache()
  gc.collect()
  print(f"✅ Finished Chunk {start_idx}-{end_idx}\n")


def tokenization(text):

    pattern = r'(<bot>|</bot>|<user>|</user>|[^a-zA-Z0-9<>?]+|[?])'
    text = re.split(pattern, text)

    text = [word for word in text if not word.isspace()]

    text = [word for word in text if word != ""]

    words_numeric_list = [word_to_index[word] if word in word_to_index else 0 for word in text]

    return words_numeric_list



def train(step):


    global model, encoder, decoder, device, model, encoder, decoder, embedding_matrix

    train_inputs = torch.load('train_inputs.pt')
    train_targets = torch.load('train_targets.pt')
    train_inputs_lenghts = torch.load('train_inputs_lenghts.pt')
    train_targets_lenghts = torch.load('train_targets_lenghts.pt')

    validation_inputs = torch.load('validation_inputs.pt')
    validation_targets = torch.load('validation_targets.pt')
    validation_inputs_lenghts = torch.load('validation_inputs_lenghts.pt')
    validation_targets_lenghts = torch.load('validation_targets_lenghts.pt')

    train_inputs = train_inputs.to(device)
    train_targets = train_targets.to(device)
    train_inputs_lenghts = train_inputs_lenghts.to(device)
    train_targets_lenghts = train_targets_lenghts.to(device)

    validation_inputs = validation_inputs.to(device)
    validation_targets = validation_targets.to(device)
    validation_inputs_lenghts = validation_inputs_lenghts.to(device)
    validation_targets_lenghts = validation_targets_lenghts.to(device)

    embedding_matrix = embedding_matrix.to(device)
    print(f"Using: {device}")


    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=padding_token)
    scaler = GradScaler(enabled=not using_tpu)

    train_dataset = TensorDataset(train_inputs, train_targets, train_inputs_lenghts, train_targets_lenghts)
    val_dataset = TensorDataset(validation_inputs, validation_targets, validation_inputs_lenghts, validation_targets_lenghts)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True, drop_last=True)

    if using_tpu:
        train_device_loader = pl.MpDeviceLoader(train_dataloader, device)
        val_device_loader = pl.MpDeviceLoader(val_dataloader, device)
    else:
        train_device_loader = train_dataloader
        val_device_loader = val_dataloader

    n_epochs = 5
    model.train()
    inital_epoch = 0

    for epoch in range(inital_epoch, n_epochs):
        idx = 0
        for batch in train_device_loader:
            batch_inputs, batch_targets, batch_input_lenghts, batch_target_lenghts = batch

            optimizer.zero_grad()
            predicted = model(batch_inputs, batch_input_lenghts, batch_targets, batch_target_lenghts, epoch, False, device)

            loss = criterion(predicted.view(-1, num_words), batch_targets.view(-1))
            loss.backward()

            if using_tpu:
                xm.optimizer_step(optimizer)
                xm.mark_step()
            else:
                optimizer.step()

            print(f"Epoch {epoch} / Loss {loss}", end="")

            if idx % 15 == 0:
                with torch.no_grad():
                    predicted_validation, sentence = model(
                        validation_inputs, validation_inputs_lenghts,
                        validation_targets, validation_targets_lenghts,
                        epoch, True, device, max_words_targets
                    )

                    validation_loss = criterion(
                        predicted_validation.view(-1, num_words),
                        validation_targets.view(-1)
                    )
                    print(f"/ Validation Loss: {validation_loss}")

                    sample_idx = randint(0, len(validation_inputs) - 1)
                    sample = validation_inputs[sample_idx].unsqueeze(0).to(device)
                    sample_len = validation_inputs_lenghts[sample_idx].unsqueeze(0)

                    user_input_tokens = validation_inputs[sample_idx][:sample_len].tolist()
                    user_input_text = [index_to_word[i] for i in user_input_tokens if i != padding_token]
                    print("User:", " ".join(user_input_text))

                    _, output_seq = model(sample, sample_len, None, None, epoch, True, device, max_words_targets)
                    output_seq = [token for token in output_seq]

                    if end_token_id in output_seq:
                        output_seq = output_seq[:output_seq.index(end_token_id)]
                    if using_tpu:
                      pass

                    else:
                      response = [index_to_word[i.item()] for i in output_seq if i.item() != padding_token]
                    print("Bot:", " ".join(response))
            else:
                print("")

            idx += 1

        print("================================= Model Saved")
        torch.save(
            {'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict()},
            f"model_{step}_{epoch}_.pth"
        )



def main_chunk_loop():
    for i in range(0, TOTAL_EXAMPLES, CHUNK_SIZE):
        first = (i == 0)
        process_and_save_chunk(i, i + CHUNK_SIZE, i, first_train=first)


if __name__ == "__main__":
    if using_tpu:
        xmp.spawn(main_chunk_loop, start_method='fork')
    else:
        main_chunk_loop()
