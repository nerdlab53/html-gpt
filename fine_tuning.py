#importing utilities
import os
import time
import pandas as pd
import seaborn as sns
import numpy as np
import random
import matplotlib.pyplot as plt
from datasets import load_dataset
import argparse
# %matplotlib inline
import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument("--hf-token", type=str, required=True)
parser.add_argument("--hf-repo", type=str, required=True)
args = parser.parse_args()

token = args.hf_token
new_model = args.hf_repo


# defining a dataset class to create a GPT2 like dataset for e.g adding necessary input_ids and attention masks
class CreateGPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=768):

    self.tokenizer = tokenizer          # setting the tokenizer
    self.input_ids = []                 # creating inputmasks
    self.attn_masks = []                # attention masks
    for txt in txt_list:
      encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx]

# importing the tokenizer for using on the dataset
print("Loading Tokenizer ..")
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium

# loading and creating the dataset for training and evaluation, I have already preprocessed the dataset and uploaded it to huggingface
from datasets import load_dataset
print("Loading dataset ..")
dataset = load_dataset('retr0sushi04/html_pre_processed', split='train')
dataset = CreateGPT2Dataset(dataset['text'], tokenizer)

# Split into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{} training samples'.format(train_size))
print('{} validation samples'.format(val_size))

# setting batch size
batch_size = 2
print("Important training info : ")
# creating pytorch dataloaders for the training and validation datasets
print("Creating PyTorch dataloader ..")
train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size=batch_size)
validation_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size=batch_size)

# loading the GPT2 model for fine-tuning
config = GPT2Config.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
model.resize_token_embeddings(len(tokenizer))
device = torch.device("cuda")
model.cuda() #putting the model on GPU

# setting up the parameters for the model
print("Printing important info : \n Epochs : {:}\n Learining Rate : {:}, Optimizer : AdamW\n")
epochs = 20 ; learning_rate = 2e-4 ; warmup_steps = 1e2 ; epsilon = 1e-8
# setting up the optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=epsilon)
# defining number of steps
total_steps = len(train_dataloader) * epochs
# setting up the scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

import datetime
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

import time

# training loop
model = model.to(device)
total_time = time.time()
training_stats = []
for epoch_i in range(0, epochs):
  print("\nTraining ....")
  print('Epoch {:} / {:}'.format(epoch_i + 1, epochs))
  t0 = time.time()
  total_train_loss = 0
  model.train()
  for step, batch in enumerate(train_dataloader):
    batch_input_ids = batch[0].to(device)
    batch_labels = batch[0].to(device)
    batch_masks = batch[1].to(device)
    model.zero_grad()
    outputs = model(input_ids=batch_input_ids, attention_mask=batch_masks, labels=batch_labels, token_type_ids=None)
    loss = outputs[0]
    batch_loss = loss.item()
    total_train_loss += batch_loss
    if step % 100 == 0 and not step == 0:
      elapsed_time = format_time(time.time() - time)
      print(' Batch {:} of {:}. Loss : {:}.  Elapsed : {:}'.format(step, len(train_dataloader)), batch_loss, elapsed)
      model.eval()
      sample_outputs = model.generate(
          bos_token_id=random.randint(1,30000),
          do_sample=True,
          top_k=50,
          max_length=200,
          top_p=0.95,
          num_return_sequences=1
      )
      for i, sample_output in enumerate(sample_outputs):
          print("{}: {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
      model.train()
    loss.backward()
    optimizer.step()
    scheduler.step()
  avg_train_loss = total_train_loss / len(train_dataloader)
  training_time = format_time(time.time() - t0)
  print("Time taken by epoch : {:}".format(training_time))
  print("Average training loss : {:}".format(avg_train_loss))
  # validation
  print("")
  print("Running validation ....")
  t0 = time.time()
  model.eval()
  total_eval_loss = 0
  no_eval_steps = 0
  for batch in validation_dataloader:
    batch_input_ids = batch[0].to(device)
    batch_labels = batch[0].to(device)
    batch_masks = batch[1].to(device)
    with torch.no_grad():
      outputs = model(input_ids=batch_input_ids, attention_mask=batch_masks, labels=batch_labels, token_type_ids=None)
      loss = outputs[0]
    total_eval_loss += loss.item()
    no_eval_steps += 1
  avg_eval_loss = total_eval_loss / no_eval_steps
  validation_time = format_time(time.time() - t0)
  print("Validation loss : {:}".format(avg_eval_loss))
  print("Validation took : {:}".format(validation_time))
  training_stats.append(
      {
          'epoch': epoch_i + 1,
          'Training Loss': avg_train_loss,
          'Valid. Loss': avg_eval_loss,
          'Training Time': training_time,
          'Validation Time': validation_time
      }
  )

print("")
print("Training complete !")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_time)))

# plotting the run
stats = pd.DataFrame(data = training_stats)
stats.set_index('epoch')
# Use plot styling from seaborn.
sns.set(style='darkgrid')
# Increase the plot size and font size.
# Plot the learning curve.
plt.plot(stats['Training Loss'], 'b-o', label="Training")
plt.plot(stats['Valid. Loss'], 'g-o', label="Validation")
# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])
plt.show()


output_dir = './model_save/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Saving model to %s" % output_dir)
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Pushing model to hub..")
model_to_save.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)
print("Model pushed to hub")
# infer model
# model.eval()
# prompt = "<s> [INST]Create a landing page [/INST]"
# generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
# generated = generated.to(device)
# print(generated)
# sample_outputs = model.generate(
#                                 generated,
#                                 #bos_token_id=random.randint(1,30000),
#                                 do_sample=True,
#                                 top_k=50,
#                                 max_length = 300,
#                                 top_p=0.95,
#                                 num_return_sequences=3
#                                 )

# for i, sample_output in enumerate(sample_outputs):
#   print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))