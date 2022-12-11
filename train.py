import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from datasets import load_dataset,load_from_disk,load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from logging_.logger import logging

from google.colab import drive
drive.mount('/content/drive')

def data_preparation():
    full_dataset = load_dataset("csv", data_files='/content/drive/MyDrive/title_generation_final/train.csv')
    cols_to_remove = list(full_dataset['train'].features.keys())
    cols_to_remove.remove("title")
    cols_to_remove.remove("text")
    data=full_dataset.remove_columns(cols_to_remove)
    dataset = data['train'].train_test_split(test_size=0.1)
    test_val = dataset['test'].train_test_split(test_size=0.5)
    dataset['val'] = test_val['train']
    dataset['test'] = test_val['test']
    dataset.save_to_disk('/content/drive/MyDrive/title_generation_final/title_generation_dataset')
    dataset = load_from_disk('/content/drive/MyDrive/title_generation_final/title_generation_dataset')
    nltk.download('punkt')
    return dataset

def tokenizer_():
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    return tokenizer

def preprocess_data(data,MAX_SOURCE_LEN = 512,MAX_TARGET_LEN = 128):    
    # Initialize T5-base tokenizer
    tokenizer=tokenizer_()
    model_inputs = tokenizer(data['text'], max_length=MAX_SOURCE_LEN, padding=True, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(data['title'], max_length=MAX_TARGET_LEN, padding=True, truncation=True)
    # Replace all pad token ids in the labels by -100 to ignore padding in the loss
    labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    model_inputs['labels'] = labels["input_ids"]
    return model_inputs

def run_process_to_data_preprocess():
    data=data_preparation()
    processed_dataset = data.map(
        preprocess_data,
        batched=True,
        remove_columns=['text', 'title'],
        desc="Running tokenizer on dataset",)
    return processed_dataset

batch_size = 4
num_epochs = 2
learning_rate = 5.6e-5
weight_decay = 0.01
log_every = 50
eval_every = 1000
lr_scheduler_type = "linear"

# Define training arguments
def training_params():
  training_args = Seq2SeqTrainingArguments(
      output_dir="/content/drive/MyDrive/title_generation_final/model-t5-base",
      evaluation_strategy="steps",
      eval_steps=eval_every,
      learning_rate=learning_rate,
      per_device_train_batch_size=batch_size,
      per_device_eval_batch_size=batch_size,
      weight_decay=weight_decay,
      save_steps=6500,
      save_total_limit=3,
      num_train_epochs=num_epochs,
      predict_with_generate=True,
      logging_steps=log_every,
      group_by_length=True,
      lr_scheduler_type=lr_scheduler_type,
      report_to="wandb",
      resume_from_checkpoint=True,
  )
  return training_args

def model_():
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')

    return model

def train_initialize(eval_pred):
  # Initialize T5-base model
    metric = load_metric("rouge")
    predictions, labels = eval_pred
    tokenizer=tokenizer_()
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    
    # Compute ROUGE scores and get the median scores
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return {k: round(v, 4) for k, v in result.items()}

# Dynamic padding in batch using a data collator
def train():
    logging.info("Training started")
    logging.info("Loading data")
    data_preparation()
    logging.info("Data loaded")
    print("PROCESSED DATASET RUNNING")
    logging.info("Running process to data preprocess")
    processed_dataset=run_process_to_data_preprocess()
    logging.info("Process to data preprocess completed")
    print("TOKENIZER RUNNING")
    logging.info("Running tokenizer")
    tokenizer=tokenizer_()
    logging.info("Tokenizer completed")
    print("MODEL RUNNING")
    logging.info("Running model")
    model=model_()
    print("TRAINNG ARGS RUNNING")
    logging.info("Running training args")
    training_args=training_params()
    logging.info("Training args completed")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    # Define the trainer
    logging.info("Actual Training started....:)")
    trainer = Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["val"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=train_initialize,)
    trainer.train()
    return trainer

if __name__ == "__main__":
    print('Running training script')
    train()