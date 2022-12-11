import pandas as pd
from rouge_score import rouge_scorer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from logging_.logger import logging

model_trained = AutoModelForSeq2SeqLM.from_pretrained('/model-t5-base/checkpoint-6500').to('cpu')
token_trained = AutoTokenizer.from_pretrained('/model-t5-base/checkpoint-6500')
submission = pd.read_csv('train.csv')
submission=submission.loc[:20,:]
def evaluate(model_output,actual_titles):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = list()
    for output,actual in zip(model_output,actual_titles):
        s = scorer.score(output,actual)
        scores.append(s['rouge1'].fmeasure)
    print('Evaluation result',np.mean(scores))
    return scores

def predict(texts,temperature=0.6,num_beams = 4,max_gen_length = 256):
    # write code to output a list of title for each text input to the predict method
    data=texts.tolist()
    inputs = token_trained(data, max_length=512, return_tensors='pt',truncation=True, padding=True)
    title_ids = model_trained.generate(
    inputs['input_ids'].to('cpu'), 
    num_beams=num_beams, 
    temperature=temperature, 
    max_length=max_gen_length, 
    early_stopping=True)
    pred_title=[]
    for i in range(len(data)):
      title = token_trained.decode(title_ids[i], skip_special_tokens=True, clean_up_tokenization_spaces=False)
      pred_title.append(title)
    #check accuracy
    rouge_score=evaluate(pred_title,submission['title'].tolist())
    logging.info(f"Rouge1 Score is :{rouge_score}")
    print(rouge_score)
    return pred_title

def test_model():
    pred = predict(submission["text"])
    submission['predicted_title'] = pred
    submission.to_csv('submission_top_20.csv',index=False)




if __name__=="__main__":
    #write model loading code here
    test_model()