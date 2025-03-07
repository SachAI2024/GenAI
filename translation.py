from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings

warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#Define the tokenizer and the model
model_name = 'yasmineelabbar/marian-finetuned-kde4-en-to-fr-accelerate'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

#Define the input text to be translated

text = "I am a software engineer working like a donkey to make a living"
print(f'Input text is: {text}')
print('_'*120)


#Translate English to French
#Tokenizer the input text

imputs = tokenizer.encode(text,return_tensors="pt")
#Generate the embeddings by passing the tokenizer input text to the model
outputs = model.generate(imputs,max_length=150,num_beams=4,early_stopping=True)

#Decode the output using the tokenizer 
translated_output = tokenizer.decode(outputs[0],skip_special_tokens=True)

print(f'Tranlated text in French is: {translated_output}')
print('-'*20)