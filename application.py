from flask import Flask,request, render_template,jsonify

import pickle
import pandas as pd

import torch
from transformers import AutoTokenizer,AutoModelWithLMHead

checkpoint="t5-small"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
summ_model= AutoModelWithLMHead.from_pretrained(checkpoint,return_dict=True)







application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET',"POST"])
def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    

    else:
        data = request.form.get('text')
        new_data=pd.Series(data=data)

        inputs=tokenizer.encode("summarize "+data,return_tensors='pt',max_length=512,truncation=True,padding=True)

        output=summ_model.generate(inputs,min_length=75,max_length=100)


        summary=tokenizer.decode(output[0],skip_special_tokens=True)

       
         
        
        

        with open('classifier.pkl','rb') as file_obj:
          model=pickle.load(file_obj)
        
        result=model.predict_proba(new_data)

    
            
        

        return render_template('form.html',text=data,summary=summary,positive=result[:][0][1],negative=result[:][0][0])
    



if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)