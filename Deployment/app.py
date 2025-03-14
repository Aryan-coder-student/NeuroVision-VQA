import torch
from flask import Flask, request, jsonify
from transformers import BlipProcessor, BlipForQuestionAnswering
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper, PubMedAPIWrapper
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
from PIL import Image
from flask_cors import CORS

 
load_dotenv()
os.getenv("SERPAPI_API_KEY")
os.getenv("GROQ_API_KEY")

 
app = Flask(__name__)
CORS(app)

 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
fine_tuned_model = BlipForQuestionAnswering.from_pretrained(os.path.join("Deployment/Model/14-last-blip-saved-model")).to(device)
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
 
memory = ConversationBufferMemory()
def chat_bot_query(query: str):
    search_tool = Tool(
        name="Medical_Web_Search",
        func=SerpAPIWrapper().run,
        description="Searches the web for medical information related to brain, CT, and MRI scans."
    )

    pubmed_tool = Tool(
        name="PubMed_Search",
        func=PubMedAPIWrapper().run,
        description="Searches PubMed for research papers related to brain, CT, and MRI scans."
    )

    agent = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=[search_tool, pubmed_tool],
        llm=ChatGroq(model="gemma2-9b-it"),
        memory=memory,
        verbose=False,
        max_iterations=10,
        handle_parsing_errors=True
    )

    response = agent.run(query)
    return response


 
def predict_answer(image, question):
    try:
        image = image.convert('RGB')
        inputs = processor(image, question, return_tensors="pt").to(device)
        fine_tuned_output = fine_tuned_model.generate(**inputs)
        fine_tuned_answer = processor.tokenizer.decode(fine_tuned_output[0], skip_special_tokens=True)
        return fine_tuned_answer
    except Exception as e:
        return f"Error in prediction: {e}"

 
@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    try:
        file = request.files['file']
        question = request.form['question']
        if not file or not question:
            return jsonify({'error': 'No file or question provided'}), 400
        image = Image.open(file)
        answer = predict_answer(image, question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
@app.route('/chat/', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('query')
        if not query:
            return jsonify({'error': 'No query provided'}), 400

        response = chat_bot_query(query)
        return jsonify({'response': response})
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

 
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port='5000')