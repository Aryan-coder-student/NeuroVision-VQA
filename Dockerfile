FROM python:3.10 
WORKDIR /app
COPY requirements.txt . 
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000 8501
CMD ["sh", "-c", "python Deployment/app.py & while ! nc -z localhost 5000; do sleep 1; done; streamlit run Deployment/streamlit/main.py --server.port=8501 --server.address=0.0.0.0"]