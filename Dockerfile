FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY project_6.py .

EXPOSE 8501

CMD ["streamlit", "run", "project_6.py", "--server.address=0.0.0.0", "--server.port=8501"]
