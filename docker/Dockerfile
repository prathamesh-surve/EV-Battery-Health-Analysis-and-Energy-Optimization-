FROM apache/airflow:2.7.1-python3.11

USER root
RUN apt-get update && apt-get install -y git build-essential && apt-get clean

USER airflow
# 1. Copy your local requirements file into the container image
COPY requirements.txt /requirements.txt

# 2. Force install the compatible NumPy 1.x library BEFORE processing your other requirements
RUN pip install --no-cache-dir "numpy<2.0.0"

# 3. Install the rest of your engineering packages safely
RUN pip install --no-cache-dir -r /requirements.txt