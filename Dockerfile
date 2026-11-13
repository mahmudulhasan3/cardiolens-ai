FROM python:3.12.3

WORKDIR /code

# Install CPU-only PyTorch first (much smaller than the CUDA build)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code and model files
COPY app ./app
COPY src ./src
COPY models ./models

# The app may write feedback here, but outputs/ is ignored
RUN mkdir -p outputs

# Non-secret default only. The API key is passed at run time
ENV APP_ENV=production

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]