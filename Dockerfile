FROM python:3.12-slim

WORKDIR /code

# Install CPU-only PyTorch first (much smaller than the CUDA build)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Dependencies change rarely, so copy them first to keep this layer cached
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Model files change rarely, so copy them before the code
COPY models ./models

# Code changes often, so copy it last
COPY src ./src
COPY app ./app

# The app writes feedback here
RUN mkdir -p outputs

# Non-secret default only. The API key is passed at run time
ENV APP_ENV=production

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]