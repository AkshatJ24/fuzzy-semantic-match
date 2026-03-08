FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-build the corpus and cluster model at image build time so the container
# starts instantly. Trade: larger image, longer build. Acceptable for this use case.
RUN python scripts/prepare_data.py && python scripts/run_clustering.py

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
