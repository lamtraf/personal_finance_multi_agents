FROM python:3.12-bookworm

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python packages during build phase
RUN pip install -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000

# Set default command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
