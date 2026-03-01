# Use Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the ML experiment
CMD ["python", "palmer-panguin-decisionTree.py"]