# Use an official lightweight Python image as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependency file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the proxy script into the container
COPY openai_proxy.py .

# Create a non-root user to run the application for better security
RUN useradd --create-home appuser
USER appuser

# Expose the port the server will run on
EXPOSE 8888

# The command to run when the container starts
CMD ["uvicorn", "openai_proxy:app", "--host", "0.0.0.0", "--port", "8888"]
