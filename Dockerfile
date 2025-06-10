FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 streamlit && chown -R streamlit:streamlit /app

# Ensure assets directory and files have proper permissions
RUN chmod -R 755 /app/assets && \
    chown -R streamlit:streamlit /app/assets

USER streamlit

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "prompt_engineering_workshop_app.py", "--server.port=8501", "--server.address=0.0.0.0"] 