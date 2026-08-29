FROM python:3.10

# Create app directory and copy project
WORKDIR /app
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 7860

# Run the Flask app (app.py is at repo root)
CMD ["python", "app.py"]
