FROM python:3.12-slim-bookworm

# Update package lists and install required packages including Node.js and npm
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    nodejs \
    npm

# Ensure npx is installed globally (npm versions from 5.2.0 onward include npx, but this guarantees it)
RUN npm install -g npx

# Download the latest uv installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the PATH
ENV PATH="/root/.local/bin/:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY app.py /

# Run the application with uvicorn (via uv)
CMD ["uv", "run", "app.py"]
