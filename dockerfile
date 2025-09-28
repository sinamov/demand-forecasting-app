# /Dockerfile

# 1. Use an official, lightweight Python image as the base.
FROM python:3.9-slim

# --- FIX 2: Install system dependencies needed for headless Chrome ---
# This layer installs the graphics and rendering libraries that Chrome depends on.
# The `rm -rf` command is a best practice to keep the final image size smaller.
RUN apt-get update && apt-get install -y \
    libnss3 \
    libatk-bridge2.0-0 \
    libcups2 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
 && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory inside the container.
WORKDIR /app

# 3. Copy the requirements file and install Python packages.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- FIX 1: Install the headless Chrome browser needed by Plotly/Kaleido ---
RUN plotly_get_chrome

# 4. Copy the rest of your project files into the working directory.
COPY . .

# 5. Expose port 8081.
EXPOSE 8081

# 6. Define the command to run when the container starts.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8081"]