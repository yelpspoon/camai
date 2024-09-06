# Step 1: Use the official Python 3.12 slim image
FROM python:3.12-slim

# Step 2: Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libwebp-dev \
    libtiff-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    ca-certificates \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Install pip and set up Python environment
RUN pip install --upgrade pip
COPY --from=ghcr.io/astral-sh/uv:0.4.2 /uv /bin/uv

# Step 4: Install Python dependencies
#RUN mkdir /app
RUN git clone https://github.com/THU-MIG/yolov10.git /app
COPY requirements.txt /app
WORKDIR /app
RUN pip install -r requirements.txt 
#RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN curl -L -o yolov10n.pt "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt"

# Step 5: Copy application code
COPY app.py ./

# Step 6: Expose the port Streamlit will run on
EXPOSE 8501

# Step 7: Command to run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
