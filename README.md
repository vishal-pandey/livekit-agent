# LiveKit Voice Agent - Docker Deployment

This repository contains a Hindi telemedical examination voice agent built with LiveKit, integrated with various AI services and MongoDB for data persistence.

[![Build and Publish Docker Image](https://github.com/vishal-pandey/livekit-agent/actions/workflows/build-image.yml/badge.svg)](https://github.com/vishal-pandey/livekit-agent/actions/workflows/build-image.yml)

## 🐳 Docker Images

Pre-built Docker images are available on GitHub Container Registry:

```bash
# Latest version
docker pull ghcr.io/vishal-pandey/livekit-voice-agent:latest

# Specific version
docker pull ghcr.io/vishal-pandey/livekit-voice-agent:v1.0.0
```

## Prerequisites

- Docker and Docker Compose installed
- Required API keys and credentials (see Environment Variables section)

## Quick Start

### Option 1: Using Pre-built Image (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/vishal-pandey/livekit-agent.git
   cd livekit-agent
   ```

2. **Configure your environment**
   ```bash
   cp .env.example .env
   # Edit .env with your actual credentials
   ```

3. **Run with pre-built image**
   ```bash
   docker-compose -f docker-compose.prod.yml up -d
   ```

### Option 2: Build Locally

1. **Clone the repository and navigate to the project directory**
   ```bash
   git clone https://github.com/vishal-pandey/livekit-agent.git
   cd livekit-agent
   ```

2. **Copy the environment template and configure your credentials**
   ```bash
   cp .env.example .env
   # Edit .env with your actual credentials
   ```

3. **Build and run locally**
   ```bash
   docker-compose up --build
   ```

## Environment Variables

Create a `.env` file based on `.env.example` with the following variables:

### Core AI Service Configuration
- `OPENAI_API_KEY`: OpenAI API key (for LLM processing)
- `SARVAM_API_KEY`: Sarvam AI API key (for Hindi STT/TTS)
- `GOOGLE_API_KEY`: Google Generative AI API key (for medical report generation)

### LiveKit Configuration
- `LIVEKIT_URL`: Your LiveKit server URL (e.g., http://192.168.1.9:7880)
- `LIVEKIT_API_KEY`: LiveKit API key
- `LIVEKIT_API_SECRET`: LiveKit API secret

### MongoDB Configuration
- `MONGO_DB_PASSWORD`: MongoDB password for the connection string

### AWS S3 Configuration (for Egress Recording)
- `AWS_ACCESS_KEY_ID`: AWS access key ID
- `AWS_SECRET_ACCESS_KEY`: AWS secret access key
- `AWS_REGION`: AWS region (default: ap-south-1)
- `AWS_S3_BUCKET`: S3 bucket name for storing recordings

### Optional API Keys
- `DEEPGRAM_API_KEY`: Deepgram API key (if using Deepgram STT)
- `CARTESIA_API_KEY`: Cartesia API key (if using Cartesia TTS)

## Docker Commands

### Build the image
```bash
docker build -t lumiq-voice-agent .
```

### Run with Docker Compose (recommended)
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Run with Docker directly
```bash
docker run -d \
  --name lumiq-voice-agent \
  --env-file .env \
  -v $(pwd)/recordings:/app/recordings \
  -v $(pwd)/logs:/app/logs \
  lumiq-voice-agent
```

## File Structure

```
.
├── agent.py                 # Main application
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker image configuration
├── docker-compose.yml      # Docker Compose configuration
├── .dockerignore           # Docker build exclusions
├── .env.example            # Environment template
└── recordings/             # Generated recordings and transcripts
    ├── transcript_*.json
    ├── medical_report_*.json
    ├── enhanced_summary_*.txt
    └── session_*.wav
```

## Features

- **Hindi Voice Processing**: Uses Sarvam AI for Hindi STT/TTS
- **Multi-language Support**: Optional Deepgram and Cartesia integration
- **Medical Question Flow**: Structured telemedical examination questionnaire  
- **Volume-based VAD**: Filters audio to focus on primary speaker
- **Multiple Storage Options**: 
  - Local WAV recordings (fallback)
  - AWS S3 via LiveKit Egress (primary)
  - MongoDB for transcripts and medical reports
- **AI-powered Analysis**: 
  - OpenAI GPT for conversation processing
  - Google Gemini for medical report generation
- **Local LiveKit Server Support**: Compatible with self-hosted LiveKit instances

## Monitoring and Logs

### View application logs
```bash
docker-compose logs -f livekit-voice-agent
```

### Health check
```bash
docker-compose ps
```

### Access recordings
Recordings are stored in the `./recordings` directory and are automatically mounted to the container.

## Troubleshooting

### Common Issues

1. **Audio dependencies missing**
   - The Dockerfile installs necessary audio libraries (portaudio, alsa)

2. **Permission issues with recordings**
   - Ensure the recordings directory is writable
   - Check Docker volume permissions

3. **API connection failures**
   - Verify all API keys in `.env` file
   - Check network connectivity to external services

4. **MongoDB connection issues**
   - Verify MongoDB password and connection string
   - Ensure MongoDB cluster allows connections from your IP

### Debug mode
To run with more verbose logging:
```bash
docker-compose up --build
# Then check logs for detailed information
```

## Production Considerations

1. **Resource Limits**: Adjust memory and CPU limits in docker-compose.yml
2. **Security**: Use Docker secrets for sensitive credentials
3. **Monitoring**: Add monitoring solutions (Prometheus, etc.)
4. **Scaling**: Consider using Docker Swarm or Kubernetes for scaling
5. **Backup**: Ensure MongoDB and S3 data are backed up regularly

## Development

For development purposes, you can mount the source code:
```yaml
volumes:
  - .:/app
  - ./recordings:/app/recordings
```

This allows live code changes without rebuilding the container.
