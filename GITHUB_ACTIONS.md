# GitHub Actions CI/CD Setup

This repository includes GitHub Actions workflows for automatically building and publishing Docker images to GitHub Container Registry.

## Workflows

### 1. Build and Publish (`build-image.yml`)

**Triggers:**
- Push to `master` branch
- Push of version tags (e.g., `v1.0.0`)
- Manual trigger via GitHub UI

**What it does:**
- Builds Docker image for multiple architectures (AMD64, ARM64)
- Publishes to `ghcr.io/vishal-pandey/livekit-voice-agent`
- Tags with both `latest` and commit SHA/version

### 2. Deploy (`deploy.yml`)

**Triggers:**
- Manual trigger via GitHub UI
- Allows selecting environment and image tag

**What it does:**
- Currently a template for deployment steps
- Can be extended to deploy to your servers

## Published Images

Your Docker images will be available at:
```
ghcr.io/vishal-pandey/livekit-voice-agent:latest
ghcr.io/vishal-pandey/livekit-voice-agent:<commit-sha>
ghcr.io/vishal-pandey/livekit-voice-agent:<tag-version>
```

## Using the Published Image

### Development (local build)
```bash
docker-compose up --build
```

### Production (using published image)
```bash
# Copy your environment variables
cp .env.example .env
# Edit .env with your credentials

# Use the production compose file
docker-compose -f docker-compose.prod.yml up -d
```

### Pull the latest image manually
```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Pull the image
docker pull ghcr.io/vishal-pandey/livekit-voice-agent:latest
```

## Setting up GitHub Actions

1. **No additional setup required** - The workflows use `GITHUB_TOKEN` which is automatically provided.

2. **Repository Settings:**
   - Go to your repository → Settings → Actions → General
   - Ensure "Allow GitHub Actions to create and approve pull requests" is enabled
   - Under "Workflow permissions", ensure "Read and write permissions" is selected

3. **Package Visibility:**
   - After first build, go to your GitHub profile → Packages
   - Find `livekit-voice-agent` package
   - Click on it and change visibility to "Public" if desired

## Deployment Options

### Option 1: Manual Deployment
1. SSH to your server
2. Pull the latest image: `docker pull ghcr.io/vishal-pandey/livekit-voice-agent:latest`
3. Update your docker-compose.prod.yml
4. Restart services: `docker-compose -f docker-compose.prod.yml up -d`

### Option 2: Automated Deployment
Extend the `deploy.yml` workflow with:
- SSH actions to connect to your server
- Commands to pull new image and restart services
- Notification systems (Slack, Discord, etc.)

### Option 3: Container Orchestration
Use the published image with:
- Kubernetes deployments
- Docker Swarm
- Nomad
- AWS ECS/Fargate
- Google Cloud Run

## Monitoring Builds

- View build status: Repository → Actions tab
- Download build logs and artifacts
- See published packages: Your Profile → Packages

## Version Management

### Create a new release:
```bash
git tag v1.0.0
git push origin v1.0.0
```

This will trigger a build with the version tag `v1.0.0`.

## Environment Variables for Production

Ensure these are set in your production environment:
- All API keys (OpenAI, Google AI, Sarvam, etc.)
- LiveKit server configuration
- AWS S3 credentials
- MongoDB credentials

Never commit these to the repository - use `.env` files or environment variable injection.
