# 🚀 Digital Ocean Production Deployment Guide

This guide covers deploying the Star Wars RAG Dashboard to Digital Ocean for production use.

## 📋 Prerequisites

### Digital Ocean Requirements
- **Droplet**: Minimum 4GB RAM, 2 CPU cores, 80GB SSD
- **Operating System**: Ubuntu 20.04 or 22.04 LTS
- **Domain**: Optional, for custom domain and HTTPS

### Local Requirements
- Docker & Docker Compose
- Git
- SSH access to your droplet

## 🏗️ Architecture Overview

### Container Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Digital Ocean Droplet                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐  │
│  │   Dashboard     │  │   API Backend   │  │ PostgreSQL  │  │
│  │   (Port 80)     │  │   (Port 8002)   │  │ + pgvector  │  │
│  │                 │  │                 │  │             │  │
│  │ • Streamlit     │  │ • FastAPI       │  │ • Dialogue  │  │
│  │ • Plotly Charts │  │ • RAG Pipeline  │  │ • Embeddings│  │
│  │ • Dark Theme    │  │ • LLM + Models  │  │ • Sessions  │  │
│  └─────────────────┘  └─────────────────┘  └─────────────┘  │
├─────────────────────────────────────────────────────────────┤
│            Docker Network: starwars-network                 │
└─────────────────────────────────────────────────────────────┘
```

### Yes, the web app is in ONE container ✅
- **Dashboard Container**: Streamlit + UI (lightweight, ~1GB)
- **API Container**: FastAPI + RAG + LLM (heavy, ~4GB)
- **Database Container**: PostgreSQL + pgvector

### Pipeline Usage ✅
The dashboard **DOES use the full RAG pipeline**:
1. **User Query** → Dashboard (Streamlit)
2. **API Request** → Backend via HTTP
3. **Embedding** → sentence-transformers 
4. **Vector Search** → pgvector database
5. **Context Retrieval** → 6 dialogue lines
6. **LLM Generation** → Local LLM (phi-2)
7. **Response** → Back to dashboard

## 🚀 Quick Deployment

### 1. Prepare Your Droplet
```bash
# SSH into your Digital Ocean droplet
ssh root@your-droplet-ip

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose-plugin -y
```

### 2. Clone and Configure
```bash
# Clone repository
git clone https://github.com/your-username/star-wars-chat-app.git
cd star-wars-chat-app

# Set up environment
cp env.example .env
nano .env  # Edit with your production values
```

### 3. Deploy with One Command
```bash
# Make deployment script executable
chmod +x deploy_digital_ocean.py

# Deploy everything
python3 deploy_digital_ocean.py --firewall
```

## ⚙️ Production Configuration

### Environment Variables (.env)
```bash
# Database
POSTGRES_DB=star_wars_rag
POSTGRES_USER=starwars_admin
POSTGRES_PASSWORD=your_secure_password_123

# API Configuration  
API_BASE_URL=http://star-wars-api:8000  # Internal Docker network
LOG_LEVEL=info

# Dashboard
STREAMLIT_THEME_BASE=dark
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Optional: External Storage
DO_SPACES_KEY=your_do_spaces_key
DO_SPACES_SECRET=your_do_spaces_secret
```

### Resource Limits
```yaml
# Production containers have resource limits:
api:
  memory: 4GB (limit) / 2GB (reservation)
  cpu: 2.0 cores (limit) / 1.0 cores (reservation)

dashboard:
  memory: 1GB (limit) / 512MB (reservation) 
  cpu: 1.0 cores (limit) / 0.5 cores (reservation)

postgres:
  memory: 2GB (limit) / 1GB (reservation)
  cpu: 1.0 cores (limit) / 0.5 cores (reservation)
```

## 🔐 Security Best Practices ✅

### Container Security
- **Non-root user**: Dashboard runs as user 1001
- **Resource limits**: Prevents container resource exhaustion
- **Health checks**: Automatic restart on failure
- **Network isolation**: Internal Docker network

### Firewall Configuration
```bash
# Configured automatically with --firewall flag
ufw allow ssh      # SSH access
ufw allow 80/tcp   # HTTP dashboard
ufw allow 443/tcp  # HTTPS (optional)
ufw allow 8002/tcp # API access
ufw enable
```

### Database Security
- Strong passwords required
- Internal network communication
- Persistent volumes for data

## 📊 Monitoring & Maintenance

### Health Checks
```bash
# Check all services
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f

# Monitor resources
docker stats
```

### Access URLs
- **Dashboard**: `http://your-droplet-ip/`
- **API**: `http://your-droplet-ip:8002/`
- **API Docs**: `http://your-droplet-ip:8002/docs`
- **Health**: `http://your-droplet-ip:8002/health`

### Scaling
```bash
# Scale dashboard for high traffic
docker-compose -f docker-compose.production.yml up -d --scale star-wars-dashboard=3

# Update deployment
python3 deploy_digital_ocean.py --update
```

## 🌐 Domain & HTTPS Setup

### 1. Configure Domain
```bash
# Point your domain A record to droplet IP
# Example: starwars-rag.yourdomain.com → 142.93.xxx.xxx
```

### 2. Enable HTTPS
```bash
# Install Certbot
apt install certbot python3-certbot-nginx -y

# Get SSL certificate
certbot --nginx -d starwars-rag.yourdomain.com

# Enable HTTPS in docker-compose
docker-compose -f docker-compose.production.yml --profile https up -d
```

## 🔧 Troubleshooting

### Common Issues

#### API Not Starting
```bash
# Check logs
docker logs star-wars-chat-app-star-wars-api-1

# Common causes:
# - Insufficient memory (need 4GB+)
# - Model download failed
# - Database connection issues
```

#### Dashboard Connection Issues
```bash
# Check API URL in environment
echo $API_BASE_URL

# Should be: http://star-wars-api:8000 (internal)
# NOT: http://localhost:8002 (external)
```

#### Database Issues
```bash
# Check PostgreSQL
docker logs star-wars-chat-app-postgres-1

# Reset database (CAUTION: deletes data)
docker-compose -f docker-compose.production.yml down -v
```

### Performance Optimization

#### For High Traffic
```bash
# Increase dashboard replicas
docker-compose -f docker-compose.production.yml up -d --scale star-wars-dashboard=3

# Add load balancer (nginx)
docker-compose -f docker-compose.production.yml --profile https up -d
```

#### For Limited Resources
```bash
# Reduce memory limits in docker-compose.production.yml
# Use smaller LLM model
# Disable analytics panel
```

## 📈 Production Checklist ✅

### Deployment Readiness
- ✅ **Containerized**: Each service in separate container
- ✅ **Environment Config**: All settings via environment variables
- ✅ **Health Checks**: Automatic restart on failure
- ✅ **Resource Limits**: Prevents resource exhaustion
- ✅ **Persistent Storage**: Database and models survive restarts
- ✅ **Security**: Non-root users, network isolation, firewall
- ✅ **Monitoring**: Logs, health endpoints, resource monitoring
- ✅ **Scalability**: Easy horizontal scaling with Docker Compose

### Best Practices Followed
- ✅ **Microservices**: API, Dashboard, Database separated
- ✅ **12-Factor App**: Environment config, stateless processes
- ✅ **Docker**: Production Dockerfiles with optimization
- ✅ **CI/CD Ready**: Automated deployment scripts
- ✅ **Observability**: Comprehensive logging and health checks

## 🎯 Cost Estimation

### Digital Ocean Costs (Monthly)
- **$48/month**: 4GB RAM, 2 CPU, 80GB SSD droplet
- **$5/month**: Optional load balancer
- **$0**: No additional bandwidth charges for normal usage

### Total: ~$50/month for production deployment

## 🆘 Support

### Quick Commands
```bash
# Emergency stop
docker-compose -f docker-compose.production.yml down

# Quick restart
docker-compose -f docker-compose.production.yml restart

# Full redeploy
python3 deploy_digital_ocean.py --update

# View real-time logs
docker-compose -f docker-compose.production.yml logs -f star-wars-api
```

The deployment is production-ready with proper containerization, security, monitoring, and scalability! 🌟
