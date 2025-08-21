#!/usr/bin/env python3
"""
Digital Ocean Production Deployment Script for Star Wars RAG Dashboard

This script automates the deployment of the Star Wars RAG system to Digital Ocean,
including Docker setup, SSL configuration, and health monitoring.
"""

import subprocess
import sys
import time
import requests
import argparse
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(command, check=True, capture_output=False):
    """Run shell command with error handling."""
    logger.info(f"Executing: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check,
            capture_output=capture_output,
            text=True
        )
        if capture_output:
            return result.stdout.strip()
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        return None if capture_output else False

def check_requirements():
    """Check if required tools are installed."""
    logger.info("Checking deployment requirements...")
    
    required_commands = ['docker', 'docker-compose', 'curl']
    missing = []
    
    for cmd in required_commands:
        if not run_command(f"which {cmd}", check=False):
            missing.append(cmd)
    
    if missing:
        logger.error(f"Missing required tools: {', '.join(missing)}")
        logger.info("Please install the missing tools and try again.")
        return False
    
    logger.info("✅ All requirements satisfied")
    return True

def setup_environment():
    """Set up production environment files."""
    logger.info("Setting up production environment...")
    
    # Check if .env exists
    env_file = Path(".env")
    if not env_file.exists():
        logger.warning("⚠️ .env file not found. Please create one from env.example")
        logger.info("Run: cp env.example .env")
        logger.info("Then edit .env with your production values")
        return False
    
    logger.info("✅ Environment file found")
    return True

def build_production_images():
    """Build production Docker images."""
    logger.info("Building production Docker images...")
    
    # Build with production compose file
    success = run_command(
        "docker-compose -f docker-compose.production.yml build --no-cache"
    )
    
    if not success:
        logger.error("❌ Failed to build production images")
        return False
    
    logger.info("✅ Production images built successfully")
    return True

def deploy_services():
    """Deploy services using production configuration."""
    logger.info("Deploying services...")
    
    # Stop any existing services
    run_command(
        "docker-compose -f docker-compose.production.yml down",
        check=False
    )
    
    # Start production services
    success = run_command(
        "docker-compose -f docker-compose.production.yml up -d"
    )
    
    if not success:
        logger.error("❌ Failed to deploy services")
        return False
    
    logger.info("✅ Services deployed successfully")
    return True

def wait_for_health_checks():
    """Wait for all services to be healthy."""
    logger.info("Waiting for services to be healthy...")
    
    services = [
        {"name": "API", "url": "http://localhost:8002/health", "timeout": 60},
        {"name": "Dashboard", "url": "http://localhost:80", "timeout": 30}
    ]
    
    for service in services:
        logger.info(f"Checking {service['name']} health...")
        
        for attempt in range(service['timeout']):
            try:
                response = requests.get(service['url'], timeout=5)
                if response.status_code == 200:
                    logger.info(f"✅ {service['name']} is healthy")
                    break
            except requests.exceptions.RequestException:
                pass
            
            if attempt < service['timeout'] - 1:
                time.sleep(1)
        else:
            logger.error(f"❌ {service['name']} failed health check")
            return False
    
    return True

def show_deployment_info():
    """Display deployment information."""
    logger.info("🎉 Deployment completed successfully!")
    
    print("\n" + "="*60)
    print("🌟 STAR WARS RAG DASHBOARD - PRODUCTION DEPLOYMENT")
    print("="*60)
    print("📊 Dashboard:      http://your-droplet-ip/")
    print("🚀 API Backend:    http://your-droplet-ip:8002/")
    print("📖 API Docs:       http://your-droplet-ip:8002/docs")
    print("🔍 Health Check:   http://your-droplet-ip:8002/health")
    print()
    print("🛠️ Management Commands:")
    print("   # View logs")
    print("   docker-compose -f docker-compose.production.yml logs -f")
    print()
    print("   # Scale dashboard")
    print("   docker-compose -f docker-compose.production.yml up -d --scale star-wars-dashboard=3")
    print()
    print("   # Update deployment")
    print("   python deploy_digital_ocean.py --update")
    print()
    print("   # Stop services") 
    print("   docker-compose -f docker-compose.production.yml down")
    print()
    print("📈 Performance Monitoring:")
    print("   # System resources")
    print("   docker stats")
    print()
    print("   # Container health")
    print("   docker-compose -f docker-compose.production.yml ps")
    print("="*60)
    print("May the Force be with your deployment! ⭐")

def update_deployment():
    """Update existing deployment."""
    logger.info("Updating deployment...")
    
    # Pull latest code (if using git)
    logger.info("Pulling latest changes...")
    run_command("git pull", check=False)
    
    # Rebuild and restart
    if build_production_images():
        if deploy_services():
            if wait_for_health_checks():
                logger.info("✅ Update completed successfully")
                return True
    
    logger.error("❌ Update failed")
    return False

def setup_firewall():
    """Configure UFW firewall for production."""
    logger.info("Configuring firewall...")
    
    firewall_commands = [
        "ufw allow ssh",
        "ufw allow 80/tcp",    # HTTP
        "ufw allow 443/tcp",   # HTTPS
        "ufw allow 8002/tcp",  # API
        "ufw --force enable"
    ]
    
    for cmd in firewall_commands:
        run_command(f"sudo {cmd}", check=False)
    
    logger.info("✅ Firewall configured")

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description="Deploy Star Wars RAG to Digital Ocean")
    parser.add_argument("--update", action="store_true", help="Update existing deployment")
    parser.add_argument("--firewall", action="store_true", help="Configure firewall")
    parser.add_argument("--https", action="store_true", help="Enable HTTPS with Let's Encrypt")
    args = parser.parse_args()
    
    print("🌟 Star Wars RAG - Digital Ocean Deployment Script")
    print("="*50)
    
    if args.update:
        if update_deployment():
            show_deployment_info()
        sys.exit(0)
    
    # Full deployment
    if not check_requirements():
        sys.exit(1)
    
    if not setup_environment():
        sys.exit(1)
    
    if args.firewall:
        setup_firewall()
    
    if not build_production_images():
        sys.exit(1)
    
    if not deploy_services():
        sys.exit(1)
    
    if not wait_for_health_checks():
        logger.warning("⚠️ Some health checks failed, but deployment may still work")
    
    show_deployment_info()

if __name__ == "__main__":
    main()
