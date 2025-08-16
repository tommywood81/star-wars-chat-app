#!/usr/bin/env python3
"""
Windows-compatible local deployment script for Star Wars RAG Chat Application.

This script builds and deploys the complete Star Wars chat system locally using Docker,
making it accessible at http://localhost:8002 for the API and http://localhost:8501 for the web interface.
"""

import subprocess
import sys
import time
import requests
from pathlib import Path
import json
import logging

# Configure Windows-compatible logging (no emojis)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deploy_local.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
API_PORT = 8002
WEB_PORT = 8501
DB_PORT = 5432
API_URL = f"http://localhost:{API_PORT}"
WEB_URL = f"http://localhost:{WEB_PORT}"

def run_command(command, capture_output=False, check=True, show_output=True):
    """Run a shell command and handle errors with detailed logging."""
    logger.info(f"[EXEC] {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=check
        )
        
        # Log outputs for debugging
        if result.stdout:
            if show_output:
                logger.info(f"[SUCCESS] Command output:\n{result.stdout}")
            else:
                logger.debug(f"Command stdout:\n{result.stdout}")
                
        if result.stderr:
            logger.warning(f"[WARN] Command stderr:\n{result.stderr}")
        
        if capture_output:
            return result.stdout.strip()
        else:
            return result.returncode == 0
            
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Command failed: {command}")
        logger.error(f"[ERROR] Exit code: {e.returncode}")
        if e.stdout:
            logger.error(f"[ERROR] Stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"[ERROR] Stderr:\n{e.stderr}")
        return None

def check_docker():
    """Check if Docker is installed and running."""
    logger.info("[CHECK] Checking Docker installation...")
    
    # Check if docker command exists
    result = run_command("docker --version", capture_output=True, check=False)
    if result is None:
        logger.error("[ERROR] Docker is not installed or not in PATH")
        logger.info("Please install Docker Desktop: https://www.docker.com/products/docker-desktop")
        return False
    
    logger.info(f"[SUCCESS] Docker found: {result}")
    
    # Check if Docker is running
    result = run_command("docker info", capture_output=True, check=False)
    if result is None:
        logger.error("[ERROR] Docker is not running")
        logger.info("Please start Docker Desktop and try again")
        return False
    
    logger.info("[SUCCESS] Docker is running")
    return True

def check_docker_compose():
    """Check if Docker Compose is available."""
    logger.info("[CHECK] Checking Docker Compose...")
    
    # Try docker compose (newer syntax)
    result = run_command("docker compose version", capture_output=True, check=False)
    if result:
        logger.info(f"[SUCCESS] Docker Compose found: {result}")
        return "docker compose"
    
    # Try docker-compose (older syntax)
    result = run_command("docker-compose --version", capture_output=True, check=False)
    if result:
        logger.info(f"[SUCCESS] Docker Compose found: {result}")
        return "docker-compose"
    
    logger.error("[ERROR] Docker Compose not found")
    return None

def cleanup_existing_containers():
    """Stop and remove existing containers."""
    logger.info("[CLEANUP] Cleaning up existing containers...")
    
    compose_cmd = check_docker_compose()
    if not compose_cmd:
        return False
    
    # Stop and remove containers
    run_command(f"{compose_cmd} down --remove-orphans", check=False)
    
    # Remove any orphaned containers
    run_command("docker container prune -f", check=False)
    
    logger.info("[SUCCESS] Cleanup completed")
    return True

def build_docker_image():
    """Build the Docker image."""
    logger.info("[BUILD] Building Docker image...")
    
    compose_cmd = check_docker_compose()
    if not compose_cmd:
        return False
    
    # Show build progress
    logger.info("[BUILD] Starting Docker build process (this may take a few minutes)...")
    
    # Build the image with verbose output
    success = run_command(f"{compose_cmd} build --no-cache --progress=plain", check=False, show_output=True)
    if not success:
        logger.error("[ERROR] Docker image build failed")
        return False
    
    logger.info("[SUCCESS] Docker image built successfully")
    return True

def start_services():
    """Start the Docker services."""
    logger.info("[START] Starting services...")
    
    compose_cmd = check_docker_compose()
    if not compose_cmd:
        return False
    
    # Start core services (API + Database)
    success = run_command(f"{compose_cmd} up -d", check=False)
    if not success:
        logger.error("[ERROR] Failed to start services")
        return False
    
    logger.info("[SUCCESS] Services started successfully")
    return True

def wait_for_service(url, service_name, max_attempts=30, delay=2):
    """Wait for a service to become available."""
    logger.info(f"[WAIT] Waiting for {service_name} to be ready at {url}...")
    
    for attempt in range(max_attempts):
        try:
            logger.debug(f"[ATTEMPT] Connecting to {url}/health (attempt {attempt + 1})")
            response = requests.get(f"{url}/health", timeout=5)
            logger.debug(f"[RESPONSE] Status: {response.status_code}")
            
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"[SUCCESS] {service_name} is ready! Health status: {health_data}")
                return True
            else:
                logger.debug(f"[WAIT] Service returned status {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.debug(f"[RETRY] Connection attempt failed: {e}")
        
        if attempt < max_attempts - 1:
            logger.info(f"[WAIT] {service_name} not ready yet, waiting {delay}s... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)
        else:
            logger.warning(f"[FINAL] Final attempt for {service_name}...")
    
    logger.error(f"[ERROR] {service_name} failed to start within {max_attempts * delay} seconds")
    
    # Show container status for debugging
    logger.info("[DEBUG] Checking container status...")
    run_command("docker compose ps", show_output=True)
    
    return False

def test_api():
    """Test the API functionality."""
    logger.info("[TEST] Testing API functionality...")
    
    try:
        # Test health endpoint
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code != 200:
            logger.error(f"[ERROR] Health check failed: {response.status_code}")
            return False
        
        health_data = response.json()
        logger.info(f"[SUCCESS] Health check passed: {health_data.get('status', 'unknown')}")
        
        # Test characters endpoint
        response = requests.get(f"{API_URL}/characters", timeout=10)
        if response.status_code == 200:
            characters_data = response.json()
            char_count = len(characters_data.get('characters', []))
            logger.info(f"[SUCCESS] Characters endpoint working: {char_count} characters available")
        else:
            logger.warning(f"[WARN] Characters endpoint returned {response.status_code}")
        
        # Test system info
        response = requests.get(f"{API_URL}/system/info", timeout=10)
        if response.status_code == 200:
            system_data = response.json()
            logger.info(f"[SUCCESS] System info: {system_data.get('dialogue_lines', 0)} dialogue lines loaded")
        else:
            logger.warning(f"[WARN] System info endpoint returned {response.status_code}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"[ERROR] API test failed: {e}")
        return False

def show_logs():
    """Show recent logs from services."""
    logger.info("[LOGS] Recent service logs:")
    
    compose_cmd = check_docker_compose()
    if compose_cmd:
        run_command(f"{compose_cmd} logs --tail=20", check=False)

def display_access_info():
    """Display access information for the user."""
    print("\n" + "="*60)
    print("*** STAR WARS RAG CHAT - LOCAL DEPLOYMENT COMPLETE! ***")
    print("="*60)
    print(f"API Backend:     {API_URL}")
    print(f"   - Health Check:  {API_URL}/health")
    print(f"   - API Docs:      {API_URL}/docs")
    print(f"   - Characters:    {API_URL}/characters")
    print(f"   - Chat Endpoint: {API_URL}/chat")
    print()
    print(f"Web Interface:   {WEB_URL}")
    print(f"   - Chat UI:       {WEB_URL}")
    print()
    print(f"Database:        postgresql://postgres:star_wars_password@localhost:{DB_PORT}/star_wars_rag")
    print()
    print("Usage Examples:")
    print("   # Test the API")
    print(f"   curl {API_URL}/health")
    print()
    print("   # Chat with a character")
    print(f"""   curl -X POST {API_URL}/chat \\
     -H "Content-Type: application/json" \\
     -d '{{"character": "Luke Skywalker", "message": "Tell me about the Force", "session_id": "test"}}'""")
    print()
    print("Management Commands:")
    print("   # View logs")
    print("   docker compose logs -f")
    print()
    print("   # Stop services")
    print("   docker compose down")
    print()
    print("   # Restart services")
    print("   docker compose restart")
    print()
    print("Available Characters:")
    
    try:
        response = requests.get(f"{API_URL}/characters", timeout=5)
        if response.status_code == 200:
            characters_data = response.json()
            characters = characters_data.get('characters', [])
            for char in characters[:8]:  # Show first 8
                char_name = char.get('name', 'Unknown') if isinstance(char, dict) else char
                dialogue_count = char.get('dialogue_count', 0) if isinstance(char, dict) else 0
                print(f"   - {char_name}" + (f" ({dialogue_count} lines)" if dialogue_count > 0 else ""))
            if len(characters) > 8:
                print(f"   ... and {len(characters) - 8} more!")
        else:
            print("   (Characters list not available yet)")
    except:
        print("   (Characters list not available yet)")
    
    print()
    print("*** May the Force be with you! ***")
    print("="*60)

def main():
    """Main deployment function."""
    print("*** Star Wars RAG Chat - Local Deployment Script ***")
    print("="*50)
    
    logger.info("[START] Starting deployment process...")
    logger.info(f"[INFO] Target URLs:")
    logger.info(f"   - API: {API_URL}")
    logger.info(f"   - Web: {WEB_URL}")
    logger.info(f"   - DB: localhost:{DB_PORT}")
    
    # Check prerequisites
    logger.info("[STEP-1] Checking prerequisites...")
    if not check_docker():
        logger.error("[ERROR] Docker check failed")
        sys.exit(1)
    
    compose_cmd = check_docker_compose()
    if not compose_cmd:
        logger.error("[ERROR] Docker Compose check failed")
        sys.exit(1)
    
    # Verify project structure
    logger.info("[STEP-2] Verifying project structure...")
    required_files = ['Dockerfile', 'docker-compose.yml', 'requirements.txt', 'src/star_wars_rag/api.py']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        logger.error(f"[ERROR] Missing required files: {missing_files}")
        sys.exit(1)
    
    logger.info("[SUCCESS] Project structure verified")
    logger.info(f"[INFO] Found all required files: {required_files}")
    
    try:
        # Deployment steps
        logger.info("[STEP-3] Cleaning up existing containers...")
        if not cleanup_existing_containers():
            logger.error("[ERROR] Cleanup failed")
            sys.exit(1)
        
        logger.info("[STEP-4] Building Docker images...")
        if not build_docker_image():
            logger.error("[ERROR] Build failed")
            sys.exit(1)
        
        logger.info("[STEP-5] Starting services...")
        if not start_services():
            logger.error("[ERROR] Service startup failed")
            sys.exit(1)
        
        # Wait for API to be ready
        logger.info("[STEP-6] Waiting for services to be ready...")
        if not wait_for_service(API_URL, "API Backend"):
            logger.error("[ERROR] API failed to start properly")
            show_logs()
            sys.exit(1)
        
        # Test API functionality
        logger.info("[STEP-7] Testing API functionality...")
        if not test_api():
            logger.error("[ERROR] API tests failed")
            show_logs()
            sys.exit(1)
        
        # Display success information
        logger.info("[STEP-8] Deployment complete!")
        display_access_info()
        
        # Optionally show logs
        print(f"\n[INFO] To view live logs: docker compose logs -f")
        print(f"[INFO] To stop services: docker compose down")
        
    except KeyboardInterrupt:
        logger.info("\n[INTERRUPT] Deployment interrupted by user")
        cleanup_existing_containers()
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR] Deployment failed: {e}")
        show_logs()
        sys.exit(1)

if __name__ == "__main__":
    main()
