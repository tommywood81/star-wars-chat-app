#!/usr/bin/env python3
"""
Local deployment script for Star Wars RAG Chat Application.

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
import threading
import argparse

# Configure comprehensive logging with UTF-8 encoding for Windows compatibility
import codecs

# Set stdout to use UTF-8 encoding for Windows compatibility
if sys.platform.startswith('win'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deploy_local.log', mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Progress indicator for long-running operations
class ProgressIndicator:
    def __init__(self, message="Working"):
        self.message = message
        self.running = False
        self.thread = None
        
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._show_progress)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the progress line
        print("\r" + " " * 80 + "\r", end="", flush=True)
        
    def _show_progress(self):
        spinner = ["|", "/", "-", "\\"]
        i = 0
        while self.running:
            print(f"\r{self.message}... {spinner[i % len(spinner)]}", end="", flush=True)
            i += 1
            time.sleep(0.5)

# Also capture subprocess output
def log_subprocess_output(process_result, command):
    """Log subprocess output for debugging."""
    if process_result and hasattr(process_result, 'stdout') and process_result.stdout:
        logger.debug(f"Command: {command}")
        logger.debug(f"STDOUT:\n{process_result.stdout}")
    if process_result and hasattr(process_result, 'stderr') and process_result.stderr:
        logger.debug(f"STDERR:\n{process_result.stderr}")

# Configuration
API_PORT = 8002
WEB_PORT = 8501
DB_PORT = 5432
API_URL = f"http://localhost:{API_PORT}"
WEB_URL = f"http://localhost:{WEB_PORT}"

def run_command(command, capture_output=False, check=True, show_output=True, show_progress=False, progress_message="Processing"):
    """Run a shell command and handle errors with detailed logging."""
    logger.info(f"[EXEC] Executing: {command}")
    
    progress = None
    if show_progress:
        progress = ProgressIndicator(progress_message)
        progress.start()
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            check=check,
            timeout=600  # 10 minute timeout for builds
        )
        
        if progress:
            progress.stop()
        
        # Log outputs for debugging
        if result.stdout:
            if show_output:
                logger.info(f"[SUCCESS] Command output:\n{result.stdout}")
            else:
                logger.debug(f"Command stdout:\n{result.stdout}")
                
        if result.stderr:
            logger.warning(f"[WARNING] Command stderr:\n{result.stderr}")
        
        if capture_output:
            return result.stdout.strip()
        else:
            return result.returncode == 0
            
    except subprocess.TimeoutExpired:
        if progress:
            progress.stop()
        logger.error(f"[ERROR] Command timed out after 10 minutes: {command}")
        return None
    except subprocess.CalledProcessError as e:
        if progress:
            progress.stop()
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

def check_existing_images():
    """Check if Docker images already exist."""
    logger.info("[CHECK] Checking for existing Docker images...")
    
    # Get list of images for this project
    images_output = run_command("docker images --format \"{{.Repository}}:{{.Tag}}\"", capture_output=True, check=False)
    if images_output:
        existing_images = images_output.split('\n')
        project_images = [img for img in existing_images if 'star-wars-chat-app' in img or 'star_wars_rag' in img]
        
        if project_images:
            logger.info(f"[INFO] Found existing images: {', '.join(project_images)}")
            return True
        else:
            logger.info("[INFO] No existing project images found")
            return False
    else:
        logger.info("[INFO] No existing images found")
        return False

def build_docker_image(clean_build=False):
    """Build the Docker image with intelligent caching."""
    logger.info("[BUILD] Building Docker image...")
    
    compose_cmd = check_docker_compose()
    if not compose_cmd:
        return False
    
    # Check for existing images
    has_existing_images = check_existing_images()
    
    # Determine build strategy
    if has_existing_images and not clean_build:
        logger.info("[BUILD] Existing images found - using smart caching for faster build")
        build_type = "incremental build (smart caching)"
        cache_flag = ""
    elif clean_build:
        logger.info("[BUILD] Clean build requested - rebuilding from scratch")
        build_type = "clean build (no cache)"
        cache_flag = "--no-cache"
    else:
        logger.info("[BUILD] No existing images - building from scratch with caching enabled")
        build_type = "initial build (with caching)"
        cache_flag = ""
    
    logger.info(f"[BUILD] Starting Docker {build_type} process...")
    
    # Build command with appropriate caching strategy
    build_cmd = f"{compose_cmd} build {cache_flag} --progress=plain".strip()
    
    # Build the image with verbose output and progress indicator
    success = run_command(
        build_cmd, 
        check=False, 
        show_output=True, 
        show_progress=True, 
        progress_message=f"Building Docker image ({build_type.split(' ')[0]})"
    )
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
    if success is None:
        logger.error("[ERROR] Failed to start services")
        return False
    
    logger.info("[SUCCESS] Services started successfully")
    return True

def start_web_interface():
    """Start the web interface (optional)."""
    logger.info("[WEB] Starting web interface...")
    
    compose_cmd = check_docker_compose()
    if not compose_cmd:
        return False
    
    # Start web interface
    success = run_command(f"{compose_cmd} --profile web up -d", check=False)
    if success is None:
        logger.warning("[WARNING] Web interface failed to start (this is optional)")
        return False
    
    logger.info("[SUCCESS] Web interface started successfully")
    return True

def wait_for_service(url, service_name, max_attempts=30, delay=2):
    """Wait for a service to become available."""
    logger.info(f"[WAIT] Waiting for {service_name} to be ready at {url}...")
    
    # Give the service a bit more time to start up initially
    logger.info(f"[WAIT] Giving {service_name} a moment to initialize...")
    time.sleep(5)
    
    for attempt in range(max_attempts):
        try:
            logger.debug(f"[CHECK] Attempting to connect to {url}/health (attempt {attempt + 1})")
            # Use a more lenient timeout and handle connection issues better
            response = requests.get(
                f"{url}/health", 
                timeout=10,
                headers={'Connection': 'close'}  # Avoid connection pooling issues
            )
            logger.debug(f"[RESPONSE] Response status: {response.status_code}")
            
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"[SUCCESS] {service_name} is ready! Health status: {health_data}")
                return True
            else:
                logger.debug(f"[WARNING] Service returned status {response.status_code}")
                
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            logger.debug(f"[RETRY] Connection failed (expected during startup): {type(e).__name__}")
        except requests.exceptions.RequestException as e:
            logger.debug(f"[RETRY] Request failed: {e}")
        
        if attempt < max_attempts - 1:
            # Reduce log noise - only show every 5th attempt
            if attempt % 5 == 0:
                logger.info(f"[WAIT] {service_name} not ready yet, waiting {delay}s... (attempt {attempt + 1}/{max_attempts})")
            time.sleep(delay)
        else:
            logger.warning(f"[WARNING] Final attempt for {service_name}...")
    
    # Before failing, try one more direct check
    logger.info("[CHECK] Final verification - testing API directly...")
    try:
        response = requests.get(f"{url}/health", timeout=15)
        if response.status_code == 200:
            health_data = response.json()
            logger.info(f"[SUCCESS] {service_name} is actually ready! Health status: {health_data}")
            return True
    except Exception as e:
        logger.debug(f"[ERROR] Final check failed: {e}")
    
    logger.error(f"[ERROR] {service_name} failed to start within {max_attempts * delay} seconds")
    
    # Show container status for debugging
    logger.info("[CHECK] Checking container status...")
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
            logger.warning(f"[WARNING] Characters endpoint returned {response.status_code}")
        
        # Test system info
        response = requests.get(f"{API_URL}/system/info", timeout=10)
        if response.status_code == 200:
            system_data = response.json()
            logger.info(f"[SUCCESS] System info: {system_data.get('dialogue_lines', 0)} dialogue lines loaded")
        else:
            logger.warning(f"[WARNING] System info endpoint returned {response.status_code}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"[ERROR] API test failed: {e}")
        return False

def show_logs():
    """Show recent logs from services."""
    logger.info("[INFO] Recent service logs:")
    
    compose_cmd = check_docker_compose()
    if compose_cmd:
        run_command(f"{compose_cmd} logs --tail=20", check=False)

def display_access_info():
    """Display access information for the user."""
    print("\n" + "="*60)
    print("*** STAR WARS RAG CHAT - LOCAL DEPLOYMENT COMPLETE! ***")
    print("="*60)
    print(f"[START] API Backend:     {API_URL}")
    print(f"   - Health Check:  {API_URL}/health")
    print(f"   - API Docs:      {API_URL}/docs")
    print(f"   - Characters:    {API_URL}/characters")
    print(f"   - Chat Endpoint: {API_URL}/chat")
    print()
    print(f"[WEB] Web Interface:   {WEB_URL}")
    print(f"   - Chat UI:       {WEB_URL}")
    print()
    print(f"[DATABASE] Database:        postgresql://postgres:star_wars_password@localhost:{DB_PORT}/star_wars_rag")
    print()
    print("📖 Usage Examples:")
    print("   # Test the API")
    print(f"   curl {API_URL}/health")
    print()
    print("   # Chat with a character")
    print(f"""   curl -X POST {API_URL}/chat \\
     -H "Content-Type: application/json" \\
     -d '{{"character": "Luke Skywalker", "message": "Tell me about the Force", "session_id": "test"}}'""")
    print()
    print("[TOOLS] Management Commands:")
    print("   # View logs")
    print("   docker compose logs -f")
    print()
    print("   # Stop services")
    print("   docker compose down")
    print()
    print("   # Restart services")
    print("   docker compose restart")
    print()
    print("🎭 Available Characters:")
    
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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Deploy Star Wars RAG Chat locally")
    parser.add_argument("--clean", action="store_true", help="Perform clean build (no cache) - slower but ensures fresh build")
    parser.add_argument("--quick", action="store_true", help="Quick mode - skip optional components")
    args = parser.parse_args()
    
    print("*** Star Wars RAG Chat - Local Deployment Script ***")
    print("="*50)
    
    if args.clean:
        print("[INFO] Clean build mode - will rebuild everything from scratch")
    else:
        print("[INFO] Incremental build mode - using Docker cache for faster builds")
    
    logger.info("[START] Starting deployment process...")
    logger.info(f"[INFO] Target URLs:")
    logger.info(f"   - API: {API_URL}")
    logger.info(f"   - Web: {WEB_URL}")
    logger.info(f"   - DB: localhost:{DB_PORT}")
    
    # Check prerequisites
    logger.info("[CHECK] Step 1: Checking prerequisites...")
    if not check_docker():
        logger.error("[ERROR] Docker check failed")
        sys.exit(1)
    
    compose_cmd = check_docker_compose()
    if not compose_cmd:
        logger.error("[ERROR] Docker Compose check failed")
        sys.exit(1)
    
    # Verify project structure
    logger.info("[CHECK] Step 2: Verifying project structure...")
    required_files = ['Dockerfile', 'docker-compose.yml', 'requirements.txt', 'src/star_wars_rag/api.py']
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        logger.error(f"[ERROR] Missing required files: {missing_files}")
        sys.exit(1)
    
    logger.info("[SUCCESS] Project structure verified")
    logger.info(f"[INFO] Found all required files: {required_files}")
    
    try:
        # Deployment steps
        logger.info("[CLEANUP] Step 3: Cleaning up existing containers...")
        if not cleanup_existing_containers():
            logger.error("[ERROR] Cleanup failed")
            sys.exit(1)
        
        logger.info("[BUILD] Step 4: Building Docker images...")
        if not build_docker_image(clean_build=args.clean):
            logger.error("[ERROR] Build failed")
            sys.exit(1)
        
        logger.info("[START] Step 5: Starting services...")
        if not start_services():
            logger.error("[ERROR] Service startup failed")
            sys.exit(1)
        
        # Wait for API to be ready
        logger.info("[WAIT] Step 6: Waiting for services to be ready...")
        if not wait_for_service(API_URL, "API Backend"):
            logger.error("[ERROR] API failed to start properly")
            show_logs()
            sys.exit(1)
        
        # Test API functionality
        logger.info("[TEST] Step 7: Testing API functionality...")
        if not test_api():
            logger.error("[ERROR] API tests failed")
            show_logs()
            sys.exit(1)
        
        # Try to start web interface (optional)
        logger.info("[WEB] Step 8: Starting web interface...")
        web_started = start_web_interface()
        if web_started:
            logger.info("[WAIT] Giving web interface time to initialize...")
            time.sleep(10)  # Give web interface time to start
        
        # Display success information
        logger.info("[COMPLETE] Step 9: Deployment complete!")
        display_access_info()
        
        # Optionally show logs
        print(f"\n[INFO] To view live logs: docker compose logs -f")
        print(f"[STOP] To stop services: docker compose down")
        
    except KeyboardInterrupt:
        logger.info("\n[WARNING] Deployment interrupted by user")
        cleanup_existing_containers()
        sys.exit(1)
    except Exception as e:
        logger.error(f"[ERROR] Deployment failed: {e}")
        show_logs()
        sys.exit(1)

if __name__ == "__main__":
    main()
