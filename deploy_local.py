#!/usr/bin/env python3
"""
Star Wars Chat App - Local Deployment Script

This script manages building, running, and testing individual Docker services
for the Star Wars chat application.
"""

import subprocess
import sys
import time
import argparse
from typing import Dict, Any, Optional

class DockerDeployer:
    """Manages Docker deployment for Star Wars Chat services."""
    
    def __init__(self):
        """Initialize the deployer with service configurations."""
        self.services = {
            "stt": {
                "context": "./stt-service",
                "dockerfile": "Dockerfile",
                "tag": "star-wars-stt:latest",
                "port": 5001,
                "description": "Speech-to-Text Service (Whisper)"
            },
            "tts": {
                "context": "./tts-service",
                "dockerfile": "Dockerfile",
                "tag": "star-wars-tts:latest",
                "port": 5002,
                "description": "Text-to-Speech Service (gTTS)"
            },
            "llm": {
                "context": "./llm-service",
                "dockerfile": "Dockerfile",
                "tag": "star-wars-llm:latest",
                "port": 5003,
                "description": "LLM Service (Phi-2)"
            },
            "frontend": {
                "context": "./frontend",
                "dockerfile": "Dockerfile",
                "tag": "star-wars-frontend:latest",
                "port": 3000,
                "description": "React Frontend"
            }
        }
    
    def run_command(self, command: list, description: str = "") -> bool:
        """Run a command and return success status."""
        print(f"\n🔧 {description}")
        print(f"Running: {' '.join(command)}")
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            print(f"✅ {description} - SUCCESS")
            if result.stdout:
                print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} - FAILED")
            print(f"Error: {e}")
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
            return False
    
    def check_image_exists(self, tag: str) -> bool:
        """Check if a Docker image exists."""
        try:
            result = subprocess.run(
                ["docker", "images", "-q", tag], 
                capture_output=True, text=True, check=True
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False
    
    def build_service(self, service: str, force: bool = False) -> bool:
        """Build a specific service."""
        if service not in self.services:
            print(f"❌ Unknown service: {service}")
            return False
        
        config = self.services[service]
        
        # Check if image exists and force rebuild
        if not force and self.check_image_exists(config["tag"]):
            print(f"✅ {service} image already exists: {config['tag']}")
            return True
        
        # Build the service
        build_command = [
            "docker", "build", 
            "-f", config["dockerfile"], 
            "-t", config["tag"], 
            config["context"]
        ]
        
        if force:
            build_command.append("--no-cache")
        
        return self.run_command(build_command, f"Building {service} service")
    
    def build_all_services(self, force: bool = False) -> bool:
        """Build all services."""
        print("\n🚀 Building all services...")
        success = True
        
        for service in self.services:
            if not self.build_service(service, force):
                success = False
        
        return success
    
    def run_service(self, service: str) -> bool:
        """Run a specific service."""
        if service not in self.services:
            print(f"❌ Unknown service: {service}")
            return False
        
        config = self.services[service]
        
        # Stop existing container if running
        self.stop_service(service)
        
        # Run the service
        run_command = [
            "docker", "run", "-d",
            "--name", f"star-wars-{service}",
            "-p", f"{config['port']}:{config['port']}",
            config["tag"]
        ]
        
        return self.run_command(run_command, f"Running {service} service")
    
    def run_all_services(self) -> bool:
        """Run all services."""
        print("\n🚀 Running all services...")
        success = True
        
        for service in self.services:
            if not self.run_service(service):
                success = False
        
        return success
    
    def health_check_service(self, service: str) -> bool:
        """Health check a specific service."""
        if service not in self.services:
            print(f"❌ Unknown service: {service}")
            return False
        
        config = self.services[service]
        
        try:
            import requests
            response = requests.get(f"http://localhost:{config['port']}/health", timeout=10)
            if response.status_code == 200:
                print(f"✅ {service} service is healthy")
                print(f"Response: {response.json()}")
                return True
            else:
                print(f"❌ {service} service health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ {service} service health check failed: {e}")
            return False
    
    def health_check_all(self) -> bool:
        """Health check all services."""
        print("\n🏥 Health checking all services...")
        success = True
        
        for service in self.services:
            if not self.health_check_service(service):
                success = False
        
        return success
    
    def stop_service(self, service: str) -> bool:
        """Stop a specific service."""
        if service not in self.services:
            print(f"❌ Unknown service: {service}")
            return False
        
        stop_command = ["docker", "stop", f"star-wars-{service}"]
        remove_command = ["docker", "rm", f"star-wars-{service}"]
        
        # Try to stop and remove, ignore errors if container doesn't exist
        subprocess.run(stop_command, capture_output=True)
        subprocess.run(remove_command, capture_output=True)
        
        print(f"✅ {service} service stopped and removed")
        return True
    
    def stop_all_services(self) -> bool:
        """Stop all services."""
        print("\n🛑 Stopping all services...")
        success = True
        
        for service in self.services:
            if not self.stop_service(service):
                success = False
        
        return success
    
    def show_status(self):
        """Show status of all services."""
        print("\n📊 Service Status:")
        print("=" * 50)
        
        for service, config in self.services.items():
            image_exists = self.check_image_exists(config["tag"])
            print(f"{service:10} | Image: {'✅' if image_exists else '❌'} | Port: {config['port']}")
        
        print("\n🐳 Running Containers:")
        print("=" * 50)
        
        try:
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=star-wars-"], 
                capture_output=True, text=True, check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError:
            print("No running containers found")

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="Star Wars Chat App - Local Deployment")
    parser.add_argument("action", choices=["build", "run", "health", "stop", "status", "deploy"], 
                       help="Action to perform")
    parser.add_argument("service", nargs="?", choices=["stt", "tts", "llm", "frontend", "all"], 
                       help="Service to target (default: all)")
    parser.add_argument("--force", action="store_true", help="Force rebuild images")
    
    args = parser.parse_args()
    
    deployer = DockerDeployer()
    
    print("🌟 Star Wars Chat App - Local Deployment")
    print("=" * 50)
    
    if args.service is None:
        args.service = "all"
    
    if args.action == "build":
        if args.service == "all":
            deployer.build_all_services(args.force)
        else:
            deployer.build_service(args.service, args.force)
    
    elif args.action == "run":
        if args.service == "all":
            deployer.run_all_services()
        else:
            deployer.run_service(args.service)
    
    elif args.action == "health":
        if args.service == "all":
            deployer.health_check_all()
        else:
            deployer.health_check_service(args.service)
    
    elif args.action == "stop":
        if args.service == "all":
            deployer.stop_all_services()
        else:
            deployer.stop_service(args.service)
    
    elif args.action == "status":
        deployer.show_status()
    
    elif args.action == "deploy":
        print("🚀 Full deployment: build -> run -> health check")
        if deployer.build_all_services(args.force):
            time.sleep(2)
            if deployer.run_all_services():
                time.sleep(5)
                deployer.health_check_all()

if __name__ == "__main__":
    main()
