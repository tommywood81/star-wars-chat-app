#!/usr/bin/env python3
"""
Star Wars Chat App - Local Deployment Script

This script manages the Star Wars chat application using docker-compose
for simplified deployment and management.
"""

import subprocess
import sys
import time
import argparse
import requests
import socket
import psutil
from typing import Dict, Any, Optional

class DockerComposeDeployer:
    """Manages Docker Compose deployment for Star Wars Chat services."""
    
    def __init__(self):
        """Initialize the deployer."""
        self.services = {
            "stt": {"port": 5001, "description": "Speech-to-Text Service"},
            "tts": {"port": 5002, "description": "Text-to-Speech Service"},
            "llm": {"port": 5003, "description": "LLM Service"},
            "frontend": {"port": 3000, "description": "React Frontend"},
            "postgres": {"port": 5432, "description": "PostgreSQL Database"}
        }
    
    def check_port_in_use(self, port: int) -> bool:
        """Check if a port is in use."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                result = s.connect_ex(('localhost', port))
                return result == 0
        except Exception:
            return False
    
    def find_process_using_port(self, port: int) -> Optional[psutil.Process]:
        """Find the process using a specific port."""
        try:
            for conn in psutil.net_connections():
                if conn.laddr.port == port and conn.status == 'LISTEN':
                    return psutil.Process(conn.pid)
        except Exception:
            pass
        return None
    
    def stop_process_on_port(self, port: int) -> bool:
        """Stop the process using a specific port."""
        process = self.find_process_using_port(port)
        if process:
            # Don't stop Docker processes or system processes
            process_name = process.name().lower()
            if any(name in process_name for name in ['docker', 'postgres', 'wsl', 'system']):
                print(f"⚠️  Skipping system process {process.name()} (PID: {process.pid}) on port {port}")
                return True
            
            try:
                print(f"🛑 Stopping process {process.name()} (PID: {process.pid}) on port {port}")
                process.terminate()
                process.wait(timeout=5)
                return True
            except Exception as e:
                print(f"❌ Failed to stop process on port {port}: {e}")
                return False
        return False
    
    def check_port_conflicts(self, service: str = None) -> bool:
        """Check for port conflicts and optionally resolve them."""
        conflicts = []
        
        services_to_check = [service] if service else self.services.keys()
        
        for service_name in services_to_check:
            if service_name not in self.services:
                continue
                
            port = self.services[service_name]["port"]
            if self.check_port_in_use(port):
                process = self.find_process_using_port(port)
                # Don't consider Docker/system processes as conflicts
                if process:
                    process_name = process.name().lower()
                    if any(name in process_name for name in ['docker', 'postgres', 'wsl', 'system']):
                        continue  # Skip system processes
                conflicts.append((service_name, port))
        
        if conflicts:
            print(f"\n⚠️  Port conflicts detected:")
            for service_name, port in conflicts:
                process = self.find_process_using_port(port)
                if process:
                    print(f"   Port {port} ({service_name}): {process.name()} (PID: {process.pid})")
                else:
                    print(f"   Port {port} ({service_name}): Unknown process")
            
            return True
        
        return False
    
    def resolve_port_conflicts(self, service: str = None) -> bool:
        """Resolve port conflicts by stopping conflicting processes."""
        print("\n🔍 Checking for port conflicts...")
        
        if not self.check_port_conflicts(service):
            print("✅ No port conflicts detected")
            return True
        
        print("\n🛠️  Resolving port conflicts...")
        services_to_check = [service] if service else self.services.keys()
        
        for service_name in services_to_check:
            if service_name not in self.services:
                continue
                
            port = self.services[service_name]["port"]
            if self.check_port_in_use(port):
                if not self.stop_process_on_port(port):
                    print(f"❌ Failed to resolve conflict on port {port}")
                    return False
        
        # Wait a moment for processes to fully stop
        time.sleep(2)
        
        # Verify conflicts are resolved
        if self.check_port_conflicts(service):
            print("❌ Some port conflicts could not be resolved")
            return False
        
        print("✅ All port conflicts resolved")
        return True
    
    def run_command(self, command: list, description: str = "") -> bool:
        """Run a command and return success status."""
        if description:
            print(f"\n🔧 {description}")
            print(f"Running: {' '.join(command)}")
        
        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            if description:
                print(f"✅ {description} - SUCCESS")
            if result.stdout:
                print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            if description:
                print(f"❌ {description} - FAILED")
                print(f"Error: {e}")
            if e.stdout:
                print(f"STDOUT: {e.stdout}")
            if e.stderr:
                print(f"STDERR: {e.stderr}")
            return False
    
    def build_services(self, force: bool = False, service: str = None) -> bool:
        """Build services using docker-compose."""
        print("\n🚀 Building services...")
        command = ["docker-compose", "build"]
        if force:
            command.append("--no-cache")
        if service:
            command.append(service)
            print(f"Targeting service: {service}")
        return self.run_command(command, "Building services with docker-compose")
    
    def start_services(self, service: str = None) -> bool:
        """Start services using docker-compose."""
        print("\n🚀 Starting services...")
        command = ["docker-compose", "up", "-d"]
        if service:
            command.append(service)
            print(f"Targeting service: {service}")
        return self.run_command(command, "Starting services with docker-compose")
    
    def stop_services(self, service: str = None) -> bool:
        """Stop services using docker-compose."""
        print("\n🛑 Stopping services...")
        if service:
            command = ["docker-compose", "stop", service]
            print(f"Targeting service: {service}")
        else:
            command = ["docker-compose", "down"]
        return self.run_command(command, "Stopping services with docker-compose")
    
    def restart_services(self, service: str = None) -> bool:
        """Restart services using docker-compose."""
        print("\n🔄 Restarting services...")
        command = ["docker-compose", "restart"]
        if service:
            command.append(service)
            print(f"Targeting service: {service}")
        return self.run_command(command, "Restarting services with docker-compose")
    
    def show_status(self, service: str = None):
        """Show status of services."""
        print("\n📊 Service Status:")
        print("=" * 50)
        
        # Show docker-compose status
        print("\n🐳 Docker Compose Status:")
        print("=" * 50)
        if service:
            self.run_command(["docker-compose", "ps", service])
        else:
            self.run_command(["docker-compose", "ps"])
        
        # Show individual service health
        print("\n🏥 Service Health:")
        print("=" * 50)
        if service:
            self.health_check_service(service)
        else:
            self.health_check_all()
    
    def health_check_service(self, service: str) -> bool:
        """Health check a specific service."""
        if service not in self.services:
            print(f"❌ Unknown service: {service}")
            return False
        
        config = self.services[service]
        
        # Skip health check for postgres (docker-compose handles it)
        if service == "postgres":
            print(f"✅ {service} service health managed by docker-compose")
            return True
        
        try:
            response = requests.get(f"http://localhost:{config['port']}/health", timeout=10)
            if response.status_code == 200:
                print(f"✅ {service} service is healthy")
                try:
                    print(f"Response: {response.json()}")
                except:
                    print(f"Response: {response.text}")
                return True
            else:
                print(f"❌ {service} service health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ {service} service health check failed: {e}")
            return False
    
    def health_check_all(self) -> bool:
        """Health check all services."""
        success = True
        
        for service in self.services:
            if not self.health_check_service(service):
                success = False
        
        return success
    
    def logs(self, service: str = None, follow: bool = False):
        """Show logs for services."""
        command = ["docker-compose", "logs"]
        if follow:
            command.append("-f")
        if service:
            command.append(service)
        
        # Don't capture output for logs, let them stream
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to show logs: {e}")
    
    def deploy_service(self, service: str, force: bool = False) -> bool:
        """Deploy a specific service: build -> start -> health check."""
        print(f"🚀 Deploying {service} service: build -> start -> health check")
        
        # Resolve port conflicts first
        if not self.resolve_port_conflicts(service):
            return False
        
        # Build the specific service
        if not self.build_services(force, service):
            return False
        
        # Start the specific service
        if not self.start_services(service):
            return False
        
        # Wait for service to be ready
        print(f"\n⏳ Waiting for {service} service to be ready...")
        time.sleep(5)
        
        # Health check the specific service
        print(f"\n🏥 Running health check for {service}...")
        return self.health_check_service(service)
    
    def deploy(self, force: bool = False, service: str = None) -> bool:
        """Full deployment: build -> start -> health check."""
        if service:
            return self.deploy_service(service, force)
        
        print("🚀 Full deployment: build -> start -> health check")
        
        # Resolve port conflicts first
        if not self.resolve_port_conflicts():
            return False
        
        # Build services
        if not self.build_services(force):
            return False
        
        # Start services
        if not self.start_services():
            return False
        
        # Wait for services to be ready
        print("\n⏳ Waiting for services to be ready...")
        time.sleep(10)
        
        # Health check
        print("\n🏥 Running health checks...")
        return self.health_check_all()

def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="Star Wars Chat App - Local Deployment")
    parser.add_argument("action", choices=["build", "start", "stop", "restart", "health", "status", "logs", "deploy", "check-ports"], 
                       help="Action to perform")
    parser.add_argument("service", nargs="?", 
                       help="Service to target (stt, tts, llm, frontend, postgres)")
    parser.add_argument("--force", action="store_true", help="Force rebuild images")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow logs (for logs command)")
    parser.add_argument("--resolve-conflicts", action="store_true", help="Automatically resolve port conflicts")
    
    args = parser.parse_args()
    
    deployer = DockerComposeDeployer()
    
    print("🌟 Star Wars Chat App - Local Deployment")
    print("=" * 50)
    
    # Validate service name if provided
    if args.service and args.service not in deployer.services:
        print(f"❌ Unknown service: {args.service}")
        print(f"Available services: {', '.join(deployer.services.keys())}")
        sys.exit(1)
    
    if args.action == "check-ports":
        if deployer.check_port_conflicts(args.service):
            print("\n❌ Port conflicts detected!")
            if args.resolve_conflicts:
                deployer.resolve_port_conflicts(args.service)
        else:
            print("\n✅ No port conflicts detected")
    
    elif args.action == "build":
        deployer.build_services(args.force, args.service)
    
    elif args.action == "start":
        if args.resolve_conflicts:
            deployer.resolve_port_conflicts(args.service)
        deployer.start_services(args.service)
    
    elif args.action == "stop":
        deployer.stop_services(args.service)
    
    elif args.action == "restart":
        deployer.restart_services(args.service)
    
    elif args.action == "health":
        if args.service:
            deployer.health_check_service(args.service)
        else:
            deployer.health_check_all()
    
    elif args.action == "status":
        deployer.show_status(args.service)
    
    elif args.action == "logs":
        deployer.logs(args.service, args.follow)
    
    elif args.action == "deploy":
        deployer.deploy(args.force, args.service)

if __name__ == "__main__":
    main()
