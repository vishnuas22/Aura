#!/usr/bin/env python3
"""
Comprehensive System Test Suite
Tests all components to identify issues systematically.
"""

import asyncio
import logging
import sys
import requests
import json
from pathlib import Path
from datetime import datetime
import subprocess
import time

class SystemTester:
    def __init__(self):
        self.issues = []
        self.passed_tests = []
        self.backend_url = "http://localhost:8001"
        self.frontend_url = "http://localhost:3000"
        
    def log_issue(self, test_name: str, issue: str, severity: str = "ERROR"):
        """Log an identified issue."""
        self.issues.append({
            "test": test_name,
            "issue": issue,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat()
        })
        print(f"❌ [{severity}] {test_name}: {issue}")
    
    def log_success(self, test_name: str, message: str = "Passed"):
        """Log a successful test."""
        self.passed_tests.append({
            "test": test_name,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })
        print(f"✅ {test_name}: {message}")

    def test_backend_service(self):
        """Test if backend service is running and accessible."""
        print("\n🔧 Testing Backend Service...")
        
        try:
            # Test basic API endpoint
            response = requests.get(f"{self.backend_url}/api/", timeout=10)
            if response.status_code == 200:
                self.log_success("Backend API", f"Accessible - {response.json()}")
            else:
                self.log_issue("Backend API", f"HTTP {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            self.log_issue("Backend Service", "Service not running or not accessible on port 8001")
        except requests.exceptions.Timeout:
            self.log_issue("Backend Service", "Service timeout - too slow to respond")
        except Exception as e:
            self.log_issue("Backend Service", f"Unexpected error: {str(e)}")

    def test_frontend_service(self):
        """Test if frontend service is running and accessible."""
        print("\n🔧 Testing Frontend Service...")
        
        try:
            response = requests.get(self.frontend_url, timeout=10)
            if response.status_code == 200:
                self.log_success("Frontend Service", "Accessible and serving React app")
            else:
                self.log_issue("Frontend Service", f"HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            self.log_issue("Frontend Service", "Service not running or not accessible on port 3000")
        except requests.exceptions.Timeout:
            self.log_issue("Frontend Service", "Service timeout")
        except Exception as e:
            self.log_issue("Frontend Service", f"Unexpected error: {str(e)}")

    def test_backend_endpoints(self):
        """Test all backend API endpoints."""
        print("\n🔧 Testing Backend API Endpoints...")
        
        endpoints = [
            ("/api/", "GET", "Basic API endpoint"),
            ("/api/status", "GET", "Status checks endpoint"),  
            ("/api/agents/health", "GET", "Agent health check"),
            ("/api/agents/test", "POST", "Agent testing endpoint")
        ]
        
        for endpoint, method, description in endpoints:
            try:
                if method == "GET":
                    response = requests.get(f"{self.backend_url}{endpoint}", timeout=10)
                elif method == "POST":
                    # Test with minimal data for POST endpoints
                    test_data = {
                        "agent_type": "researcher",
                        "task_type": "test",
                        "description": "Test task"
                    }
                    response = requests.post(
                        f"{self.backend_url}{endpoint}", 
                        json=test_data,
                        timeout=30
                    )
                
                if response.status_code in [200, 201]:
                    self.log_success(f"Endpoint {endpoint}", f"{description} - Working")
                else:
                    self.log_issue(f"Endpoint {endpoint}", 
                                 f"{description} - HTTP {response.status_code}: {response.text[:200]}")
                    
            except requests.exceptions.ConnectionError:
                self.log_issue(f"Endpoint {endpoint}", f"{description} - Connection refused")
            except requests.exceptions.Timeout:
                self.log_issue(f"Endpoint {endpoint}", f"{description} - Timeout")
            except Exception as e:
                self.log_issue(f"Endpoint {endpoint}", f"{description} - Error: {str(e)}")

    def test_database_connectivity(self):
        """Test MongoDB database connectivity."""
        print("\n🔧 Testing Database Connectivity...")
        
        try:
            # Test by trying to access status endpoint which uses DB
            response = requests.get(f"{self.backend_url}/api/status", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    self.log_success("Database Connectivity", "MongoDB accessible via API")
                else:
                    self.log_issue("Database Connectivity", "Unexpected response format from status endpoint")
            else:
                self.log_issue("Database Connectivity", f"Status endpoint failed: HTTP {response.status_code}")
        except Exception as e:
            self.log_issue("Database Connectivity", f"Error testing DB via API: {str(e)}")

    def test_supervisor_services(self):
        """Test supervisor service status."""
        print("\n🔧 Testing Supervisor Services...")
        
        try:
            result = subprocess.run(['sudo', 'supervisorctl', 'status'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                output = result.stdout
                services = ['backend', 'frontend']
                
                for service in services:
                    if f"{service}                          RUNNING" in output:
                        self.log_success(f"Supervisor {service}", "Service running")
                    elif service in output and "RUNNING" in output:
                        self.log_success(f"Supervisor {service}", "Service running")
                    else:
                        self.log_issue(f"Supervisor {service}", f"Service not running properly")
            else:
                self.log_issue("Supervisor", f"supervisorctl failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.log_issue("Supervisor", "supervisorctl command timeout")
        except FileNotFoundError:
            self.log_issue("Supervisor", "supervisorctl not found")
        except Exception as e:
            self.log_issue("Supervisor", f"Error checking supervisor: {str(e)}")

    def test_environment_configuration(self):
        """Test environment configuration files."""
        print("\n🔧 Testing Environment Configuration...")
        
        config_files = [
            ("/app/backend/.env", "Backend environment"),
            ("/app/frontend/.env", "Frontend environment"),
            ("/app/config.yaml", "System configuration")
        ]
        
        for config_path, description in config_files:
            try:
                if Path(config_path).exists():
                    with open(config_path, 'r') as f:
                        content = f.read()
                        if content.strip():
                            self.log_success(f"Config {config_path}", f"{description} exists and has content")
                        else:
                            self.log_issue(f"Config {config_path}", f"{description} exists but is empty")
                else:
                    self.log_issue(f"Config {config_path}", f"{description} file missing")
            except Exception as e:
                self.log_issue(f"Config {config_path}", f"Error reading {description}: {str(e)}")

    def test_log_files(self):
        """Test log file accessibility and recent activity."""
        print("\n🔧 Testing Log Files...")
        
        log_files = [
            "/var/log/supervisor/backend.err.log",
            "/var/log/supervisor/backend.out.log", 
            "/var/log/supervisor/frontend.err.log",
            "/var/log/supervisor/frontend.out.log"
        ]
        
        for log_path in log_files:
            try:
                if Path(log_path).exists():
                    # Check if log file has recent activity (within last hour)
                    stat = Path(log_path).stat()
                    age_seconds = time.time() - stat.st_mtime
                    
                    if age_seconds < 3600:  # 1 hour
                        self.log_success(f"Log {log_path}", "Exists and has recent activity")
                    else:
                        self.log_issue(f"Log {log_path}", "Exists but no recent activity", "WARNING")
                else:
                    self.log_issue(f"Log {log_path}", "Log file missing")
            except Exception as e:
                self.log_issue(f"Log {log_path}", f"Error checking log: {str(e)}")

    def test_frontend_backend_integration(self):
        """Test if frontend can communicate with backend."""
        print("\n🔧 Testing Frontend-Backend Integration...")
        
        try:
            # Check if REACT_APP_BACKEND_URL is properly configured
            env_path = Path("/app/frontend/.env")
            if env_path.exists():
                with open(env_path, 'r') as f:
                    content = f.read()
                    if "REACT_APP_BACKEND_URL" in content:
                        # Extract the backend URL
                        for line in content.split('\n'):
                            if line.startswith('REACT_APP_BACKEND_URL'):
                                backend_url = line.split('=')[1].strip()
                                self.log_success("Frontend Config", f"Backend URL configured: {backend_url}")
                                
                                # Test if this URL is accessible
                                try:
                                    test_url = f"{backend_url}/api/"
                                    response = requests.get(test_url, timeout=10)
                                    if response.status_code == 200:
                                        self.log_success("Frontend-Backend Integration", 
                                                       "Frontend can reach configured backend URL")
                                    else:
                                        self.log_issue("Frontend-Backend Integration", 
                                                     f"Configured backend URL not accessible: HTTP {response.status_code}")
                                except Exception as e:
                                    self.log_issue("Frontend-Backend Integration", 
                                                 f"Cannot reach configured backend URL: {str(e)}")
                                break
                        else:
                            self.log_issue("Frontend Config", "REACT_APP_BACKEND_URL found but value not extracted")
                    else:
                        self.log_issue("Frontend Config", "REACT_APP_BACKEND_URL not found in .env")
            else:
                self.log_issue("Frontend Config", "Frontend .env file missing")
                
        except Exception as e:
            self.log_issue("Frontend-Backend Integration", f"Error testing integration: {str(e)}")

    def test_dependencies(self):
        """Test if critical dependencies are installed."""
        print("\n🔧 Testing Dependencies...")
        
        try:
            # Test Python backend dependencies
            result = subprocess.run(['pip', 'list'], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                pip_output = result.stdout.lower()
                
                critical_deps = ['fastapi', 'uvicorn', 'pymongo', 'motor', 'groq', 'tiktoken', 'aiohttp']
                for dep in critical_deps:
                    if dep in pip_output:
                        self.log_success(f"Python Dependency", f"{dep} installed")
                    else:
                        self.log_issue(f"Python Dependency", f"{dep} missing")
            else:
                self.log_issue("Python Dependencies", f"pip list failed: {result.stderr}")
                
        except Exception as e:
            self.log_issue("Python Dependencies", f"Error checking dependencies: {str(e)}")
        
        try:
            # Test Node.js frontend dependencies
            result = subprocess.run(['yarn', 'list', '--depth=0'], 
                                  cwd='/app/frontend', capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                yarn_output = result.stdout.lower()
                
                critical_deps = ['react', 'axios', 'tailwindcss']
                for dep in critical_deps:
                    if dep in yarn_output:
                        self.log_success(f"Node.js Dependency", f"{dep} installed")
                    else:
                        self.log_issue(f"Node.js Dependency", f"{dep} missing")
            else:
                self.log_issue("Node.js Dependencies", f"yarn list failed: {result.stderr}")
                
        except Exception as e:
            self.log_issue("Node.js Dependencies", f"Error checking dependencies: {str(e)}")

    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE TEST REPORT")
        print("="*80)
        
        print(f"\n✅ PASSED TESTS: {len(self.passed_tests)}")
        for test in self.passed_tests:
            print(f"   ✅ {test['test']}: {test['message']}")
        
        print(f"\n❌ ISSUES FOUND: {len(self.issues)}")
        
        # Group issues by severity
        errors = [i for i in self.issues if i['severity'] == 'ERROR']
        warnings = [i for i in self.issues if i['severity'] == 'WARNING']
        
        if errors:
            print(f"\n🚨 CRITICAL ERRORS ({len(errors)}):")
            for issue in errors:
                print(f"   ❌ {issue['test']}: {issue['issue']}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS ({len(warnings)}):")
            for issue in warnings:
                print(f"   ⚠️  {issue['test']}: {issue['issue']}")
        
        print(f"\n📈 OVERALL HEALTH: {len(self.passed_tests)}/{len(self.passed_tests) + len(self.issues)} tests passed")
        
        # Priority recommendations
        if errors:
            print(f"\n🔧 IMMEDIATE ACTION REQUIRED:")
            print("   Fix critical errors to restore system functionality")
        elif warnings:
            print(f"\n🔧 RECOMMENDED ACTIONS:")
            print("   Address warnings to improve system reliability")
        else:
            print(f"\n🎉 SYSTEM STATUS: HEALTHY")
            print("   All tests passed successfully!")
        
        return len(errors) == 0  # Return True if no critical errors

def main():
    """Run comprehensive system tests."""
    print("🚀 Starting Comprehensive System Tests...")
    print("="*80)
    
    tester = SystemTester()
    
    # Run all test suites
    tester.test_supervisor_services()
    tester.test_backend_service()
    tester.test_frontend_service()
    tester.test_backend_endpoints()
    tester.test_database_connectivity()
    tester.test_environment_configuration()
    tester.test_frontend_backend_integration()
    tester.test_dependencies()
    tester.test_log_files()
    
    # Generate comprehensive report
    system_healthy = tester.generate_report()
    
    # Save detailed report to file
    report_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "passed_tests": tester.passed_tests,
        "issues": tester.issues,
        "system_healthy": system_healthy
    }
    
    with open("/app/test_report.json", "w") as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n💾 Detailed report saved to: /app/test_report.json")
    
    return system_healthy

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)