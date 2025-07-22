# 🤖 AI Research Assistant

[![CI/CD Pipeline](https://github.com/your-username/ai-research-assistant/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/your-username/ai-research-assistant/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/your-username/ai-research-assistant/branch/main/graph/badge.svg)](https://codecov.io/gh/your-username/ai-research-assistant)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 16+](https://img.shields.io/badge/node.js-16+-green.svg)](https://nodejs.org/)

> **Production-ready multi-agent AI research assistant powered by CrewAI and Groq's free Llama models**

A sophisticated AI research assistant that leverages three specialized agents (Researcher, Analyst, Writer) to conduct comprehensive research, analyze findings, and produce professional reports. Built with modern technologies and production-grade architecture.

## 🌟 **Features**

### 🤖 **Multi-Agent Intelligence**
- **Researcher Agent**: Conducts thorough research using multiple sources
- **Analyst Agent**: Performs strategic analysis and pattern recognition  
- **Writer Agent**: Creates well-structured, professional content

### 🚀 **Production Architecture**
- **FastAPI Backend** with async/await support
- **React Frontend** with modern hooks and routing
- **MongoDB Database** with optimized schemas and indexing
- **Redis Caching** for improved performance
- **Docker Containerization** for consistent deployments

### 🔧 **Developer Experience**
- **Comprehensive Testing** (unit, integration, e2e)
- **Code Quality Tools** (linting, formatting, type checking)
- **CI/CD Pipeline** with GitHub Actions
- **Development Scripts** and automation
- **API Documentation** with OpenAPI/Swagger

### 🔒 **Security & Monitoring**
- **Security Scanning** with Bandit and Trivy
- **Rate Limiting** and CORS protection
- **Health Checks** and monitoring
- **Error Tracking** and logging

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.9+ 
- Node.js 16+
- Docker & Docker Compose (recommended)
- Groq API Key (free at [console.groq.com](https://console.groq.com))

### **1. Clone & Setup**
```bash
git clone https://github.com/your-username/ai-research-assistant.git
cd ai-research-assistant

# Automated setup (recommended)
chmod +x scripts/setup.sh
./scripts/setup.sh

# Or use Make
make install
```

### **2. Configure Environment**
```bash
# Copy environment template
cp .env.example backend/.env

# Add your Groq API key to backend/.env
GROQ_API_KEY="your_groq_api_key_here"
```

### **3. Start Development Environment**

**Option A: Docker (Recommended)**
```bash
# Start all services
make dev
# or
docker-compose up -d

# View logs
make logs
```

**Option B: Native Development**
```bash
# Start backend
cd backend && uvicorn server:app --reload

# Start frontend (new terminal)
cd frontend && yarn start
```

### **4. Access the Application**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8001/api
- **API Documentation**: http://localhost:8001/docs
- **MongoDB**: localhost:27017
- **Redis**: localhost:6379

## 📋 **Development Commands**

### **Using Make (Recommended)**
```bash
make help              # Show all available commands
make dev               # Start development environment  
make test              # Run all tests
make lint              # Run code quality checks
make format            # Format code
make build             # Build production images
make clean             # Clean up environment
```

## 🧪 **Testing Strategy**

### **Test Commands**
```bash
# Run specific test types
make test-backend      # Backend unit & integration tests
make test-frontend     # Frontend component tests  
make test-e2e          # End-to-end tests
make test-coverage     # Generate coverage reports
```

## 🚀 **Deployment**

### **Production Build**
```bash
# Build production images
make build

# Or with Docker
docker build --target production -t ai-research-assistant:latest .
```

## 📞 **Support**

- **Documentation**: [Full documentation](https://your-username.github.io/ai-research-assistant/)
- **Issues**: [GitHub Issues](https://github.com/your-username/ai-research-assistant/issues)

**Built with ❤️ by the AI Research Team**
