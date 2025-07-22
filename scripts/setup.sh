#!/bin/bash

# =============================================================================
# AI RESEARCH ASSISTANT - SETUP SCRIPT
# =============================================================================
# This script sets up the development environment for the AI Research Assistant

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        log_error "Please don't run this script as root"
        exit 1
    fi
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [ "$(printf '%s\n' "3.9" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.9" ]; then
        log_error "Python 3.9 or higher is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed"
        exit 1
    fi
    
    NODE_VERSION=$(node --version | sed 's/v//' | cut -d'.' -f1)
    if [ "$NODE_VERSION" -lt 16 ]; then
        log_error "Node.js 16 or higher is required"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_warning "Docker is not installed. Container features will not be available."
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_warning "Docker Compose is not installed. Container features will not be available."
    fi
    
    log_success "System requirements check passed"
}

# Setup Python environment
setup_python() {
    log_info "Setting up Python environment..."
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    # Install dependencies
    log_info "Installing Python dependencies..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi
    
    # Install development dependencies
    if [ -f "pyproject.toml" ]; then
        pip install -e ".[dev,test]"
    fi
    
    log_success "Python environment setup completed"
}

# Setup Node.js environment
setup_nodejs() {
    log_info "Setting up Node.js environment..."
    
    cd frontend
    
    # Check if yarn is installed
    if ! command -v yarn &> /dev/null; then
        log_info "Installing Yarn..."
        npm install -g yarn
    fi
    
    # Install dependencies
    log_info "Installing Node.js dependencies..."
    yarn install
    
    cd ..
    
    log_success "Node.js environment setup completed"
}

# Setup environment files
setup_env() {
    log_info "Setting up environment files..."
    
    # Backend environment
    if [ ! -f "backend/.env" ]; then
        log_info "Creating backend .env file..."
        cp .env.example backend/.env
        log_warning "Please update backend/.env with your actual configuration"
    fi
    
    # Frontend environment
    if [ ! -f "frontend/.env" ]; then
        log_info "Creating frontend .env file..."
        echo "REACT_APP_BACKEND_URL=http://localhost:8001" > frontend/.env
    fi
    
    log_success "Environment files setup completed"
}

# Setup database
setup_database() {
    log_info "Setting up database..."
    
    if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
        log_info "Starting MongoDB with Docker..."
        docker-compose up -d mongodb
        
        # Wait for MongoDB to be ready
        log_info "Waiting for MongoDB to be ready..."
        timeout 60 bash -c 'until docker-compose exec mongodb mongosh --eval "db.adminCommand(\"ping\")" &> /dev/null; do sleep 2; done'
        
        log_success "MongoDB is ready"
    else
        log_warning "Docker not available. Please ensure MongoDB is installed and running on localhost:27017"
    fi
}

# Setup pre-commit hooks
setup_hooks() {
    log_info "Setting up pre-commit hooks..."
    
    if [ -d "venv" ]; then
        source venv/bin/activate
        if command -v pre-commit &> /dev/null; then
            pre-commit install
            log_success "Pre-commit hooks installed"
        else
            log_warning "pre-commit not installed. Run 'pip install pre-commit' to enable git hooks"
        fi
    fi
}

# Create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p logs
    mkdir -p data
    mkdir -p tests/unit
    mkdir -p tests/integration
    mkdir -p tests/e2e
    mkdir -p docs
    
    log_success "Directories created"
}

# Run initial tests
run_tests() {
    log_info "Running initial tests..."
    
    if [ -d "venv" ]; then
        source venv/bin/activate
        
        # Run backend tests
        if [ -d "tests" ]; then
            log_info "Running backend tests..."
            python -m pytest tests/ -v --tb=short
        fi
    fi
    
    # Run frontend tests
    if [ -d "frontend/src" ]; then
        log_info "Running frontend tests..."
        cd frontend
        yarn test --watchAll=false --coverage --silent
        cd ..
    fi
    
    log_success "Initial tests completed"
}

# Main setup function
main() {
    log_info "Starting AI Research Assistant setup..."
    
    check_root
    check_requirements
    create_directories
    setup_env
    setup_python
    setup_nodejs
    setup_database
    setup_hooks
    
    if [ "$1" != "--skip-tests" ]; then
        run_tests
    fi
    
    log_success "Setup completed successfully!"
    echo ""
    log_info "Next steps:"
    echo "  1. Update backend/.env with your API keys"
    echo "  2. Start the development servers:"
    echo "     - Backend: cd backend && uvicorn server:app --reload"
    echo "     - Frontend: cd frontend && yarn start"
    echo "  3. Or use Docker: docker-compose up -d"
    echo ""
    log_info "Access the application at:"
    echo "  - Frontend: http://localhost:3000"
    echo "  - Backend API: http://localhost:8001/api"
    echo "  - API Documentation: http://localhost:8001/docs"
}

# Parse command line arguments
case "${1:-}" in
    --help|-h)
        echo "AI Research Assistant Setup Script"
        echo ""
        echo "Usage: $0 [options]"
        echo ""
        echo "Options:"
        echo "  --skip-tests    Skip running initial tests"
        echo "  --help, -h      Show this help message"
        exit 0
        ;;
    *)
        main "$@"
        ;;
esac