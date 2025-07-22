#!/bin/bash

# =============================================================================
# AI RESEARCH ASSISTANT - DEVELOPMENT SCRIPT
# =============================================================================
# This script provides convenient commands for development

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Start development servers
start_dev() {
    log_info "Starting development environment..."
    
    # Check if .env files exist
    if [ ! -f "backend/.env" ]; then
        log_warning "Backend .env file not found. Creating from template..."
        cp .env.example backend/.env
    fi
    
    if [ ! -f "frontend/.env" ]; then
        log_warning "Frontend .env file not found. Creating..."
        echo "REACT_APP_BACKEND_URL=http://localhost:8001" > frontend/.env
    fi
    
    # Start with Docker Compose
    docker-compose up -d --build
    
    log_success "Development environment started!"
    log_info "Services available at:"
    echo "  - Frontend: http://localhost:3000"
    echo "  - Backend API: http://localhost:8001/api"
    echo "  - API Docs: http://localhost:8001/docs"
    echo "  - MongoDB: localhost:27017"
    echo "  - Redis: localhost:6379"
}

# Stop development servers
stop_dev() {
    log_info "Stopping development environment..."
    docker-compose down
    log_success "Development environment stopped!"
}

# Restart development servers
restart_dev() {
    log_info "Restarting development environment..."
    docker-compose restart
    log_success "Development environment restarted!"
}

# View logs
logs() {
    local service=${1:-}
    if [ -n "$service" ]; then
        docker-compose logs -f "$service"
    else
        docker-compose logs -f
    fi
}

# Run tests
test() {
    local test_type=${1:-all}
    
    case $test_type in
        backend)
            log_info "Running backend tests..."
            docker-compose run --rm backend python -m pytest tests/ -v
            ;;
        frontend)
            log_info "Running frontend tests..."
            docker-compose run --rm frontend yarn test --watchAll=false
            ;;
        e2e)
            log_info "Running e2e tests..."
            docker-compose --profile testing up --build --exit-code-from e2e-tests e2e-tests
            ;;
        all)
            log_info "Running all tests..."
            test backend
            test frontend
            test e2e
            ;;
        *)
            log_warning "Unknown test type: $test_type"
            echo "Available types: backend, frontend, e2e, all"
            ;;
    esac
}

# Code quality checks
lint() {
    log_info "Running code quality checks..."
    
    # Backend linting
    log_info "Checking backend code..."
    docker-compose run --rm backend black --check .
    docker-compose run --rm backend isort --check-only .
    docker-compose run --rm backend flake8 .
    docker-compose run --rm backend mypy backend/
    
    # Frontend linting
    log_info "Checking frontend code..."
    docker-compose run --rm frontend yarn lint
    
    log_success "Code quality checks completed!"
}

# Format code
format() {
    log_info "Formatting code..."
    
    # Backend formatting
    docker-compose run --rm backend black .
    docker-compose run --rm backend isort .
    
    # Frontend formatting
    docker-compose run --rm frontend yarn format
    
    log_success "Code formatting completed!"
}

# Database operations
db() {
    local action=${1:-status}
    
    case $action in
        status)
            log_info "Database status:"
            docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')"
            ;;
        shell)
            log_info "Opening database shell..."
            docker-compose exec mongodb mongosh ai_research_assistant
            ;;
        reset)
            log_warning "Resetting database..."
            docker-compose down mongodb
            docker volume rm $(docker-compose config --volumes | grep mongodb) 2>/dev/null || true
            docker-compose up -d mongodb
            log_success "Database reset completed!"
            ;;
        backup)
            local backup_file="backup-$(date +%Y%m%d-%H%M%S).gz"
            log_info "Creating database backup: $backup_file"
            docker-compose exec mongodb mongodump --db ai_research_assistant --gzip --archive=/tmp/backup.gz
            docker cp $(docker-compose ps -q mongodb):/tmp/backup.gz "./$backup_file"
            log_success "Database backup created: $backup_file"
            ;;
        restore)
            local backup_file=$2
            if [ -z "$backup_file" ] || [ ! -f "$backup_file" ]; then
                log_warning "Please provide a valid backup file"
                echo "Usage: $0 db restore <backup-file>"
                exit 1
            fi
            log_info "Restoring database from: $backup_file"
            docker cp "$backup_file" $(docker-compose ps -q mongodb):/tmp/restore.gz
            docker-compose exec mongodb mongorestore --db ai_research_assistant --gzip --archive=/tmp/restore.gz
            log_success "Database restore completed!"
            ;;
        *)
            echo "Available database operations:"
            echo "  status  - Check database status"
            echo "  shell   - Open database shell"
            echo "  reset   - Reset database (WARNING: destroys all data)"
            echo "  backup  - Create database backup"
            echo "  restore - Restore from backup file"
            ;;
    esac
}

# Build production images
build() {
    log_info "Building production images..."
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml build
    log_success "Production images built!"
}

# Deploy to staging
deploy_staging() {
    log_info "Deploying to staging..."
    # Add your staging deployment logic here
    log_warning "Staging deployment not configured yet"
}

# Clean up
clean() {
    log_info "Cleaning up development environment..."
    docker-compose down -v --remove-orphans
    docker system prune -f
    log_success "Cleanup completed!"
}

# Show help
show_help() {
    echo "AI Research Assistant Development Script"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start           Start development environment"
    echo "  stop            Stop development environment"
    echo "  restart         Restart development environment"
    echo "  logs [service]  View logs (optionally for specific service)"
    echo "  test [type]     Run tests (backend|frontend|e2e|all)"
    echo "  lint            Run code quality checks"
    echo "  format          Format code"
    echo "  db <action>     Database operations (status|shell|reset|backup|restore)"
    echo "  build           Build production images"
    echo "  deploy-staging  Deploy to staging environment"
    echo "  clean           Clean up development environment"
    echo "  help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 logs backend"
    echo "  $0 test backend"
    echo "  $0 db backup"
}

# Main command handler
case "${1:-}" in
    start)
        start_dev
        ;;
    stop)
        stop_dev
        ;;
    restart)
        restart_dev
        ;;
    logs)
        logs "$2"
        ;;
    test)
        test "$2"
        ;;
    lint)
        lint
        ;;
    format)
        format
        ;;
    db)
        db "$2" "$3"
        ;;
    build)
        build
        ;;
    deploy-staging)
        deploy_staging
        ;;
    clean)
        clean
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo "Unknown command: ${1:-}"
        echo "Run '$0 help' for available commands"
        exit 1
        ;;
esac