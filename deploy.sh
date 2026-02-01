#!/bin/bash
# DubYou Enterprise Deployment Script

set -e

echo "======================================"
echo "DubYou Enterprise Deployment"
echo "======================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    log_info "✓ Docker found"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    log_info "✓ Docker Compose found"
    
    # Check NVIDIA Docker (optional)
    if command -v nvidia-smi &> /dev/null; then
        log_info "✓ NVIDIA GPU detected"
    else
        log_warn "NVIDIA GPU not detected - CPU mode will be used"
    fi
}

setup_environment() {
    log_info "Setting up environment..."
    
    if [ ! -f .env ]; then
        log_info "Creating .env file from template..."
        cp .env.example .env
        log_warn "Please edit .env file with your configuration"
    else
        log_info "✓ .env file exists"
    fi
    
    # Create required directories
    mkdir -p data/audio data/embeddings logs models
    log_info "✓ Directories created"
}

download_models() {
    log_info "Checking models..."
    
    if [ -d "models/whisper" ] && [ -d "models/nllb" ]; then
        log_info "✓ Models already downloaded"
    else
        log_info "Downloading models (this may take a while)..."
        python scripts/download_models.py
    fi
}

build_containers() {
    log_info "Building Docker containers..."
    docker-compose build --parallel
    log_info "✓ Containers built successfully"
}

start_services() {
    log_info "Starting services..."
    docker-compose up -d
    log_info "✓ Services started"
    
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Check health
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_info "✓ Application is healthy"
    else
        log_error "Application health check failed"
        docker-compose logs dubyou-app
        exit 1
    fi
}

show_info() {
    echo ""
    echo "======================================"
    log_info "Deployment completed successfully!"
    echo "======================================"
    echo ""
    echo "Services:"
    echo "  - API: http://localhost:8000"
    echo "  - Health: http://localhost:8000/health"
    echo "  - Metrics: http://localhost:9090"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo ""
    echo "Useful commands:"
    echo "  - View logs: docker-compose logs -f"
    echo "  - Stop services: docker-compose down"
    echo "  - Restart: docker-compose restart"
    echo ""
}

# Main execution
main() {
    check_requirements
    setup_environment
    download_models
    build_containers
    start_services
    show_info
}

# Parse arguments
case "${1:-}" in
    start)
        log_info "Starting services..."
        docker-compose up -d
        ;;
    stop)
        log_info "Stopping services..."
        docker-compose down
        ;;
    restart)
        log_info "Restarting services..."
        docker-compose restart
        ;;
    logs)
        docker-compose logs -f
        ;;
    clean)
        log_warn "This will remove all containers and volumes"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down -v
            log_info "Cleaned up"
        fi
        ;;
    *)
        main
        ;;
esac
