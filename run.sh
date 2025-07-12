#!/bin/bash

# Script chạy RAG Chatbot
# Sử dụng: ./run.sh [mode]
# Modes: dev, prod, docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip &> /dev/null; then
        print_error "pip is not installed"
        exit 1
    fi
    
    # Check credentials.json
    if [ ! -f "credentials.json" ]; then
        print_warning "credentials.json not found"
        print_status "Please download credentials.json from Google Cloud Console"
    fi
    
    # Check .env file
    if [ ! -f ".env" ]; then
        print_warning ".env file not found"
        print_status "Creating .env template..."
        cat > .env << EOF
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_DRIVE_FOLDER_ID=
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_SEARCH_RESULTS=5
MAX_FILE_SIZE_MB=50
EOF
        print_status "Please edit .env file with your configuration"
    fi
    
    print_success "Prerequisites checked"
}

# Install dependencies
install_dependencies() {
    print_status "Installing dependencies..."
    
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    pip install -r requirements.txt
    
    print_success "Dependencies installed"
}

# Run in development mode
run_dev() {
    print_status "Starting in development mode..."
    check_prerequisites
    install_dependencies
    
    source venv/bin/activate
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    
    print_success "Starting Streamlit development server..."
    streamlit run app.py --server.port=8501 --server.address=localhost
}

# Run in production mode
run_prod() {
    print_status "Starting in production mode..."
    check_prerequisites
    install_dependencies
    
    source venv/bin/activate
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    
    print_success "Starting Streamlit production server..."
    streamlit run app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.fileWatcherType=none
}

# Run with Docker
run_docker() {
    print_status "Starting with Docker..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check docker-compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed"
        exit 1
    fi
    
    # Check .env
    if [ ! -f ".env" ]; then
        print_error ".env file is required for Docker mode"
        exit 1
    fi
    
    print_status "Building and starting containers..."
    docker-compose up --build
}

# Setup function
setup() {
    print_status "Setting up RAG Chatbot..."
    check_prerequisites
    install_dependencies
    
    # Create necessary directories
    mkdir -p cache vector_store logs
    
    print_success "Setup completed!"
    print_status "Next steps:"
    echo "1. Edit .env file with your API keys"
    echo "2. Add credentials.json from Google Cloud Console"
    echo "3. Run: ./run.sh dev"
}

# Test function
test() {
    print_status "Running tests..."
    source venv/bin/activate
    
    # Basic import test
    python3 -c "
import google.generativeai as genai
import streamlit as st
from langchain.embeddings import GoogleGenerativeAIEmbeddings
print('✅ All imports successful')
"
    
    # Test Gemini API if key is available
    if [ -f ".env" ]; then
        source .env
        if [ ! -z "$GEMINI_API_KEY" ] && [ "$GEMINI_API_KEY" != "your_gemini_api_key_here" ]; then
            python3 -c "
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

try:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content('Hello')
    print('✅ Gemini API connection successful')
except Exception as e:
    print(f'❌ Gemini API test failed: {e}')
"
        fi
    fi
    
    print_success "Tests completed"
}

# Clean function
clean() {
    print_status "Cleaning up..."
    
    # Remove cache
    rm -rf cache/*
    rm -rf vector_store/*
    rm -rf logs/*
    rm -f token.pickle
    
    # Docker cleanup
    if command -v docker &> /dev/null; then
        docker-compose down --volumes --remove-orphans 2>/dev/null || true
        docker system prune -f 2>/dev/null || true
    fi
    
    print_success "Cleanup completed"
}

# Update function
update() {
    print_status "Updating dependencies..."
    
    source venv/bin/activate
    pip install --upgrade -r requirements.txt
    
    print_success "Update completed"
}

# Help function
show_help() {
    echo "RAG Chatbot Runner"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  dev      Run in development mode (default)"
    echo "  prod     Run in production mode"
    echo "  docker   Run with Docker"
    echo "  setup    Initial setup"
    echo "  test     Run tests"
    echo "  clean    Clean cache and temporary files"
    echo "  update   Update dependencies"
    echo "  help     Show this help"
    echo ""
    echo "Examples:"
    echo "  $0              # Run in development mode"
    echo "  $0 dev          # Run in development mode"
    echo "  $0 prod         # Run in production mode"
    echo "  $0 docker       # Run with Docker"
    echo "  $0 setup        # Initial setup"
}

# Main logic
MODE=${1:-dev}

case $MODE in
    dev)
        run_dev
        ;;
    prod)
        run_prod
        ;;
    docker)
        run_docker
        ;;
    setup)
        setup
        ;;
    test)
        test
        ;;
    clean)
        clean
        ;;
    update)
        update
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $MODE"
        show_help
        exit 1
        ;;
esac