# 🏠 House Price Predictor - MLOps Pipeline

A complete, reproducible MLOps pipeline for predicting house prices using machine learning. This project demonstrates industry-standard practices for model development, tracking, containerization, and deployment using **MLflow** and **Docker**.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Architecture & Tools](#architecture--tools)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Training & MLflow Tracking](#training--mlflow-tracking)
- [Dockerization](#dockerization)
- [Deployment & API Serving](#deployment--api-serving)
- [API Testing](#api-testing)
- [MLflow UI](#mlflow-ui)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)
- [References](#references)

## 🚀 Project Overview

This project implements a full MLOps lifecycle for a house price prediction model, covering:

- **Experiment Tracking**: MLflow for parameter, metric, and artifact logging
- **Model Packaging**: MLflow Model format for reproducible packaging
- **Containerization**: Docker for environment consistency
- **Model Serving**: REST API with MLflow's built-in server
- **Governance**: MLflow Model Registry for versioning and aliasing

### Key Features
- ✅ Complete reproducibility across environments
- ✅ Automated dependency management
- ✅ Model versioning and lineage tracking
- ✅ Zero-downtime deployment via model aliases
- ✅ Input validation via model signatures
- ✅ Optimized Docker container for serving

## 🏗 Architecture & Tools

### Core Components
- **MLflow**: Experiment tracking, model registry, and serving
- **Docker**: Containerization for environment consistency
- **Scikit-learn**: Machine learning (LinearRegression)
- **Pandas**: Data manipulation

### MLOps Workflow
Data → Training → MLflow Tracking → Model Registry → Dockerization → API Serving → Monitoring


## 📁 Project Structure
house_price_mlops/
├── data/
│   └── housing.csv              # Sample training data
├── scripts/
│   └── train.py                 # Training script with MLflow integration
├── mlruns/                      # MLflow tracking data (auto-generated)
├── Dockerfile                   # Docker configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file


## ⚙️ Installation & Setup

### Prerequisites
- Python 3.8+
- Docker
- Git

### 1. Clone and Setup
# Clone repository
git clone <your-repo-url>
cd house_price_mlops

# Create virtual environment
python3 -m venv mlops-demo
source mlops-demo/bin/activate  # On Windows: mlops-demo\Scripts\activate

# Install dependencies
pip install -r requirements.txt

### 2. Verify MLflow Installation
mlflow --version

## 🧪 Training & MLflow Tracking

### 1. Prepare Data
The project includes sample data in data/housing.csv with features:
- area, bedrooms, bathrooms, stories, parking, price

### 2. Run Training Script
python scripts/train.py

### What Happens During Training:
1. **Data Loading**: CSV data is loaded and split (80/20 train/test)
2. **Model Training**: LinearRegression model is trained
3. **MLflow Logging**:
   - Parameters: test_size, model_type
   - Metrics: RMSE (Root Mean Squared Error)
   - Model Artifact: Saved to MLflow with signature
   - Model Registration: Registered as HousePricePredictor in Model Registry

### 3. View Training Results
# Start MLflow UI
mlflow ui

# Access at: http://localhost:5000


## 🐳 Dockerization

### 1. Build Docker Image
# Dockerfile content
FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

EXPOSE 5000

# Serve model using MLflow
CMD ["mlflow", "models", "serve", "-m", "models:/HousePricePredictor@champion", "--host", "0.0.0.0", "--port", "5000", "--env-manager", "local"]

Build the image:
docker build -t house-price-mlops .

### 2. Docker Build Optimization
- Uses slim Python base image
- Installs only required dependencies
- Leverages MLflow's local environment manager
- References model via registry alias

## 🚀 Deployment & API Serving

### 1. Run Docker Container
# Run container (map host port 8000 to container port 5000)
docker run -d -p 8000:5000 --name house-price-server house-price-mlops

# Verify container is running
docker ps

### 2. Container Management
# Stop container
docker stop house-price-server

# Start container
docker start house-price-server

# View logs
docker logs house-price-server

# Remove container
docker rm house-price-server


## 🔧 API Testing

### 1. Test API Endpoint
# Send prediction request
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "dataframe_split": {
      "columns": ["area", "bedrooms", "bathrooms", "stories", "parking"],
      "data": [[5000, 4, 2, 3, 2]]
    }
  }'


### 2. Expected Response
{
  "predictions": [650000]
}


### 3. API Specifications
- **Endpoint**: POST /invocations
- **Content-Type**: application/json
- **Format**: dataframe_split (MLflow standard)
- **Required Fields**: columns, data

## 📊 MLflow UI

### Access Tracking Interface
# Ensure Docker container on port 5000 is stopped first
docker stop house-price-server

# Start MLflow UI
mlflow ui

# Access: http://localhost:5000


### UI Features
- ✅ Experiment comparison
- ✅ Parameter and metric visualization
- ✅ Model artifact browsing
- ✅ Model Registry management
- ✅ Run filtering and searching

## 🐛 Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Port 5000 already in use** | Stop conflicting service or use different port: mlflow ui --port 5001 |
| **Docker build fails** | Ensure Docker is running and you have sufficient disk space |
| **Model not found in registry** | Check model registration in MLflow UI |
| **API returns 400 Bad Request** | Verify JSON format uses dataframe_split |
| **Missing dependencies in container** | Rebuild with updated requirements.txt |
| **Slow container startup** | Use --env-manager=local flag |

### Debug Commands
# Check container logs
docker logs house-price-server

# Check MLflow model registration
mlflow models list

# Test API connectivity
curl http://localhost:8000/health


## 🔮 Future Enhancements

### Planned Features
1. **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
2. **Monitoring**: Integration with Prometheus/Grafana for performance tracking
3. **Data Drift Detection**: Automated detection of distribution changes
4. **A/B Testing**: Canary deployments for model comparison
5. **Multi-model Support**: Support for ensemble and multiple models
6. **Kubernetes Deployment**: Scalable orchestration with K8s

### MLOps Maturity Roadmap
- [x] Basic reproducibility with Docker
- [x] Model versioning with MLflow
- [ ] Automated retraining pipeline
- [ ] Comprehensive monitoring
- [ ] Governance and compliance tracking

## 📚 References

### Documentation
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Docker Documentation](https://docs.docker.com/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

### Key Concepts
1. **Reproducibility**: Ensuring consistent results across environments
2. **Model Lineage**: Tracking data, code, and parameters for each model version
3. **Containerization**: Isolating application environments
4. **Model Serving**: Deploying models as scalable APIs
5. **MLOps Governance**: Managing model lifecycle and compliance

### Academic References
- MLOps lifecycle principles (ProjectPro, 2025)
- Docker for AI model deployment (Runpod, 2025)
- MLflow model management (Microsoft Learn, 2025)

## 👥 Contributing

1. Fork the repository
2. Create a feature branch (git checkout -b feature/AmazingFeature)
3. Commit changes (git commit -m 'Add AmazingFeature')
4. Push to branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MLflow team for excellent MLOps tooling
- Docker community for containerization standards
- Open-source ML community for best practices

---

**Note**: This project is for educational and demonstration purposes. For production use, consider additional security, monitoring, and scalability measures.

---
*Last Updated: February 2025*  
*MLflow Version: 2.0+*  
*Docker Version: 20.10+*
