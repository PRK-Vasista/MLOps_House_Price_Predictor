# Stage 1: Base Image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from the project root into the container's /app directory
# This copies the 'mlruns' folder containing your model artifact
COPY . /app

# --- Dependency Installation (Build Phase) ---
# Install necessary packages for model serving.
RUN echo "--- Installing required Python packages ---" && \
    pip install --no-cache-dir mlflow scikit-learn pandas numpy && \
    echo "--- Package installation complete ---"

# --- Configuration ---
# Expose the default port used by mlflow models serve
EXPOSE 5000

# --- Model Serving (Run Phase) ---
# Define the command to run when the container starts.
# NOTE: The Run ID is hardcoded here from your successful experiment.
#CMD ["/bin/bash", "-c", "echo '--- Starting MLflow Model Server ---' && \
 #   export MLFLOW_TRACKING_URI=./mlruns/ && \
  #  mlflow models serve \
   #     -m 'runs:/860ef31c96c545639faeedae30fd43a5/model' \
    ##   --no-conda \
      #  --host 0.0.0.0 \
    #&& echo '--- Model Server process ended ---'"]
    
# --- Model Serving (Run Phase) ---
# NOTE: Removed the MLFLOW_TRACKING_URI export and use the direct file path.
CMD ["/bin/bash", "-c", "echo '--- Starting MLflow Model Server (Direct Path) ---' && \
    mlflow models serve \
        -m '/app/mlruns/0/models/m-9a2b7bfa08ab4d73a6801bb8ee89017e/artifacts' \
        -p 5000 \
        --no-conda \
        --host 0.0.0.0 \
    && echo '--- Model Server process ended ---'"]
