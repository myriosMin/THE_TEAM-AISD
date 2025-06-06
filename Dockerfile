# Stage 1: Base container with JupyterLab and Kedro
ARG BASE_IMAGE=python:3.10-slim
FROM ${BASE_IMAGE} AS runtime-environment

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        openjdk-17-jdk \
        build-essential \
        libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for PySpark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Update pip
RUN pip install --upgrade pip

# Clone the project repository
ARG GIT_REPO=https://github.com/myriosMin/THE_TEAM-AISD
ARG GIT_BRANCH=main
RUN git clone --branch $GIT_BRANCH $GIT_REPO /home/kedro_docker

# Set the working directory
WORKDIR /home/kedro_docker

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Make sure run.sh is executable (if present)
RUN chmod +x run.sh || true

# Add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker && \
    chown -R ${KEDRO_UID}:${KEDRO_GID} /home/kedro_docker

USER kedro_docker

# Expose port for JupyterLab
EXPOSE 8888

# Default command: launch JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser"]

# Stage 2: Runner container for pipeline execution
FROM runtime-environment AS runner
CMD ["./run.sh"]
