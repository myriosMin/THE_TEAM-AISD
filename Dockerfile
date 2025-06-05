ARG BASE_IMAGE=python:3.10-slim
FROM $BASE_IMAGE AS runtime-environment

# Install system dependencies: git, Java 17
RUN apt-get update && \
    apt-get install -y git openjdk-17-jdk && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for Spark
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Update pip
RUN python -m pip install --upgrade pip

# Clone the repository
ARG GIT_REPO=https://github.com/myriosMin/THE_TEAM-AISD
ARG GIT_BRANCH=main
RUN git clone --branch $GIT_BRANCH $GIT_REPO /home/kedro_docker

# Set workdir to the cloned project
WORKDIR /home/kedro_docker

# install project requirements
RUN pip install --no-cache-dir -r requirements.txt

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker && \
    mkdir -p /home/kedro_docker/.local/share/jupyter/runtime && \
    chown -R ${KEDRO_UID}:${KEDRO_GID} /home/kedro_docker/.local

USER kedro_docker

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

