FROM python:3.12-slim

# System dependencies for psycopg2 and dbt
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    dagster==1.6.13 \
    dagster-webserver==1.6.13 \
    dagster-postgres==0.22.13 \
    dagster-dbt==0.22.13 \
    dbt-core==1.7.18 \
    dbt-postgres==1.7.18 \
    psycopg2-binary==2.9.9

WORKDIR /app

# Copy configuration files first (better layer caching)
COPY dagster.yaml /app/dagster.yaml
COPY workspace.yaml /app/workspace.yaml

# Copy project code
COPY dagster_project/ /app/dagster_project/
COPY dbt_project/ /app/dbt_project/
COPY data/ /app/data/

# Set Dagster home so it finds dagster.yaml
ENV DAGSTER_HOME=/app

# Parse the dbt project so Dagster can load the manifest
# (This will be done at runtime via a startup script instead,
# since the DB may not be ready at build time)

EXPOSE 3000
