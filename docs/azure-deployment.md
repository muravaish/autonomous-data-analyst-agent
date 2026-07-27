# Azure Deployment Guide

This project is Azure-ready through a containerized Streamlit app and a database adapter that supports Azure Database for PostgreSQL.

## Recommended Azure Architecture

```text
GitHub Repository
   |
   | GitHub Actions builds Docker image
   v
Azure Container Registry
   |
   | Azure Container Apps pulls image
   v
Streamlit App Container
   |
   | DATABASE_URL
   v
Azure Database for PostgreSQL
```

## Azure Services Used

| Service | Purpose |
|---|---|
| Azure Container Registry | Stores the Docker image built from this repository |
| Azure Container Apps | Runs the Streamlit application as a container |
| Azure Database for PostgreSQL | Hosts the e-commerce analytics database |
| GitHub Actions | CI/CD pipeline for validation and manual deployment |

## Environment Variables

Local SQLite mode:

```text
GOOGLE_API_KEY=your_key_here
DATABASE_BACKEND=sqlite
SQLITE_DB_PATH=olist.db
DATABASE_SCHEMA=public
```

Azure PostgreSQL mode:

```text
GOOGLE_API_KEY=your_key_here
DATABASE_BACKEND=postgres
DATABASE_URL=postgresql://username:password@server.postgres.database.azure.com:5432/database?sslmode=require
DATABASE_SCHEMA=public
```

## GitHub Secrets Needed For Deployment

Add these in GitHub repository settings under **Settings > Secrets and variables > Actions**:

| Secret | Meaning |
|---|---|
| `AZURE_CREDENTIALS` | Azure service principal JSON used by `azure/login` |
| `ACR_LOGIN_SERVER` | Azure Container Registry login server, for example `myregistry.azurecr.io` |
| `ACR_USERNAME` | Registry username |
| `ACR_PASSWORD` | Registry password |
| `AZURE_CONTAINER_APP_NAME` | Name of the Azure Container App |
| `AZURE_RESOURCE_GROUP` | Azure resource group name |

The app secrets `GOOGLE_API_KEY` and `DATABASE_URL` should be stored as Azure Container App secrets named:

```text
google-api-key
database-url
```

## Deployment Flow

1. Load the Olist tables into Azure Database for PostgreSQL.
2. Create Azure Container Registry.
3. Create Azure Container Apps environment and container app.
4. Add GitHub deployment secrets.
5. Run the manual GitHub Actions workflow: `Deploy Container To Azure`.
6. Open the Azure Container App public URL.

## Local Docker Run

Build the image:

```powershell
docker build -t autonomous-data-analyst-agent .
```

Run with local SQLite:

```powershell
docker run --env-file .env -p 8501:8501 -v ${PWD}/olist.db:/app/olist.db:ro autonomous-data-analyst-agent
```

Open:

```text
http://localhost:8501
```

## Why This Matters For The Portfolio

This shows the project is not only a notebook or local demo. It has a path to production using cloud-hosted data, containerized deployment, environment-based configuration, and CI/CD automation.
