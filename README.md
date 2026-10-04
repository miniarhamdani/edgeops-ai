# EdgeOps AI — Distributed Edge/Cloud Anomaly Detection

A production-grade distributed monitoring system with local anomaly detection at the edge, centralized cloud backend, and LLM-powered incident explanations.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     EDGE LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Edge Node 1 │  │  Edge Node 2 │  │  Edge Node N │      │
│  │  (Sensors)   │  │  (Sensors)   │  │  (Sensors)   │      │
│  │  IsoForest   │  │  IsoForest   │  │  IsoForest   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼─────────────────┼─────────────────┼──────────────┘
          │  Anomalies Only │                 │
          └─────────────────▼─────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                     CLOUD LAYER                             │
│  ┌──────────────────┐    ┌─────────────────────────────┐   │
│  │   FastAPI Backend│    │       Streamlit Dashboard    │   │
│  │   + PostgreSQL   │◄──►│  Monitoring + Analytics +    │   │
│  │   + Ollama LLM   │    │  LLM Explanations           │   │
│  └──────────────────┘    └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Components

| Component | Technology | Role |
|-----------|-----------|------|
| Edge Nodes | Python + scikit-learn | Collect sensor data, detect local anomalies with Isolation Forest |
| Cloud Backend | FastAPI + PostgreSQL | Centralized event store, REST API, incident management |
| LLM Service | Ollama | Generate anomaly explanations & recommended actions |
| Dashboard | Streamlit | Real-time monitoring, analytics, incident handling |
| Infrastructure | Docker Compose | Orchestrate all services |

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Ollama (for LLM features)

### Run with Docker Compose

```bash
# Clone the project
git clone <repo-url> && cd edgeops-ai

# Copy environment file
cp .env.example .env

# Start all services
docker compose up -d

# (Optional) Pull Ollama model
docker exec edgeops-ollama ollama pull llama3.2

# Access the dashboard
open http://localhost:8501
```

### Run Locally (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Setup PostgreSQL
createdb edgeops

# Run cloud backend
cd cloud_backend && uvicorn main:app --reload --port 8000

# Run Streamlit dashboard
cd dashboard && streamlit run app.py

# Simulate edge nodes
cd edge_node && python simulator.py --nodes 5 --interval 5
```

## Project Structure

```
edgeops-ai/
├── edge_node/              # Edge node simulation
│   ├── sensor.py           # Synthetic sensor data generator
│   ├── anomaly_detector.py # Isolation Forest pipeline
│   ├── edge_agent.py       # Main edge agent loop
│   └── simulator.py        # Multi-node simulator CLI
├── cloud_backend/          # FastAPI backend
│   ├── main.py             # FastAPI app & routes
│   ├── models.py           # SQLAlchemy models
│   ├── database.py         # DB connection
│   ├── schemas.py          # Pydantic schemas
│   ├── llm_service.py      # Ollama integration
│   └── incident_handler.py # Automated incident logic
├── dashboard/              # Streamlit dashboard
│   ├── app.py              # Main dashboard
│   ├── pages/              # Multi-page app
│   │   ├── 1_📊_Overview.py
│   │   ├── 2_🚨_Incidents.py
│   │   ├── 3_📈_Analytics.py
│   │   └── 4_🤖_LLM_Insights.py
│   └── utils/              # Dashboard utilities
├── experiments/            # Latency & comm-volume experiments
│   ├── latency_benchmark.py
│   └── communication_volume.py
├── docker/                 # Dockerfiles
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Configuration

Key environment variables (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `llama3.2` | LLM model for explanations |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection |
| `EDGE_ANOMALY_THRESHOLD` | `0.5` | Isolation Forest contamination |
| `CLOUD_API_URL` | `http://localhost:8000` | Cloud backend URL |

## Experiments

```bash
# Run latency benchmark
python experiments/latency_benchmark.py

# Run communication volume analysis
python experiments/communication_volume.py
```
