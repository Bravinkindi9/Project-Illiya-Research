<![CDATA[# 🧬 Urban DNA Sequencer

> AI-powered urban morphology analysis that decodes the *genetic structure* of cities using satellite building data and machine learning.

[![License: MIT](https://img.shields.io/badge/License-MIT-2dd4bf.svg)](LICENSE)
[![Backend: FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Frontend: Next.js](https://img.shields.io/badge/Frontend-Next.js_16-black.svg)](https://nextjs.org/)
[![Deploy: Railway + Vercel](https://img.shields.io/badge/Deploy-Railway_+_Vercel-6366f1.svg)](#deployment)

---

## 🎯 What It Does

Urban DNA Sequencer extracts morphological "DNA signatures" from satellite-detected buildings to **classify urban fabric** across African cities. It identifies:

| Category | Description | Risk Score |
|---|---|---|
| 🔴 **Informal / High Risk** | Small, dense, irregularly shaped buildings | 8 / 10 |
| 🟡 **Upgrading Zone** | Transitional areas showing improvement | 4 / 10 |
| 🟢 **Stable / Formal** | Larger, regular, well-spaced structures | 1 / 10 |

**Currently covering Kigali, Rwanda** — with 150,000+ buildings analysed at ~94% accuracy.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   FRONTEND (Vercel)                 │
│   Next.js 16 · React 19 · Deck.gl · MapLibre GL    │
│                                                     │
│   /              → Landing page                     │
│   /dashboard     → Interactive 3D map + analytics   │
│   /methodology   → Pipeline documentation           │
└────────────────────┬────────────────────────────────┘
                     │ /api/* (proxy via vercel.json)
┌────────────────────▼────────────────────────────────┐
│                 BACKEND (Railway)                    │
│   FastAPI · scikit-learn · pandas · numpy            │
│                                                     │
│   POST /api/analyze   → Classify buildings in bbox  │
│   POST /api/compare   → Compare two regions         │
│   GET  /api/cities    → List available datasets     │
│   GET  /api/export/:city → Download CSV / GeoJSON   │
│   GET  /api/health    → Health check                │
└─────────────────────────────────────────────────────┘
```

---

## 🔬 AI Pipeline

```
Satellite Imagery → Building Detection → Feature Engineering → K-Means Clustering → Risk Classification
```

1. **Satellite Acquisition** — Google Earth Engine + Open Buildings V3 (1.8B+ structures)
2. **Building Detection** — Polygons extracted at >0.7 confidence threshold
3. **Feature Engineering** — 7 morphological metrics: area, perimeter, shape index, elongation, NND, density, area_log
4. **AI Clustering** — StandardScaler → K-Means (K=3, Silhouette ≈ 0.48)
5. **Risk Classification** — Each cluster mapped to Informal / Upgrading / Stable

---

## 📁 Project Structure

```
Urban-DNA-informality-sequence/
├── backend/
│   ├── main.py              # FastAPI API server
│   └── requirements.txt     # Backend Python deps
├── urban-dna-frontend/
│   ├── app/
│   │   ├── page.tsx         # Landing page
│   │   ├── dashboard/
│   │   │   └── page.tsx     # Interactive 3D map dashboard
│   │   ├── methodology/
│   │   │   └── page.tsx     # Pipeline methodology page
│   │   ├── layout.tsx       # Root layout
│   │   └── globals.css      # Global styles
│   ├── public/
│   │   └── kigali_results_v2.csv  # Pre-computed Kigali data
│   ├── vercel.json          # Vercel deploy config + API proxy
│   ├── .env.example         # Environment variables template
│   └── package.json         # Frontend dependencies
├── apps/
│   ├── app.py               # Legacy Streamlit app
│   └── urban_dna_brain.pkl  # Trained KMeans model + scaler
├── Notebooks/
│   ├── Urban_DNA_Complete.ipynb
│   └── Urban_DNA_Sequencer_Phase1 (3).ipynb
├── feature.py               # Feature engineering pipeline
├── pipeline.py              # End-to-end data pipeline
├── kanombe_extraction.py    # Kanombe area data extraction
├── requirements.txt         # Root Python dependencies
├── Procfile                 # Railway process file
├── railway.toml             # Railway deploy config
├── PROJECT_ROADMAP.md       # Long-term project roadmap
└── MASTER_CHECKLIST.md      # Progress tracker
```

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.11+
- **Node.js** 18+
- **npm** 9+

### 1. Clone the Repository

```bash
git clone https://github.com/Bravinkindi9/Urban-DNA-informality-sequence.git
cd Urban-DNA-informality-sequence
```

### 2. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the FastAPI server
cd backend
uvicorn main:app --reload --port 8000
```

API docs available at: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Frontend Setup

```bash
# Install Node dependencies
cd urban-dna-frontend
npm install

# Create environment file
cp .env.example .env.local
# Edit .env.local → set NEXT_PUBLIC_API_URL=http://localhost:8000

# Start the dev server
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

---

## 🌐 Deployment

### Backend → Railway

The backend is deployed on [Railway](https://railway.app/) using the `Procfile` and `railway.toml`.

```
# Procfile
web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

**Live endpoint:** `https://web-production-d31bf.up.railway.app`

### Frontend → Vercel

The frontend is deployed on [Vercel](https://vercel.com/). API calls are proxied to Railway via `vercel.json`:

```json
{
  "rewrites": [
    {
      "source": "/api/:path*",
      "destination": "https://web-production-d31bf.up.railway.app/api/:path*"
    }
  ]
}
```

---

## 🧪 API Reference

### `POST /api/analyze`

Classify buildings within a bounding box.

```json
{
  "min_lon": 29.82,
  "min_lat": -2.00,
  "max_lon": 29.92,
  "max_lat": -1.90,
  "city": "kigali",
  "max_points": 2000
}
```

**Response:** Cluster distributions + per-building coordinates with risk scores.

### `POST /api/compare`

Compare two regions of interest side-by-side.

### `GET /api/cities`

List all cities with pre-computed datasets available.

### `GET /api/export/{city}?format=csv|geojson`

Download city data as CSV or GeoJSON.

### `GET /api/health`

Health check — confirms model is loaded and returns cluster info.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 16, React 19, TypeScript |
| **Mapping** | Deck.gl, MapLibre GL, react-map-gl |
| **Icons** | Lucide React |
| **Backend** | FastAPI, Uvicorn |
| **ML Model** | scikit-learn (KMeans), joblib |
| **Data** | pandas, NumPy |
| **Data Source** | Google Earth Engine, Open Buildings V3 |
| **Deployment** | Vercel (frontend), Railway (backend) |

---

## 📖 Use Cases

1. **Informal Settlement Detection** — Identify high-risk areas needing intervention
2. **Urban Growth Analysis** — Track how cities expand and densify over time
3. **Infrastructure Planning** — Target underserved neighborhoods for schools, clinics, roads
4. **Population Estimation** — Approximate population from building density
5. **Policy & Research** — Data-driven evidence for urban planning decisions

---

## 🗺️ Roadmap

See [PROJECT_ROADMAP.md](PROJECT_ROADMAP.md) for the full plan, including:

- 🏙️ Multi-city expansion (Nairobi, Lagos, Kampala, Dar es Salaam)
- 📅 Temporal analysis (2019 vs 2023 comparison)
- 📊 PDF report generation
- 🏥 Infrastructure gap analysis (distance to schools/clinics)
- 🔌 BigQuery + GEE live pipeline (replacing pre-computed CSVs)

---

## 🤝 Contributing

Contributions are welcome! All notebooks, model weights, and pipeline scripts are open source.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the **MIT License**.

---

<p align="center">
  <strong>🧬 Urban DNA Sequencer</strong><br/>
  <em>Decoding the genetic structure of African cities</em><br/><br/>
  Built with ❤️ by <a href="https://github.com/Bravinkindi9">Bravinkindi9</a>
</p>
]]>
