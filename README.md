# RoadScout (HackKU 2026)

RoadScout is a road-damage reporting app that combines:
- A local PyTorch classifier for road-damage type and severity
- A Next.js web app for upload, geolocation, map display, and report history
- Gemini-generated reporting summaries and authority recommendations
- MongoDB storage for uploaded photos and analysis results

## Repository Structure

- `web/`: Next.js frontend + API routes
- `predict.py`: CLI image inference script used by the web API
- `app.py`: Streamlit demo for quick standalone model testing
- `train_baseline.py`: Model training script
- `prepare_rdd_imagefolder.py`: Dataset conversion utility
- `artifacts_new_design/`: Trained model artifacts and label metadata

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Node.js 18+ and npm
- MongoDB database (local or Atlas)

## 1) Python Environment Setup (project root)

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Web Dependencies Setup

Install web dependencies from the project root:

```powershell
npm install
```

This delegates to the `web/` app via root scripts.

## 3) Environment Variables

Create `web/.env.local` and set the following keys:

```env
MONGODB_URI=your_mongodb_connection_string
MONGODB_DB=your_database_name
GEMINI_API=your_gemini_api_key
NOMINATIM_CONTACT_API=your_contact_email_or_identifier
```

Notes:
- `MONGODB_DB` can also be provided as `MONGO_DB_NAME`.
- `GEMINI_API_KEY` is also accepted instead of `GEMINI_API`.
- `NOMINATIM_CONTACT_EMAIL` is also accepted instead of `NOMINATIM_CONTACT_API`.
- The API route attempts to use `.venv\Scripts\python.exe` automatically for local classifier execution.

## 4) Run the App Locally

From the project root:

```powershell
npm run dev
```

Open:
- `http://localhost:3000`

## 5) Quick Inference Test (CLI)

From the project root, with virtual environment activated:

```powershell
python predict.py --image path\to\road_image.jpg
```

Expected output includes:
- `prediction`
- `confidence`
- `severity`
- `recommended_action`

## 6) Streamlit Demo (Optional)

```powershell
streamlit run app.py
```

## 7) Train a Model (Optional)

If you want to retrain instead of using existing artifacts:

1. Prepare dataset in ImageFolder format (or convert from RDD labels):

```powershell
python prepare_rdd_imagefolder.py --source archive\RDD_SPLIT --output data\rdd_imagefolder --splits train val
```

2. Train baseline model:

```powershell
python train_baseline.py --data_dir data\rdd_imagefolder\train --val_dir data\rdd_imagefolder\val --output_dir artifacts_new_design
```

## Common Issues

- `Missing MONGODB_URI` or `Missing MONGODB_DB`:
	Ensure env vars are set in `web/.env.local` before running `npm run dev`.
- `Local classifier failed`:
	Confirm `.venv` exists, dependencies are installed, and `artifacts_new_design/baseline_model.pt` plus `artifacts_new_design/labels.json` are present.
- Geocoding errors:
	Verify network access to OpenStreetMap Nominatim and provide `NOMINATIM_CONTACT_API` or `NOMINATIM_CONTACT_EMAIL`.

## Scripts

From project root:

```powershell
npm run dev
npm run build
npm run start
npm run lint
```

## License

See `LICENSE`.
