# RoadScout
Submitted to HackKU 2026: https://devpost.com/software/hackku-2026?ref_content=my-projects-tab&ref_feature=my_projects
RoadScout is a road-damage reporting app that combines:
- A PyTorch classifier for road-damage type and severity (deployed externally via Railway)
- A Next.js web app for upload, geolocation, map display, and report history
- Gemini-generated reporting summaries and authority recommendations
- MongoDB storage for uploaded photos and analysis results

## Best Results: Photo Capture Guidance

For the AI classifier to perform best, take road photos from a bird's-eye or high downward angle whenever possible.

- Prefer a top-down shot of the damaged region (standing over it or from a higher curb viewpoint).
- Keep the damage centered and clearly visible in frame.
- Avoid extreme side angles, heavy motion blur, and low-light shots.
- Include enough surrounding pavement context so crack direction/pattern is visible.
- Avoid blocking the damage with shadows, feet, or vehicle parts.

These capture habits substantially improve class and severity reliability.

## Repository Structure

- `web/`: Next.js frontend + API routes
- `api.py`: FastAPI server used to host the ML model on Railway
- `Procfile`: Command configuration to start the FastAPI server in production
- `predict.py`: CLI image inference script
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
NEXT_PUBLIC_ML_API_URL=https://your-railway-app-url.up.railway.app
```

Notes:
- `MONGODB_DB` can also be provided as `MONGO_DB_NAME`.
- `GEMINI_API_KEY` is also accepted instead of `GEMINI_API`.
- `NOMINATIM_CONTACT_EMAIL` is also accepted instead of `NOMINATIM_CONTACT_API`.
- The API route sends inference requests to `NEXT_PUBLIC_ML_API_URL` instead of running the local classifier.
- For local development, prefer `web/.env.local` over `web/.env`.

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
- Gemini quota/rate limits:
	If Gemini returns quota errors, the app falls back to a classifier-only summary so report creation still succeeds.

## Deployment
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
