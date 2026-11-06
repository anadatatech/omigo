# Lagos Waterways Bathymetry

A pipeline and web app for fetching Lagos waterway data, snapping it to OpenStreetMap routes, building a bathymetry dataset, and serving the results through a FastAPI app.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) installed
- Python 3.10+ (uv can manage this for you)

## Setup

1. Initialize the project (skip this if `pyproject.toml` already exists):

   ```bash
   uv init
   ```

2. Install dependencies:

   ```bash
   uv add -r requirements.txt
   ```

## Data Pipeline

Run the scripts in order, since each step depends on the output of the previous one.

1. **Fetch Lagos waterways**

   ```bash
   uv run ./app/fetch_lagos_waterways.py
   ```

2. **Snap waterways to OSM routes**

   ```bash
   uv run ./app/snap_to_osm_route.py
   ```

3. **Build the Lagos bathymetry dataset**

   ```bash
   uv run ./app/build_lagos_bathymetry.py
   ```

## Running the App

Start the development server with auto-reload:

```bash
uv run uvicorn app.app:app --reload
```

The app will be available at <http://127.0.0.1:8000> by default.
If you're using FastAPI's built-in docs, they are at <http://127.0.0.1:8000/docs>.

## Project Structure

```
.
├── app/
│   ├── app.py                      # FastAPI application
│   ├── fetch_lagos_waterways.py    # Step 1: fetch waterway data
│   ├── snap_to_osm_route.py        # Step 2: snap to OSM routes
│   └── build_lagos_bathymetry.py   # Step 3: build bathymetry
├── requirements.txt
└── pyproject.toml
```