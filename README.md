## Setup
pip install -r requirements.txt

python scripts/prepare_data.py

python scripts/run_clustering.py

uvicorn main:app --reload
