install:
	venv/bin/python -m pip install -r requirements.txt
run:
	venv/bin/python -m uvicorn server:app --reload