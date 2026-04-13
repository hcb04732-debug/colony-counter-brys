# Colony Counter Web App

This project includes:

- `streamlit_app.py`: public web app entrypoint
- `colony_counter_core.py`: reusable colony detection and review logic
- `colony_counter_app.py`: local PySide6 desktop app

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy publicly

The easiest public deployment path is Streamlit Community Cloud:

1. Push this project to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Choose that repository and set the entrypoint to `streamlit_app.py`.
4. Deploy.

Your users will then access the app from a public web URL.
