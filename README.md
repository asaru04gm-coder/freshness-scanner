# 🥬 Fruit & Vegetable Freshness Scanner

A Streamlit web application that uses a TensorFlow/Keras model to determine whether a fruit or vegetable is **Fresh** or **Stale**.

You can either upload an image or use your device camera to take a photo directly in the browser.

## Files

- model.keras
- app.py
- requirements.txt

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Upload all files to a public GitHub repository.
2. Visit https://share.streamlit.io
3. Sign in with GitHub.
4. Click **New App**.
5. Select your repository.
6. Set the main file to:

```
app.py
```

7. Click **Deploy**.

The app will be available online and can be opened from any phone or computer.
