News2Trade is an NLP and Machine Learning system built to predict short-term stock price movement based on real-time financial news sentiment. The project explores how breaking news influences stock volatility using transformer-based NLP models and deep learning for time-series forecasting.

---

## 🔍 Features

- 🧠 **Transformer-based NLP (FinBERT)** for sentiment scoring  
- 🏷 **Keyword extraction (KeyBERT / TF-IDF)**  
- 📈 **Time-series forecasting models (LSTM/GRU)**  
- 🧪 Experimentation and fine‑tuning with notebooks  
- 💾 Modular codebase with preprocessing pipeline  
- 🛠 Supports real-time and historical analysis  

---

## 📦 Project Structure

```
├── config/              # Configuration files (parameters, paths)
├── data/                # (Ignored) Raw + processed datasets
├── notebooks/           # Experimentation notebooks
├── pretrained_model/    # (Ignored) Saved weights & embeddings
├── src/                 # Source code: models, utils, pipelines
├── requirements.txt     # Python dependencies
└── README.md
```

> Note: `data/` and `pretrained_model/` folders are excluded for storage and security.

---

## 🛠 Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage (Coming Soon)

- Notebook examples
- CLI entry point
- Training + inference commands

---

## 📈 Planned Enhancements

- [ ] Add CLI + API interface  
- [ ] Add dataset download script  
- [ ] Add model dashboard + evaluation metrics  
- [ ] Add inference demo with live news feed integration  

---

## 📜 License

This project is licensed under the MIT License.

---