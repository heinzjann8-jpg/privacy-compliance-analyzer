
## 1) Install
```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1  (if execution policy blocks, use: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)
source .venv/bin/activate  # Windows CMD: .venv\Scripts\activate.bat
pip install -r requirements.txt
```

## 2) Configure
Edit `config.json`:
- similarity_method:compare TF-IDF or BERT by typing tfidf or bert
- coverage_threshold: configure threshold to compare compliancy

## 3) Run (in local instance)
```bash
python app.py
```
Open http://127.0.0.1:5000 in browser to view site locally

Live Website Link: https://privacy-compliance-analyzer.onrender.com
