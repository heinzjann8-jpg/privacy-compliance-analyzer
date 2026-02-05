
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

## 3) Run
```bash
python app.py
```
Open http://127.0.0.1:5000 and search for a manufacturer (e.g., Apple, AXS, Billboard) or submit your own.

