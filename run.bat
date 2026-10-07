@echo off
pip install -r requirements.txt
if not exist models\results.json python train.py
start http://127.0.0.1:5000
python app.py
