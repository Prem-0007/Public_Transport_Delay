#!/bin/bash
pip install -r requirements.txt
[ -f models/results.json ] || python train.py
python app.py
