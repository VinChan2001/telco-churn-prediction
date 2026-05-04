train:
	python src/train_model.py

score:
	python src/score_customers.py

pipeline:
	python src/train_model.py
	python src/score_customers.py