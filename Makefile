train:
	python src/train_model.py

generate:
	python src/generate_synthetic_customers.py

score:
	python src/score_customers.py

pipeline:
	python src/train_model.py
	python src/generate_synthetic_customers.py
	python src/score_customers.py

dashboard:
	streamlit run app/streamlit_app.py

test:
	python -m compileall src app tests
	python -m unittest discover -s tests
