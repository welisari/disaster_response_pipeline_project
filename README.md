# Disaster Response Pipeline Project

## Table of Contents
1. [Project Purpose](#purpose)
2. [Installation and Requirements](#installation)
3. [Code Structure](#code)
4. [File Descriptions and Instructions](#instruction)
5. [Results](#results)
6. [Credits](#credits)


### Project Purpose <a name="purpose"></a>
The concept of the project is to apply the data engineering skills to analyze disaster data from Figure Eight 
to build a model for an API that classifies disaster messages. The goal is to create a Machine pipeline to categorize 
these events so that you can send the messages to an appropriate disaster relief agency.
The projects is also including a web app where an emergency worker can input a new message and get classification results in several categories. 
The web app displays visualizations of the data.


### Installation and Requirements<a name="installation"></a>
Python version of Anaconda Distribution will be mainly enough for the necessary packages and modules for analysis and execution of codes.
The main ToolKits are python, Flask, matplotlib, numpy, pandas, pickle, plotly, scikit-learn, SQLAlchemy, sys, re, json, NLTK

### Code Structure <a name="code"></a>

- `app/`
  - `template/`
    - `master.html`  -  Main page of web application.
    - `go.html`  -  Classification result page of web application.
  - `run.py`  - Flask applications main file.

- `data/`
  - `disaster_categories.csv`  - Disaster categories dataset.
  - `disaster_messages.csv`  - Disaster Messages dataset.
  - `process_data.py` - The data processing pipeline script.
  - `DisasterResponse.db`   - The database with the merged and clean data.

- `models/`
  - `train_classifier.py` - The NLP and ML pipeline script.
  

### File Descriptions and Instructions <a name="instruction"></a>
1. data/process_data.py: The ETL pipeline used to process and clean data in preparation for model building.	
	- Combines the two given datasets (In CSV  format)
	- Cleans the data
	- Stores it in a SQLite database
	To **run ETL pipeline**:
	-> `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
	
2. models/train_classifier.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle.	
	- Splits the dataset into training and test sets
	- Builds a text processing and machine learning pipeline
	- Trains and tunes a model using GridSearchCV
	- Outputs results on the test set
	- Exports the final model as a pickle file
	To **run ML pipeline**:
	-> `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
	
3. app/templates/*.html: HTML templates required for the web app.

4. app/run.py: To start the Python server for the web app and render visualizations.
	- Flask app: When a user inputs a message into the app, the app returns classification results for all 36 categories.
	To **run web app**:
	-> `python run.py`
5. Go to http://0.0.0.0:3001/ for the visualization and input classification


### Results<a name="results"></a>
The main observations of the trained classifier can be seen by running this application.

###Credits <a name="credits"></a>
Thanks to Udacity for providing the project idea and supoort on it.