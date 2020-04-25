# Disaster Response Pipeline Project

## Table of Contents
1. [Installation](#installation)
2. [Instructions](#instructions)
3. [File Descriptions and Usage](#files)
4. [Code Structure](#codes)
5. [Results](#results)

### Installation <a name="installation"></a>
Python version of Anaconda Distribution will be mainly enough for the necessary packages and modules for analysis and execution of codes.


### Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.
    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
2. Run the following command in the app's directory to run your web app.
    `python run.py`
3. Go to http://0.0.0.0:3001/


### File Descriptions and Usage <a name="files"></a>
1. data/process_data.py: The ETL pipeline used to process and clean data in preparation for model building.
	
	* Combines the two given datasets (In CSV  format)
	* Cleans the data
	* Stores it in a SQLite database
	
2. models/train_classifier.py: The Machine Learning pipeline used to fit, tune, evaluate, and export the model to a Python pickle.
	
	* Splits the dataset into training and test sets
	* Builds a text processing and machine learning pipeline
	* Trains and tunes a model using GridSearchCV
	* Outputs results on the test set
	* Exports the final model as a pickle file  
	
3. app/templates/*.html: HTML templates required for the web app.

4. app/run.py: To start the Python server for the web app and render visualizations.
	*Flask app: When a user inputs a message into the app, the app returns classification results for all 36 categories.

### Code Structure <a name="codes"></a>

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

### Results<a name="results"></a>
The main observations of the trained classifier can be seen by running this application.