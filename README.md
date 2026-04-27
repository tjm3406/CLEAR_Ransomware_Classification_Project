# CLEAR_Ransomware_Classification_Project

**read_data.py** performs filtering and data ingestion on the raw dataset which can be found here: gs://metcs777-term-project-clear/CLEAR_NeurIPS/

This python program takes in two arguments:
  1. Input Directory (gs URI found above)
  2. Output Directory

**merge_for_ml.py** performs feature engineering on the filtered dataset which can be found here: gs://metcs777-term-project-clear/data_to_use/

This python program takes in two arguments:
  1. Input Directory (gs URI found above)
  2. Output Directory

**tuned_training.py and try_models_local.py** both perform model training and evaluation on the feature engineered dataset which can be found here: gs://metcs777-term-project-clear/ml_ready/

This python program takes in two arguments:
  1. Input Directory (gs URI found above)
  2. Output Directory

Workflow Diagram:

![Workflow Diagram](CLEAR_Ransomeware_Classification_Project/CLEAR_Workflow_Diagram.drawio.pdf)
