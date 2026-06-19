Run.bat file contains input arguments to python files

python model_classifyAppRisk.py "../input/raw" "../config/feature_name_to_number_mapping.csv" "../outputs/models" "../outputs/" "../input/processed/app_risk_dataset.csv"

First Argument: Input data files
Second Argument: Feature Mapping file-system
Third Argument: Output directory where trained model is saved
Fourth Argument: Output directory where feature_importance.png is saved
Fifth Argument: Application intermediary file location

python test.py "../input/test" "../config/feature_name_to_number_mapping.csv" "../outputs/models/malware_model.pkl"

First Argument: Input Test data files
Second Argument: Feature Mapping file-system
Third Argument: Output directory where model is saved and from where it is read by test script.

