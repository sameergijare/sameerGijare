python model_classifyAppRisk.py "../input/raw" "../config/feature_name_to_number_mapping.csv" "../outputs/models"
python test.py "../input/test" "../config/feature_name_to_number_mapping.csv" "../outputs/models/malware_model.pkl"
