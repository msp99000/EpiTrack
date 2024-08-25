# EpiTrack: Multi-Model EEG Seizure State Classifier

EpiTrack is an advanced machine learning project designed to predict seizure states (ictal, preictal, or interictal) from electrical signals in EEG data. This project leverages a robust ensemble approach, combining five distinct machine learning models to enhance prediction accuracy and reliability.

## Key Components:

1. **Individual Models**: Includes CNN, KNN, RF, SVM, and ADABoost implementations, each in separate .py files.

2. **Fusion Model**: Aggregates predictions from all five models using probability averaging ensembling technique, implemented in `display.py`.

3. **Pickle Files**: Contains pre-computed model predictions (`model_pred.pkl`) for efficient presentation and testing.

4. **Visualization**: Integrated with Streamlit for an interactive, cloud-deployed interface, also handled in `display.py`.

This project demonstrates the power of ensemble learning in medical signal processing, offering a comprehensive approach to seizure state prediction from EEG data.
