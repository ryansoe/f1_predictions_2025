# f1_predictions_2025

## Project Overview
This repository features a **Gradient Boosting model** designed to predict Formula 1 race outcomes using prior race data, qualifying results, and other structured information. The workflow incorporates:
- FastF1 API for historical datasets  
- 2024 season race outcomes  
- 2025 qualifying session data  
- Ongoing updates throughout the season to refine accuracy  
- Feature engineering to enhance predictive performance  

## Data Sources
- **FastF1 API**: Provides lap times, race results, and telemetry  
- **2025 Qualifying Results**: Input for current-season predictions  
- **Historical F1 Data**: Compiled from FastF1 to train the model  

## Workflow
1. **Data Retrieval**: Collects required F1 data through FastF1.  
2. **Preprocessing & Feature Engineering**: Cleans data, standardizes driver identifiers, and formats results.  
3. **Model Training**: Trains a **Gradient Boosting Regressor** on 2024 race data.  
4. **Prediction**: Generates predicted race times for 2025 and ranks drivers.  
5. **Evaluation**: Measures accuracy using **Mean Absolute Error (MAE)**.  

### Dependencies
- `fastf1`  
- `numpy`  
- `pandas`  
- `scikit-learn`  
- `matplotlib`  

## Model Evaluation
Performance is assessed with **MAE**, where lower scores indicate more precise predictions.  

## Future Enhancements
- Add **weather conditions** as predictive features  
- Incorporate **pit stop strategies**  
- Experiment with **deep learning approaches** for greater accuracy  
