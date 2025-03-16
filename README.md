# ExoVol

ExoVol is an advanced volatility surface modeling and forecasting tool that uses deep learning techniques including Variational Autoencoders (VAE) and Long Short-Term Memory (LSTM) networks.

## Features

- Data preprocessing and exploratory data analysis
- Volatility surface visualization
- VAE-LSTM model for volatility forecasting
- Hyperparameter optimization using Optuna
- Model evaluation and stress testing
- TensorFlow Lite conversion for deployment

## Installation

Clone the repository and install the required packages:
git clone https://github.com/premdev1234 /ExoVol.git
cd ExoVol
pip install -r requirements.txt

text

## Usage

1. Prepare your data in CSV format similar to `Prototyping_dataset.csv`.
2. Run the preprocessing script:
python src/data_preprocessing.py

3. Train the VAE-LSTM model:
python src/vae_lstm_model.py


4. Evaluate and visualize results:
python src/visualization.py



## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
