import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_sequences(data, input_length, forecast_length):
    sequences = []
    targets = []
    for i in range(len(data) - input_length - forecast_length):
        seq = data[i:i + input_length]
        target = data[i + input_length:i + input_length + forecast_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)

def main(args):
    # Load the data
    df = pd.read_json(args.input_file)
    
    # Convert timestamp to datetime, remove duplicates, and sort
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.drop_duplicates(subset='Timestamp', keep='first')
    df = df.sort_values('Timestamp')
    print(f"Number of unique timestamps: {len(df['Timestamp'].unique())}")

    # Select numerical columns for features
    feature_columns = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    print(f"Numerical feature columns: {feature_columns}")
    features = df[feature_columns].values
    print(f"Selected {len(feature_columns)} numerical features")

    # Normalize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Check if we have enough data points to create sequences
    if len(scaled_features) < args.input_length + args.forecast_length:
        raise ValueError(f"Not enough data points to create sequences. Have {len(scaled_features)}, need at least {args.input_length + args.forecast_length}.")

    # Create sequences
    sequences, targets = create_sequences(scaled_features, args.input_length, args.forecast_length)
    print(f"Generated {len(sequences)} sequences.")

    # Split into input (X) and output (y)
    X = sequences
    y = targets

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Save the preprocessed data
    np.savez(args.output_file, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
    print(f"Data saved to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preparation script for time series forecasting.")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to the output file (npz format).')
    parser.add_argument('--input_length', type=int, required=True, help='Length of the input sequences.')
    parser.add_argument('--forecast_length', type=int, required=True, help='Length of the forecast sequences.')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split.')

    args = parser.parse_args()
    main(args)
