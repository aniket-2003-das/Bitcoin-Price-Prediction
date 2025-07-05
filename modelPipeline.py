from modules.dataRequestor import DataRequestor
from modules.dataLoader import SingleCoinDataLoader
from modules.dataPreProcessor import SingleCoinPreprocessor
from modules.sequenceGenerator import SingleCoinSequenceGenerator
from modules.modelBuilding import SingleCoinLSTM
from modules.modelTraining import SingleCoinTrainer
from modules.prediction import SingleCoinPredictor
from modules.modelEvaluation import SingleCoinEvaluator


# Model Pipeline
if __name__ == "__main__":
    # Step 1:
    tickers = ["BTC-USD"]
    start_date = "2014-09-17"
    end_date = "2025-06-01"
    interval = "1d"
    dr = DataRequestor(tickers, start_date, end_date, interval)
    dr.fetch_and_save_all()


    # Step 2:
    loader = SingleCoinDataLoader(coin_name="BTC")
    # # make sure to uncomment if running for first time
    # loader.load_coin_data("BTC-USD.csv")
    # loader.save_coin_data()
    coin_data = loader.load_processed_data("./data/BTC-PROCESSED.csv") 
    

    # Step 3:
    # Initialize the preprocessor
    preprocessor = SingleCoinPreprocessor(prediction_days=60)
    # Prepare the data (splits + scaling)
    processed_data = preprocessor.prepare_data(coin_data)
    # Save preprocessor (e.g., MinMaxScaler and settings)
    preprocessor.save_preprocessor("assets/single_preprocessor.pkl")
    # Later: Load preprocessor for inference or reuse
    preprocessor.load_preprocessor("assets/single_preprocessor.pkl")


    # Step 4: Generate sequences
    seq_generator = SingleCoinSequenceGenerator(lookback=7)
    # Use scaled training data from the preprocessor output
    scaled_train = processed_data["scaled_train"]
    X_train, y_train = seq_generator.generate_sequences(scaled_train)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)


    # Step 5. Model Building
    input_shape = (X_train.shape[1], X_train.shape[2])  # (lookback, 1)
    model_builder = SingleCoinLSTM(input_shape)
    model = model_builder.build_model([128, 64], dropout_rate=0.2)
    model_builder.get_model_summary()
    
    
    # Step 6. Training
    trainer = SingleCoinTrainer(model)
    

    # Simple train/val split
    val_split = 0.2
    split_idx = int(len(X_train) * (1 - val_split))
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]
    X_train = X_train[:split_idx]
    y_train = y_train[:split_idx]
    
    history = trainer.train(X_train, y_train, X_val, y_val, epochs=200)
    trainer.plot_training_history()
    trainer.load_best_model()


    # Step 7. Prediction
    predictor = SingleCoinPredictor(model, preprocessor, seq_generator)
    predictions = predictor.predict(processed_data, num_future_days=7)


    # Step 8: Evaluation
    evaluator = SingleCoinEvaluator()
    metrics = evaluator.calculate_metrics(predictions)
    evaluator.print_metrics_summary()
    evaluator.plot_predictions(predictions)


