from src.data.make_dataset import load_data, save_dataset
from src.models.preprocess import preprocess_data
from src.models.train import train_model
from src.models.evaluate import evaluate_model
from src.models.eda import exploratory_data_analysis

if __name__ == "__main__":
    train_df, test_df = load_data(train_path="data/raw/train.csv", test_path="data/raw/test.csv")  
    save_dataset(train_df, test_df)  
    exploratory_data_analysis()
    preprocess_data()
    train_model()
    evaluate_model()