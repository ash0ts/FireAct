from transformers.integrations import WandbCallback
import pandas as pd

def decode_predictions(tokenizer, predictions):
    prediction_text = tokenizer.batch_decode(predictions.predictions.argmax(axis=-1))
    return {"predictions": prediction_text}


class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each logging step during training.
    It allows to visualize the model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset for generating predictions.
        num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, trainer, tokenizer, val_dataset, num_samples=100, freq=2):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            trainer (Trainer): The Hugging Face Trainer instance.
            tokenizer (AutoTokenizer): The tokenizer associated with the model.
            val_dataset (Dataset): The validation dataset.
            num_samples (int, optional): Number of samples to select from the validation dataset for generating predictions. Defaults to 100.
            freq (int, optional): Frequency of logging. Defaults to 2.
        """
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.sample_dataset = val_dataset.select(range(num_samples))
        self.freq = freq


    def on_evaluate(self, args, state, control,  **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        
        predictions = self.trainer.predict(self.sample_dataset)
        predictions = decode_predictions(self.tokenizer, predictions)
        predictions_df = pd.DataFrame(predictions)
        predictions_df["global_step"] = state.global_step
        predictions_df["input"] = self.sample_dataset["input"]
        predictions_df["output"] = self.sample_dataset["output"]
        predictions_df["instruction"] = self.sample_dataset["instruction"]

        records_table = self._wandb.Table(dataframe=predictions_df)
        self._wandb.log({"sample_predictions": records_table})
