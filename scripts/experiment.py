from sentence_transformers.readers import InputExample
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.wandb_ import WandbClient
from sentence_transformers.evaluation import (
    MultipleNegativesRankingLossEvaluator,
)
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)


# Load the dataset
model_name = "roberta-base"
run_name = model_name.split("/")[-1] + "-fine-tuning"
wandbc = WandbClient(run_name=run_name)

model = SentenceTransformer(model_name, wandbc=wandbc)

# load datasets from wandb
train_dataset = pd.read_csv(wandbc.load_dataset("zbmath_train"))
dev_dataset = pd.read_csv(wandbc.load_dataset("zbmath_dev"))

train_examples = []
dev_examples = []

logger.info("Creating train and dev examples...")
for i, row in tqdm(train_dataset.iterrows(), total=len(train_dataset)):
    train_examples.append(InputExample(texts=[row["text_x"], row["text_y"]]))

for i, row in tqdm(dev_dataset.iterrows(), total=len(dev_dataset)):
    dev_examples.append(InputExample(texts=[row["text_x"], row["text_y"]]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=5)
dev_dataloader = DataLoader(dev_examples, shuffle=True, batch_size=5)

train_loss = MultipleNegativesRankingLoss(model=model)
evaluator = MultipleNegativesRankingLossEvaluator(dev_dataloader)
training_args = {
    "epochs": 1,
    "scheduler": "WarmupLinear",
    "warmup_steps": 500,
    "evaluator": evaluator,
    "evaluation_steps": 200,
    "checkpoint_save_steps": 200,
    "checkpoint_save_total_limit": 3,
}

model.fit(train_objectives=[(train_dataloader, train_loss)], **training_args)
wandbc.finish()
