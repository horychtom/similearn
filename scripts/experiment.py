from similearn.readers import InputExample
from similearn.losses import MultipleNegativesRankingLoss
from similearn.wandb_ import WandbClient
from similearn.evaluation import (
    MultipleNegativesRankingLossEvaluator,
    InformationRetrievalEvaluator
)
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from similearn import SentenceTransformer
import logging
import pickle

logger = logging.getLogger(__name__)


# Load the dataset
model_name = "roberta-base"
run_name = model_name.split("/")[-1] + "-fine-tuning"
wandbc = WandbClient(run_name=run_name)

model = SentenceTransformer(model_name, wandbc=wandbc)

# load datasets from wandb
train_dataset = pd.read_csv(wandbc.load_dataset("zbmath_train"))[:20]
dev_dataset = pd.read_csv(wandbc.load_dataset("zbmath_dev"))[:20]

train_examples = []
dev_examples = []

logger.info("Creating train and dev examples...")
for i, row in tqdm(train_dataset.iterrows(), total=len(train_dataset)):
    train_examples.append(InputExample(texts=[row["text_x"], row["text_y"]]))

for i, row in tqdm(dev_dataset.iterrows(), total=len(dev_dataset)):
    dev_examples.append(InputExample(texts=[row["text_x"], row["text_y"]]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
dev_dataloader = DataLoader(dev_examples, shuffle=True, batch_size=2)

train_loss = MultipleNegativesRankingLoss(model=model)
evaluator = MultipleNegativesRankingLossEvaluator(dev_dataloader)

with open(wandbc.load_dataset("queries"), 'rb') as file:
    queries = pickle.load(file)

with open(wandbc.load_dataset("corpus"), 'rb') as file:
    corpus = pickle.load(file)

with open(wandbc.load_dataset("relevant_docs"), 'rb') as file:
    relevant_docs = pickle.load(file)


evaluator = InformationRetrievalEvaluator(queries=queries,corpus=corpus,relevant_docs=relevant_docs)

training_args = {
    "epochs": 1,
    "scheduler": "WarmupLinear",
    "warmup_steps": 500,
    "evaluator": evaluator,
    "evaluation_steps": 2,
    "checkpoint_save_steps": 200,
    "checkpoint_save_total_limit": 3,
}

model.fit(train_objectives=[(train_dataloader, train_loss)], **training_args)
wandbc.finish()
