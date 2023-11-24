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
model_name = "BAAI/llm-embedder"
run_name = model_name.split("/")[-1] + "-fine-tuning"
wandbc = WandbClient(run_name=run_name)

model = SentenceTransformer(model_name, wandbc=wandbc)

# load datasets from wandb
train_dataset = pd.read_parquet(wandbc.load_dataset("zbmath_open_train"))
dev_dataset = pd.read_parquet(wandbc.load_dataset("zbmath_open_dev"))

train_examples = []
dev_examples = []

logger.info("Creating train and dev examples...")
for i, row in tqdm(train_dataset.iterrows(), total=len(train_dataset)):
    train_examples.append(InputExample(texts=[row["text_x"], row["text_y"]]))

for i, row in tqdm(dev_dataset.iterrows(), total=len(dev_dataset)):
    dev_examples.append(InputExample(texts=[row["text_x"], row["text_y"]]))

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
dev_dataloader = DataLoader(dev_examples, shuffle=True, batch_size=16)

train_loss = MultipleNegativesRankingLoss(model=model)


with open(wandbc.load_dataset("queries"), 'rb') as file:
    queries = pickle.load(file)

with open(wandbc.load_dataset("corpus"), 'rb') as file:
    corpus = pickle.load(file)

with open(wandbc.load_dataset("relevant_docs"), 'rb') as file:
    relevant_docs = pickle.load(file)


evaluator = InformationRetrievalEvaluator(queries=queries,corpus=corpus,relevant_docs=relevant_docs,wandbc=wandbc)

training_args = {
    "epochs": 10,
    "scheduler": "WarmupLinear",
    "warmup_steps": 500,
    "evaluator": evaluator,
    "evaluation_steps": 1000,
    "checkpoint_save_steps": 1000,
    "checkpoint_save_total_limit": 3,
}

logger.info("Starting training...")
model.fit(train_objectives=[(train_dataloader, train_loss)], **training_args)


# logging.info("Uploading model...")
# base_model = train_loss.model[0].auto_model
# tokenizer = train_loss.model.tokenizer

# base_model.push_to_hub(
#     "horychtom/" + model_name.split("/")[-1] + "-zbmath-open",
# )
# tokenizer.push_to_hub(
#     "horychtom/" + model_name.split("/")[-1] + "-zbmath-open",
# )

wandbc.finish()
