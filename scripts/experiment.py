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
from peft import get_peft_model, LoraConfig
logger = logging.getLogger(__name__)


# Load the dataset
model_name = "thenlper/gte-base"
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

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
dev_dataloader = DataLoader(dev_examples, shuffle=True, batch_size=32)

train_loss = MultipleNegativesRankingLoss(model=model)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=4,
    target_modules=["query","key","value"]
)
train_loss = get_peft_model(train_loss,peft_config=peft_config)

with open(wandbc.load_dataset("queries"), 'rb') as file:
    queries = pickle.load(file)

with open(wandbc.load_dataset("corpus"), 'rb') as file:
    corpus = pickle.load(file)

with open(wandbc.load_dataset("relevant_docs"), 'rb') as file:
    relevant_docs = pickle.load(file)


evaluator = InformationRetrievalEvaluator(queries=queries,corpus=corpus,relevant_docs=relevant_docs,wandbc=wandbc)

training_args = {
    "epochs": 1,
    "scheduler": "WarmupLinear",
    "warmup_steps": 100,
    "evaluator": evaluator,
    "evaluation_steps": 100,
    "checkpoint_save_steps": 500,
    "checkpoint_save_total_limit": 3,
}

model.fit(train_objectives=[(train_dataloader, train_loss)], **training_args)
wandbc.finish()
