from sentence_transformers.wandb_.wandb_client import WandbClient
from sentence_transformers.SentenceTransformer import SentenceTransformer


# Load the dataset
model_name = "BAAI/llm-embedder"
run_name = model_name.split("/")[-1] + "-fine-tuning"
wandbc = WandbClient(run_name=run_name)

model = SentenceTransformer(model_name, wandbc=wandbc)

# # load datasets from wandb
# train_dataset = pd.read_csv(wandbc.load_dataset("final_train_dataset"))
# dev_dataset = pd.read_csv(wandbc.load_dataset("final_dev_dataset"))

# train_examples = []

# for i,row in tqdm(train_dataset.iterrows(),total=len(train_dataset)):
#     train_examples.append(InputExample(texts=[row['anchor'], row['rec']]))

# train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)

# train_loss = MultipleNegativesRankingLoss(model=model)


# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2)


wandbc.finish()
