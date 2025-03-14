import os 
import yaml 
import nltk
import pickle
import torch
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from preprocess_data import VQADataset
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader
from model import load_model_processor
from transformers import get_linear_schedule_with_warmup
from nltk.translate.bleu_score import sentence_bleu , SmoothingFunction
from mlflow.models import infer_signature

nltk.download('all')

config = yaml.safe_load(open("./config.yaml", "r"))["data_location"]
model_config = yaml.safe_load(open("./config.yaml", "r"))["finetune_model"]
params = yaml.safe_load(open("./param.yaml", "r"))["params"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, processor  = load_model_processor()

print(f"Loading processed data from {config['train_processed_data']} and {config['test_processed_data']}.")
with open(f'./{config["train_processed_data"]}', "rb") as f:
    train_dataset = pickle.load(f)
with open(f'./{config["test_processed_data"]}', "rb") as f:
    test_dataset = pickle.load(f)
print(f"Loaded processed data successfully!!!")
print(f"Length of train dataset is {len(train_dataset)} and test dataset is {len(test_dataset)}")

batch_size = params["batch_size"]
num_epochs = params["num_epochs"]
patience = params["patience"]
gradient_accumulation_steps = params["gradient_accumulation_steps"]  
print("Preparing dataloaders................")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
print("Dataloaders prepared successfully!!!")


mlflow.set_experiment("VQA_Model_Training")


min_bleu_score = 0
early_stopping_hook = 0
tracking_information = []
optimizer = torch.optim.AdamW(model.parameters(), lr=float(params["learning_rate"]), weight_decay=float(params["weight_decay"]))
total_steps = len(train_dataloader) * num_epochs
warmup_steps = total_steps // 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
scaler = torch.amp.GradScaler('cuda')

with mlflow.start_run():
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")):
            input_ids = batch.pop('input_ids').to(device)
            pixel_values = batch.pop('pixel_values').to(device)
            attention_mask = batch.pop('attention_mask').to(device)
            labels = batch.pop('labels').to(device)

            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation_steps  

            scaler.scale(loss).backward()
            if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * gradient_accumulation_steps

        # Validation Loop
        model.eval()
        eval_loss = 0
        bleu_scores = []
        smooth_fn = SmoothingFunction().method1
        with torch.no_grad():
            for batch in tqdm(valid_dataloader, desc=f"Epoch {epoch+1} Validating"):
                input_ids = batch.pop('input_ids').to(device)
                pixel_values = batch.pop('pixel_values').to(device)
                attention_mask = batch.pop('attention_mask').to(device)
                labels = batch.pop('labels').to(device)

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, labels=labels)
                    eval_loss += outputs.loss.item()

                generated_ids = model.generate(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, max_length=8)
                predictions = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                references = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)

                for pred, ref in zip(predictions, references):
                    bleu_scores.append(sentence_bleu([word_tokenize(ref)], word_tokenize(pred), smoothing_function=smooth_fn))

        avg_train_loss = epoch_loss / len(train_dataloader)
        avg_eval_loss = eval_loss / len(valid_dataloader)
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
        tracking_information.append((avg_train_loss, avg_eval_loss, avg_bleu_score, optimizer.param_groups[0]["lr"]))
        
        mlflow.log_metrics({"train_loss": avg_train_loss, "eval_loss": avg_eval_loss, "bleu_score": avg_bleu_score}, step=epoch)
        mlflow.log_param(f"learning_rate_{epoch}", optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Eval Loss: {avg_eval_loss:.4f} - BLEU Score: {avg_bleu_score:.4f} - LR: {optimizer.param_groups[0]['lr']}")
        
        if avg_bleu_score >= min_bleu_score:
            model.save_pretrained(model_config["best"], from_pt=True)
            print(f"Model improved (BLEU: {avg_bleu_score:.4f})! Saved to {model_config['best']}")
            min_bleu_score = avg_bleu_score
            early_stopping_hook = 0
        else:
            early_stopping_hook += 1
            if early_stopping_hook > patience:
                print("Early stopping triggered.")
                break
        model.save_pretrained(model_config["last"], from_pt=True)
        scheduler.step()  

    
    signature = infer_signature(input_ids.cpu().numpy(), model(input_ids).logits.cpu().detach().numpy())
    mlflow.pytorch.log_model(model, "VQA_model", signature=signature)
    print("Model logged with MLflow.")
