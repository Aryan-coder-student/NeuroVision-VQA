import os
import torch
import pickle
import yaml
from datasets import load_dataset
from model import load_model_processor

config = yaml.safe_load(open("./config.yaml", "r"))["data_location"]
class VQADataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question = self.dataset[idx]['question']
        answer = self.dataset[idx]['answer']
        image = self.dataset[idx]['image']
        image = image.convert("RGB")
        text = question

        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        labels = self.processor.tokenizer.encode(
            answer, max_length=128, padding="max_length", truncation=True,pad_to_max_length=True, return_tensors='pt'
        )
        encoding["labels"] = labels
        for k, v in encoding.items():
            encoding[k] = v.squeeze()
        return encoding

if __name__ == "__main__":
    _, processor = load_model_processor()

    
    print("Loading VQA dataset................")
    data = load_dataset(config["data"])
    train_data = data["train"]
    test_data = data["test"]

    print("VQA dataset loaded successfully!!! lenght of train data is ", len(train_data), " and test data is ", len(test_data))
    save_dir = "./data/silver"
    os.makedirs(save_dir, exist_ok=True)
    print("Processesing data to save in ../data/silver")
    train_dataset = VQADataset(dataset=train_data,
                          processor=processor)
    test_dataset = VQADataset(dataset=test_data,
                          processor=processor)
    

    print(f"Data processed successfully !!! ")
    print(f"Saving to {os.path.join(save_dir,'train_dataset.pkl')} and {os.path.join(save_dir,'test_dataset.pkl')}")
    with open(os.path.join(save_dir,"train_dataset.pkl"), "wb") as f:
        pickle.dump(train_dataset, f)
    with open(os.path.join(save_dir,"test_dataset.pkl"), "wb") as f:
        pickle.dump(test_dataset, f)
    print(f"Processed data , saved to {config['train_processed_data']} and {config['test_processed_data']}")









    