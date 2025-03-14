import os
import torch
import pickle
import yaml
import nltk
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from model import load_model_processor
from preprocess_data import VQADataset

def evaluate_model():
    """Evaluates both the last saved model and the best model based on multiple evaluation metrics."""
    config = yaml.safe_load(open("./config.yaml", "r"))
    model_config = config["finetune_model"]
    data_config = config["data_location"]
    result_dir = config["result"]
    params = yaml.safe_load(open("./param.yaml", "r"))["params"]
    
    os.makedirs(result_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, processor = load_model_processor()
    
    # Load test dataset
    with open(data_config["test_processed_data"], "rb") as f:
        test_dataset = pickle.load(f)
    
    test_dataloader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False, pin_memory=True)
    
    def evaluate(model_path):
        model = load_model_processor(model_path=model_path)[0].to(device)
        model.eval()
        bleu_scores = []
        references_list = []
        predictions_list = []
        smooth_fn = SmoothingFunction().method1
        
        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc=f"Evaluating {model_path}"):
                input_ids = batch.pop('input_ids').to(device)
                pixel_values = batch.pop('pixel_values').to(device)
                attention_mask = batch.pop('attention_mask').to(device)
                labels = batch.pop('labels').to(device)

                generated_ids = model.generate(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, max_length=8)
                predictions = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                references = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                for pred, ref in zip(predictions, references):
                    bleu_scores.append(sentence_bleu([word_tokenize(ref)], word_tokenize(pred), smoothing_function=smooth_fn))
                    references_list.append([word_tokenize(ref)])
                    predictions_list.append(word_tokenize(pred))
        
        avg_bleu = sum(bleu_scores) / len(bleu_scores)
        corpus_bleu_score = corpus_bleu(references_list, predictions_list, smoothing_function=smooth_fn)
        results = {
            "model_path": model_path,
            "avg_bleu": avg_bleu,
            "corpus_bleu": corpus_bleu_score
        }
        
        result_file = os.path.join(result_dir, f"{os.path.basename(model_path)}_evaluation.json")
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {result_file}")
        return avg_bleu, corpus_bleu_score
    
    last_model_bleu, last_model_corpus_bleu = evaluate(model_config["last"])
    best_model_bleu, best_model_corpus_bleu = evaluate(model_config["best"])
    
    print("Comparison:")
    print(f"Last Model BLEU Score: {last_model_bleu:.4f}, Corpus BLEU Score: {last_model_corpus_bleu:.4f}")
    print(f"Best Model BLEU Score: {best_model_bleu:.4f}, Corpus BLEU Score: {best_model_corpus_bleu:.4f}")
    
    if best_model_bleu >= last_model_bleu:
        print("Best model performs better or equal.")
    else:
        print("Warning: The last model outperforms the best model!")

if __name__ == "__main__":
    evaluate_model()