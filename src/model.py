import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
import yaml
# Initialize the BLIP model and processor
config = yaml.safe_load(open("./config.yaml", "r"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = config["finetune_model"]["orignal_model_id"]
def load_model_processor(model_path = model_id):
    print("Loading Model and Processor................")
    model = BlipForQuestionAnswering.from_pretrained(model_path).to(device)
    processor = BlipProcessor.from_pretrained(model_id)
    print(f"Model and Processor loaded successfully {model_path}  !!!")
    return model, processor
