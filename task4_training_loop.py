import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score
from seqeval.metrics import classification_report as ner_classification_report, f1_score as ner_f1_score
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, TaskType, get_peft_model
        
# Define class labels and compute num_class_a
class_labels = ["Abstract", "Background", "Methods", "Results", "Conclusion"]
num_class_a = len(class_labels)

# Load dataset and compute num_ner_labels
bionlp = load_dataset("tner/bionlp2004")
all_tags = [tag for example in bionlp["train"] for tag in example["tags"]]
num_ner_labels = max(all_tags) + 1

class NLPMultiTaskDataset(Dataset):
    """
    Dataset wrapper for the BioNLP dataset supporting both classification and NER tasks.
    """
    def __init__(self, hf_dataset, class_mapping):
        self.hf_dataset = hf_dataset
        self.class_mapping = class_mapping

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        sentence = " ".join(item["tokens"])
        tokens_raw = item["tokens"]
        ner_tags = item["tags"]
        # Simulate classification label (in practice, use real labels)
        classification_label = torch.randint(0, len(self.class_mapping), (1,)).item()
        return {
            "sentence": sentence,
            "tokens_raw": tokens_raw,
            "classification_label": classification_label,
            "ner_tags": ner_tags,
        }

def compute_classification_metrics(logits, labels):
    """
    Compute precision, recall, and F1 score for classification task.
    """
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    f1 = f1_score(labels, preds, average="macro")
    return precision, recall, f1

def compute_classification_accuracy(logits, labels):
    """
    Compute classification accuracy.
    """
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()

def convert_predictions_to_labels(predictions, id2label):
    """
    Convert token-level predictions to label strings.
    """
    batch_labels = []
    for i in range(predictions.shape[0]):
        sentence_labels = []
        for idx in predictions[i].tolist():
            if idx == -100:
                continue
            sentence_labels.append(id2label.get(idx, "O"))
        batch_labels.append(sentence_labels)
    return batch_labels


def convert_true_labels(aligned_labels,id2label):
    """
    Convert true labels (aligned) to label strings.
    """
    batch_labels = []
    for sent in aligned_labels.tolist():
        sent_labels = []
        for label in sent:
            if label == -100:
                continue
            sent_labels.append(id2label.get(label, "O"))
        batch_labels.append(sent_labels)
    return batch_labels

class MultiTaskTransformer(nn.Module):
    def __init__(self, model_name="roberta-large", num_class_a=num_class_a, num_ner_labels=num_ner_labels, freeze_backbone=True):
        """
        num_class_a: number of sentence classification labels (e.g., Background, Methods, Results, Conclusion)
        num_ner_labels: number of NER labels, e.g., 11 based on your id2label mapping.
        """
        super(MultiTaskTransformer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.roberta = AutoModel.from_pretrained(model_name)
        
        # Freeze the transformer backbone if specified.
        if freeze_backbone:
            for param in self.roberta.parameters():
                param.requires_grad = False
         
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_class_a)
        self.ner_classifier = nn.Linear(self.roberta.config.hidden_size, num_ner_labels)
    
    def mean_pooling(self, model_output, attention_mask):
        # Aggregates token embeddings to form a sentence-level representation.
        token_embeddings = model_output.last_hidden_state  # shape: [batch_size, seq_len, hidden_dim]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def tokenize_and_align_labels(self, examples):
        """
        Accepts a dict with keys "tokens" (list of list of tokens) and "tags" (list of corresponding label lists).
        Returns tokenized inputs with aligned labels.
        """
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding=True)
        
        labels = []
        for i, label in enumerate(examples["tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Special tokens get ignored.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)  # Only label the first sub-token.
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def forward(self, inputs, task='A'):
        if task == 'A':
            # For classification, assume inputs is a list of sentences (strings).
            encoded_input = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
            outputs = self.roberta(**encoded_input)
            pooled_output = self.mean_pooling(outputs, encoded_input['attention_mask'])
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits

        elif task == 'B':
            # For NER, we expect inputs as a dictionary with keys "tokens" and optionally "tags"
            # If tags are provided, they will be aligned; if not, you can still get the logits.
            if "tags" in inputs:
                tokenized_inputs = self.tokenize_and_align_labels(inputs)
                # Convert lists in tokenized_inputs to tensors
                tokenized_inputs = {k: torch.tensor(v) if k != "labels" else v for k, v in tokenized_inputs.items()}
            else:
                tokenized_inputs = self.tokenizer(inputs["tokens"], is_split_into_words=True, padding=True, truncation=True, return_tensors="pt")

            if "labels" in tokenized_inputs:
                labels = tokenized_inputs.pop("labels") 
            
            outputs = self.roberta(**tokenized_inputs)
            sequence_output = outputs.last_hidden_state
            sequence_output = self.dropout(sequence_output)
            logits = self.ner_classifier(sequence_output)
            return logits, tokenized_inputs
        else:
            raise ValueError("Task must be either 'A' (Classification) or 'B' (NER).")


def train_epoch(model, dataloader, optimizer, classification_loss_fn, ner_loss_fn, 
                ner_loss_weight, id2label, device="cpu"):
    """
    Train for one epoch.
    """
    model.train()
    running_loss = 0.0
    running_class_acc = 0.0
    running_class_precision = 0.0
    running_class_recall = 0.0
    running_class_f1 = 0.0
    all_true_ner = []
    all_pred_ner = []
    total_batches = 0

    for batch in dataloader:
        optimizer.zero_grad()

        # Task A: Classification
        sentences = batch["sentence"]
        class_labels = torch.tensor(batch["classification_label"]).to(device)
        class_logits = model(sentences, task='A')
        classification_loss = classification_loss_fn(class_logits, class_labels)

        # Task B: NER
        ner_inputs = {
            "tokens": batch["tokens_raw"],
            "tags": batch["ner_tags"]
        }
        ner_logits, tokenized_inputs = model(ner_inputs, task='B')
        aligned_labels = torch.tensor(tokenized_inputs["labels"]).to(device)
        ner_loss = ner_loss_fn(ner_logits.view(-1, ner_logits.shape[-1]), 
                              aligned_labels.view(-1))

        # Total loss and backpropagation
        total_loss = classification_loss + ner_loss_weight * ner_loss
        total_loss.backward()
        optimizer.step()

        # Compute metrics
        running_loss += total_loss.item()
        running_class_acc += compute_classification_accuracy(class_logits, class_labels)
        p, r, f = compute_classification_metrics(class_logits, class_labels)
        running_class_precision += p
        running_class_recall += r
        running_class_f1 += f

        # NER predictions
        ner_preds = torch.argmax(ner_logits, dim=-1)
        batch_pred_ner = convert_predictions_to_labels(ner_preds, id2label)
        batch_true_ner = convert_predictions_to_labels(aligned_labels, id2label)
        all_pred_ner.extend(batch_pred_ner)
        all_true_ner.extend(batch_true_ner)

        total_batches += 1

    # Compute average metrics
    metrics = {
        'loss': running_loss / total_batches,
        'class_accuracy': running_class_acc / total_batches,
        'class_precision': running_class_precision / total_batches,
        'class_recall': running_class_recall / total_batches,
        'class_f1': running_class_f1 / total_batches,
        'ner_f1': ner_f1_score(all_true_ner, all_pred_ner)
    }

    return metrics

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    bionlp = load_dataset("tner/bionlp2004")
    class_labels = ["Abstract", "Background", "Methods", "Results", "Conclusion"]
    class_mapping = {label: idx for idx, label in enumerate(class_labels)}
    
    # NER label mapping
    id2label = {
        0: "O", 1: "B-DNA", 2: "I-DNA", 3: "B-protein", 4: "I-protein",
        5: "B-cell_type", 6: "I-cell_type", 7: "B-cell_line", 8: "I-cell_line",
        9: "B-RNA", 10: "I-RNA"
    }

    # Create dataset and dataloader
    train_dataset = NLPMultiTaskDataset(bionlp["train"], class_mapping)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

 
    # Create the model.
    model = MultiTaskTransformer(model_name="roberta-large", num_class_a=num_class_a, num_ner_labels=num_ner_labels).to(device)

# Freeze the transformer backbone.
    for param in model.roberta.parameters():
       param.requires_grad = False


# Configure LoRA on the backbone.
    peft_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="all"
    )

    model = get_peft_model(model, peft_config)

# Create an optimizer that only updates the LoRA adapter parameters and task-specific heads.
# The backbone's original parameters are frozen.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
# automatically collects all trainable parameters, including both the task-specific heads and any adapter modules injected via PEFT.
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-5)


    # Loss functions

    classification_loss_fn = nn.CrossEntropyLoss()
    # For NER, assume no padding in our simulated ner_tags; if padding is used, set ignore_index appropriately.
    ner_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    
    
    # Training parameters
    num_epochs = 3
    ner_loss_weight = 1.0

    # Training loop
    for epoch in range(num_epochs):
        metrics = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            classification_loss_fn=classification_loss_fn,
            ner_loss_fn=ner_loss_fn,
            ner_loss_weight=ner_loss_weight,
            id2label=id2label,
            device=device
        )
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Classification - Accuracy: {metrics['class_accuracy']:.4f}, "
              f"Precision: {metrics['class_precision']:.4f}, "
              f"Recall: {metrics['class_recall']:.4f}, F1: {metrics['class_f1']:.4f}")
        print(f"  NER - F1 (entity-level): {metrics['ner_f1']:.4f}")
        # Print detailed NER classification report
       # print(ner_classification_report(all_true_ner, all_pred_ner))

if __name__ == "__main__":
    main() 