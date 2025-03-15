import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# Load dataset
bionlp = load_dataset("tner/bionlp2004")
print("Dataset feature keys:", bionlp["train"].features.keys())

# Class labels for task A
class_labels = ["Abstract", "Background", "Methods", "Results", "Conclusion"]
num_class_a = len(class_labels)
print("Number of classification labels:", num_class_a)

# Compute the number of NER labels for task B
all_tags = [tag for example in bionlp["train"] for tag in example["tags"]]
num_ner_labels = max(all_tags) + 1
print("Number of NER labels:", num_ner_labels)

class MultiTaskTransformer(nn.Module):
    def __init__(self, model_name="roberta-large", num_class_a=num_class_a, num_ner_labels=num_ner_labels):
        """
        num_class_a: number of sentence classification labels (e.g., Background, Methods, Results, Conclusion)
        num_ner_labels: number of NER labels
        """
        super(MultiTaskTransformer, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
        self.roberta = AutoModel.from_pretrained(model_name)
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
        Accepts a dict with keys "tokens" and "tags".
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
            if "tags" in inputs:
                tokenized_inputs = self.tokenize_and_align_labels(inputs)
                tokenized_inputs = {k: torch.tensor(v) for k, v in tokenized_inputs.items() if k != "labels"}
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

if __name__ == "__main__":
    # Instantiate the model
    model = MultiTaskTransformer()

    #########Test Task A########
    class_labels = ["Abstract","Background", "Methods", "Results", "Conclusion"]
    class_mapping = {label: idx for idx, label in enumerate(class_labels)}
    # Create an inverse mapping for predictions.
    id2label_a = {v: k for k, v in class_mapping.items()}
    # Extract sentences from the BioNLP dataset.
    # (Here we join tokens from the first 5 examples of the training split.)
    sample_sentences_A = [" ".join(tokens) for tokens in bionlp["train"]["tokens"][:5]]
    print("Extracted Sentences:")
    for sentence in sample_sentences_A:
       print(sentence)
    # Forward pass: Get classification logits for Task A.
    class_logits = model(sample_sentences_A, task='A')

    # Get predicted indices by taking the argmax.
    predicted_class_indices = torch.argmax(class_logits, dim=1).tolist()

# Map predicted indices to label names.
    predicted_labels = [id2label_a[idx] for idx in predicted_class_indices]

    print("\nPredicted Classification Labels for Task A:")
    for sentence, label in zip(sample_sentences_A, predicted_labels):
       print(f"Sentence: {sentence}")
       print(f"Predicted Label: {label}")
       print("---")

    #########Test Task B########
    sample_example_B = bionlp["train"][0]
    print("Extracted Tokens for Task B:")
    print(sample_example_B)
    inputs_for_ner = {"tokens": [sample_example_B["tokens"]], "tags": [sample_example_B["tags"]]}
    ner_logits, tokenized_inputs = model(inputs_for_ner, task='B')
    print("Task B - NER Logits Shape:", ner_logits.shape) 

    # Convert logits to predicted indices by taking the argmax over the last dimension.
    predicted_indices  = torch.argmax(ner_logits, dim=-1)  # shape: [batch_size, seq_len]

    # Define the id2label mapping.
    id2label_b = {
    0: "O",
    1: "B-DNA",
    2: "I-DNA",
    3: "B-protein",
    4: "I-protein",
    5: "B-cell_type",
    6: "I-cell_type",
    7: "B-cell_line",
    8: "I-cell_line",
    9: "B-RNA",
    10: "I-RNA",
    }

# convert predictions for each sentence in the batch:
    predicted_labels = []
    for i in range(predicted_indices.shape[0]):
       sentence_pred_labels = []
    # If you have word_ids (from the tokenizer), you could use it to align further; 
    # here we assume that predicted_indices already correspond to the input tokens.
       for idx in predicted_indices[i].tolist():
        # If idx is -100, it means this token is to be ignored.
          if idx == -100:
            sentence_pred_labels.append("IGN")
          else:
            sentence_pred_labels.append(id2label_b.get(idx, "UNK"))
       predicted_labels.append(sentence_pred_labels)

# Print predicted labels for each sentence.
    for i, labels in enumerate(predicted_labels):
      print(f"Predicted NER for Sample:")
      print(labels)