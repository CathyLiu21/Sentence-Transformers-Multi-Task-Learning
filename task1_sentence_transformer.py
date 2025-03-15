import torch
from transformers import BertModel, BertTokenizer

class SentenceTransformer(torch.nn.Module):
    def __init__(self, model_name='bert-base-cased'):
        super(SentenceTransformer, self).__init__()
        # Load the pre-trained transformer model and tokenizer.
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on the token embeddings.
        The pooling excludes padding tokens.
        """
        token_embeddings = model_output.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, sentences):
        # Tokenize sentences with padding and truncation.
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Forward pass through the transformer.
        with torch.no_grad():  # Disable gradients for inference
            model_output = self.model(**encoded_input)

        # Perform pooling to get fixed-length sentence embeddings.
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        return sentence_embeddings

if __name__ == "__main__":
    # Instantiate the model.
    sentence_transformer = SentenceTransformer()

    # Sample sentences for testing.
    sentences = [
        "This is a sentence transformer implementation.",
        "I love machine learning.",
        "Transformers are revolutionizing NLP!"
    ]

    # Obtain the embeddings.
    embeddings = sentence_transformer(sentences)

    # Display the embeddings.
    print("Sentence Embeddings:")
    print(embeddings)
    print("Embedding shape:", embeddings.shape) 