import torch
from transformers import RobertaModel

# model class
class RobertaClass(torch.nn.Module):
    # init method
    def __init__(self):
        super(RobertaClass, self).__init__()
        # Load pre-trained RobertaModel
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        # Define pre-classifier layer
        self.pre_classifier = torch.nn.Linear(768, 768)
        # Define dropout layer
        self.dropout = torch.nn.Dropout(0.3)
        # Define classifier layer
        self.classifier = torch.nn.Linear(768, 1)

    # forward method
    def forward(self, input_ids, attention_mask):
        '''Forward pass of the model'''
        # Perform forward pass through RobertaModel
        output_1 = self.roberta(input_ids=input_ids,
                                attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        # Apply pre-classifier layer
        pooler = self.pre_classifier(pooler)
        # Apply ReLU activation function
        pooler = torch.nn.ReLU()(pooler)
        # Apply dropout
        pooler = self.dropout(pooler)
        # Apply classifier layer
        output = self.classifier(pooler)
        return output
