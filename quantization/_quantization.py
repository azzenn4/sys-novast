import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Load your original models
model_6 = "blackstar_6"
model = RobertaForSequenceClassification.from_pretrained(model_6, num_labels=6, ignore_mismatched_sizes=True)
tokenizer_6 = RobertaTokenizer.from_pretrained(model_6)

model_1 = "blackstar_1"
model_ax1 = RobertaForSequenceClassification.from_pretrained(model_1, num_labels=2, ignore_mismatched_sizes=True)
tokenizer_1 = RobertaTokenizer.from_pretrained(model_1)

# Move models to the device (CPU or GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model_ax1 = model_ax1.to(device)

# Perform dynamic quantization on the models
quantized_model_6 = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear},  # Specify layers to quantize (linear layers in this case)
    dtype=torch.qint8  # Use int8 quantization
)

quantized_model_ax1 = torch.quantization.quantize_dynamic(
    model_ax1, 
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save the quantized models' state_dict instead of using save_pretrained
torch.save(quantized_model_6.state_dict(), "quantized_blackstar_6.pth")
torch.save(quantized_model_ax1.state_dict(), "quantized_blackstar_1.pth")

# To load back, you can do the following
# model_6 = RobertaForSequenceClassification.from_pretrained(model_6, num_labels=6)
# model_6.load_state_dict(torch.load("quantized_blackstar_6.pth"))
# model_ax1 = RobertaForSequenceClassification.from_pretrained(model_1, num_labels=2)
# model_ax1.load_state_dict(torch.load("quantized_blackstar_1.pth"))
