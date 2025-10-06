from transformers import pipeline

pipe = pipeline(
    "text-classification",
    model="./distilled-distilbert-3layer",
    tokenizer="./distilled-distilbert-3layer",
    device=0  # set to 0 for CUDA, -1 for CPU
)

a = pipe("I feel incredibly happy about the news!")
print (a)
