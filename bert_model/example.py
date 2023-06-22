import torch
from transformers import BertTokenizer, BertForQuestionAnswering


def main():
    # Load the pre-trained model and tokenizer
    model_path = "./bert_model/model/"
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForQuestionAnswering.from_pretrained(model_path)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    while True:
        # Get user input
        question = input("Enter your question (or 'q' to quit): ")
        if question == "q":
            break

        # Tokenize and encode the question
        inputs = tokenizer.encode_plus(question, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

        # Find the start and end indices with the highest logits
        start_index = torch.argmax(start_logits, dim=1).item()
        end_index = torch.argmax(end_logits, dim=1).item()

        # Decode the answer tokens and print the answer
        answer_tokens = input_ids[0][start_index : end_index + 1]
        answer = tokenizer.decode(answer_tokens)
        print("Answer:", answer)
        print()

    print("Goodbye!")


if __name__ == "__main__":
    main()
