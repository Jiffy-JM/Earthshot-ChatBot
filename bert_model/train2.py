import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW

# Prepare Dataset
class QADataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        context = example['context']
        question = example['question']
        answer_start = example['answer_start']
        answer_end = len(context)

        # Tokenize and encode the context and question
        inputs = self.tokenizer.encode_plus(question, context, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_start": torch.tensor(answer_start),
            "answer_end": torch.tensor(answer_end)
        }

data = [
    {
        "context": "Earthshot is a non-profit organization that is using video games and immersive experiences to inspire people to take action on climate change.",
        "question": "What is Earthshot?",
        "answer_start": 0
    },
    {
        "context": "Earthshot believes that the current climate change messaging is not working.",
        "question": "Why does Earthshot believe that the current climate change messaging is not working?",
        "answer_start": 0
    },
    {
        "context": "Earthshot is currently developing a series of games that will be released in 2023.",
        "question": "What is Earthshot developing?",
        "answer_start": 0
    },
        {
        "context": "Earthshot is a non-profit organization that is using video games and immersive experiences to inspire people to take action on climate change.",
        "question": "What is Earthshot?",
        "answer_start": 0
    },
    {
        "context": "Earthshot believes that the current climate change messaging is not working.",
        "question": "Why does Earthshot believe that the current climate change messaging is not working?",
        "answer_start": 0
    },
    {
        "context": "Earthshot is currently developing a series of games that will be released in 2023.",
        "question": "What is Earthshot developing?",
        "answer_start": 0
    },
    {
        "context": "Earthshot is looking for partners to help them reach a wider audience.",
        "question": "Who is Earthshot looking for partners?",
        "answer_start": 0
    },
    {
        "context": "Earthshot is a unique and innovative organization that is using games and immersive experiences to inspire people to take action on climate change.",
        "question": "What is unique about Earthshot?",
        "answer_start": 0
    },
    {
        "context": "Earthshot is committed to addressing climate change in a way that is equitable and inclusive.",
        "question": "How is Earthshot addressing climate change in an equitable and inclusive way?",
        "answer_start": 0
    },
    {
        "context": "Earthshot is a non-profit organization that is funded through a combination of in-kind contributions, individual donations, and grants.",
        "question": "How is Earthshot funded?",
        "answer_start": 0
    },
    {
        "context": "Earthshot believes that current climate change messaging is not working and that people are more likely to act when they see a positive vision of the future.",
        "question": "What does Earthshot believe about climate change messaging?",
        "answer_start": 0
    },
    {
        "context": "Earthshot believes that games are a powerful way to engage people and teach them about climate change.",
        "question": "Why does Earthshot believe that games are a powerful way to engage people?",
        "answer_start": 0
    },
    {
        "context": "Earthshot is currently working closely with the devlopment of a series of games that will be released in the future",
        "question": "What is Earthshot doing to address climate change?",
        "answer_start": 0
    },
    {
        "context": "Earthshot is looking for partners to help them reach a wider audience.",
        "question": "Who is Earthshot looking for partners?",
        "answer_start": 0
    },
    {
        "context": "Here are some additional details: Earthshot believes that the current climate change messaging is not working because it is too focused on the negative consequences of climate change. Earthshot believes that people are more likely to act when they see a positive vision of the future. Earthshot is using games and immersive experiences to create a positive vision of the future that will inspire people to take action on climate change. Earthshot believes that games are a powerful way to engage people because they are fun, interactive, and educational. Earthshot is currently developing a series of games that will be released in 2023. Earthshot is looking for partners to help them reach a wider audience. Earthshot is a non-profit organization that is using games and immersive experiences to inspire people to take action on climate change. The organization has a board of directors that includes experts in the entertainment, gaming, climate policy, sustainability, and technology industries. Earthshot is currently developing a series of hand-held games, as well as other initiatives such as podcasts, a technology showcase, and public relations and social networking campaigns. In the long term, Earthshot plans to develop museum-quality interactive and immersive experiences for museums, county fairs, and schools.",
        "question": "What does Earthshot believe about climate change messaging?",
        "answer_start": 0
    },
    {
        "context": "Here are some of the specific initiatives that Earthshot is working on: Hand-held games: Earthshot is currently working on a series of hand-held games that will be used to educate people about climate change and inspire them to take action. The games will be available on a variety of platforms, including smartphones, tablets, and gaming consoles. Earthshot Now podcasts: Earthshot Now is a podcast series that features interviews with celebrities, athletes, scientists, technologists, policymakers, and thought leaders about climate change. The podcast is available on a variety of platforms, including Spotify, Apple Podcasts, and Google Podcasts. Technology showcase: Earthshot is co-hosting a technology showcase with John Preston, a former Director of Technology Development at MIT. The showcase will feature discussions about zero- and low-carbon technology options that are currently available for US infrastructure projects. Public relations and social networking: SamsonPR is providing Earthshot with a public relations plan to activate social, digital, and print media. The plan will position Earthshot as a thought leader in the fight against climate change. Cleantech platform for existing game franchises: Earthshot is exploring the possibility of developing cleantech platforms that could be used by existing game franchises. For example, Earthshot could develop an electric car platform that could be an add-on to existing racing games such as Gran Turismo or Microsoft Forza. Character and narrative intellectual property: With sufficient funding, Earthshot intends to develop its own characters and narratives for possible use in one or more of its games and experiences. Museum-quality interactive and immersive experiences: As envisioned prior to COVID, Earthshot plans to develop museum-quality interactive and immersive experiences for museums, county fairs, and schools. These multi-platform experiences would engage participants in individual and team-oriented problem solving that demonstrates both the personal impact of climate change and the prospects for using exciting cleantech to decarbonize the economy. Norah's Arc: Earthshot may have the opportunity to work with a new film studio on the production of a film that has as its focus a multi-player game, in which players build their own wild animal park. In the story, the female lead character strives to save the world from a future in which there are no wild animals. Earthshot would help with game development and advise on environmental aspects of the film and game.",
        "question": "What are some of the specific initiatives that Earthshot is working on?",
        "answer_start": 0,
        "answer_end": 114,
    },
    # ... additional training examples ...
]

dataset = QADataset(data)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Initialize Model and Optimizer
model = BertForQuestionAnswering.from_pretrained("bert-base-uncased")
optimizer = AdamW(model.parameters(), lr=2e-5)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

num_epochs = 30
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        answer_start = batch["answer_start"].to(device)
        answer_end = batch["answer_end"].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=answer_start, end_positions=answer_end)
        loss = outputs.loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs} | Loss: {loss.item()}")

# Save the trained model
output_dir = "/Users/wrk/Desktop/Earthshot-ChatBot/bert_model"
model.save_pretrained(output_dir)
