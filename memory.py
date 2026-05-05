from ollama import chat
from loguru import logger


class ConversationMemory:
    def __init__(self, model="mistral:7b"):
        self.history = []
        self.summary = ""
        self.model = model

    def add_user(self, text):
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text):
        self.history.append({"role": "assistant", "content": text})

    def get_messages(self, system_prompt):
        messages = [{"role": "system", "content": system_prompt}]

        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Mémoire:\n{self.summary}"
            })

        messages.extend(self.history[-6:])
        return messages

    def maybe_summarize(self):
        if len(self.history) % 6 != 0:
            return

        try:
            prompt = (
                "Résume cette conversation pour mémoire future. "
                "Garde uniquement les infos importantes.\n\n"
                f"Ancien résumé:\n{self.summary}\n\n"
                f"Conversation:\n{self.history}"
            )

            response = chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Tu résumes efficacement."},
                    {"role": "user", "content": prompt},
                ],
                options={"num_predict": 120},
            )

            self.summary = response["message"]["content"].strip()

     
            self.history = self.history[-4:]

        except Exception as e:
            logger.error(f"Summary error: {e}")