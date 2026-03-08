import yaml


class PromptManager:
    """
    Loads all prompts from config/prompts.yaml.
    Changing any prompt = editing one YAML file, no Python changes needed.
    The version field enables prompt regression tracking.
    """

    def __init__(self, prompts_path: str = "config/prompts.yaml"):
        with open(prompts_path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)
        self.version: str = self._data.get("version", "unknown")

    def get_system_prompt(self) -> str:
        return self._data["system_prompt"].strip()

    def get_answer_prompt(self, context: str, question: str) -> str:
        return self._data["answer_prompt"].format(
            context=context, question=question
        ).strip()

    def get_decline_message(self) -> str:
        return self._data["decline_message"].strip()
