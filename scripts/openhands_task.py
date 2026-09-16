"""Explicit OpenHands task entry; never called by the default quality suite."""
from pathlib import Path
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("LLM_MODEL") or not os.environ.get("LLM_API_KEY"):
        parser.error("LLM_MODEL and LLM_API_KEY must be configured; credentials are not printed")
    from openhands.sdk import LLM, Agent, Conversation, Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.terminal import TerminalTool
    llm = LLM(model=os.environ["LLM_MODEL"], api_key=os.environ["LLM_API_KEY"], base_url=os.environ.get("LLM_BASE_URL"))
    agent = Agent(llm=llm, tools=[Tool(name=TerminalTool.name), Tool(name=FileEditorTool.name)])
    conversation = Conversation(agent=agent, workspace=str(args.workspace.resolve()))
    conversation.send_message(args.task.read_text())
    conversation.run()


if __name__ == "__main__":
    main()
