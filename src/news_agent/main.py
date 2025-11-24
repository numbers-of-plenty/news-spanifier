import os
import datetime
import asyncio
import yaml
from openai import AsyncOpenAI
from dotenv import load_dotenv
import argparse

load_dotenv()
client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


parser = argparse.ArgumentParser(
    description="Run news agent with configurable config file"
)
parser.add_argument(
    "--config",
    type=str,
    default="config.yaml",
    help="Path to config YAML file (default: config.yaml)",
)
parser.add_argument(
    "--runs",
    type=int,
    default=None,
    help="How many times each agent should run (overrides the value in config file)",
)
args = parser.parse_args()
config_path = os.path.abspath(args.config)

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

cli_runs = args.runs
config_runs = config.get("runs", 2)
runs = cli_runs if cli_runs is not None else config_runs

async def fetch_agent_news(agent_name: str) -> str:
    topics = config["agents"][agent_name]["topics"]
    topics_str = "\n".join(f"- {t}" for t in topics)

    prompt = f"""
    Search the web for information published strictly within the last 48 hours. Today is
    {datetime.datetime.now().strftime("%Y-%m-%d")}.
    
    Fetch information for the following topics:
    {topics_str}

    Return 10 items per topic.
    Each item should have:
    - headline
    - 2-4 sentence summary
    - publication date (must be inside the last 48 hours)
    - link

    If you can't find enough information satisfying the criteria, ignore the 48 hours criteria and return whatever you can.
    Do not ask any confirmations, search outright. If you want to propose an option, just go with the option. Ask me nothing, I approve automatically.
    Relax the recency requirement if needed. Relax any requirement if needed and do use what you found. If you still can't find 10 items, return whatever you have.
    """

    response = await client.responses.create(
        model="gpt-5-nano",
        tools=[{"type": "web_search"}],
        reasoning={"effort": "low"},
        input=prompt,
        max_output_tokens=3000,
    )
    return f"--- {agent_name} ---\n" + response.output_text


async def main():
    tasks = []
    for _ in range(runs): # 2 synchronous runs per agent by default
        for agent in config["agents"].keys():
            tasks.append(fetch_agent_news(agent))

    all_results = await asyncio.gather(*tasks)
    final_output = "\n\n".join(all_results)
    print(final_output)
    with open("news.txt", "w", encoding="utf-8") as f:
        f.write(final_output)


if __name__ == "__main__":
    asyncio.run(main())
