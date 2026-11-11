"""Orchestrator: runs the agents in order. Usage: python run.py [agent ...]"""
import logging, sys
from datetime import datetime
from core.config import load_config, ROOT
from agents.data_agent import agent as data_agent
from agents.portfolio_agent import agent as portfolio_agent
from agents.modeling_agent import agent as modeling_agent
from agents.news_agent import agent as news_agent
from agents.controller_agent import agent as controller_agent

PIPELINE = {"data": data_agent, "news": news_agent, "portfolio": portfolio_agent,
            "modeling": modeling_agent, "controller": controller_agent}

def main(names):
    (ROOT / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(ROOT / "logs" / f"{datetime.now():%Y-%m-%d}.log"),
                                  logging.StreamHandler()])
    cfg = load_config()
    for n in names or PIPELINE:
        try:
            print(n, "→", PIPELINE[n].run(cfg))
        except Exception:
            logging.exception("Agent %s failed", n)
            if n == "data":   # without data the other agents must not run
                sys.exit(1)

if __name__ == "__main__":
    main(sys.argv[1:])
