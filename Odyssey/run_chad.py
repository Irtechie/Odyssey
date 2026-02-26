from odyssey import Odyssey
from odyssey.utils import config
from odyssey.agents.llama import ModelType
from odyssey.utils.logger import get_logger


logger = get_logger("odyssey-chad")


def main() -> None:
    odyssey_chad = Odyssey(
        mc_port=config.get("MC_SERVER_PORT"),
        mc_host=config.get("MC_SERVER_HOST"),
        env_wait_ticks=80,
        skill_library_dir="./skill_library",
        reload=True,
        embedding_dir=config.get("SENTENT_EMBEDDING_DIR"),
        environment="explore",
        resume=False,
        server_port=config.get("NODE_SERVER_PORT"),
        critic_agent_model_name=ModelType.LLAMA3_70B_V1,
        comment_agent_model_name=ModelType.LLAMA3_70B_V1,
        planner_agent_qa_model_name=ModelType.LLAMA3_70B_V1,
        planner_agent_model_name=ModelType.LLAMA3_70B_V1,
        action_agent_model_name=ModelType.LLAMA3_70B_V1,
        username="chad",
    )
    logger.info("starting Odyssey agent with username=chad")
    odyssey_chad.learn()


if __name__ == "__main__":
    main()
