
from src.canada_ui import get_graph_data
from src.config import Config
def main():
    config = Config(
        max_node_count=30
    )
    get_graph_data(config)

if __name__ == "__main__":
    main()