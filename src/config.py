from pydantic import BaseModel

class Config(BaseModel):
    average_capacity_per_node: int = 100
    average_demand_per_node: int = 10
    max_node_count: int = 10