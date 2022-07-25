import json
from typing import List, Optional

from attrs import define

from .house import House


@define
class HouseGroup:
    houses: List[House]

    def to_debug_json(self, filename: Optional[str] = None) -> str:
        json_data = [{"action": "Initialize", "procedural": True}]
        for house in self.houses:
            json_data.append({"action": "CreateHouse", "house": house.data})
        json_rep = json.dumps(json_data, sort_keys=True, indent=4)
        if filename:
            with open(filename, "w") as f:
                f.write(json_rep)
        return json_rep
