from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re


# ----------------------------
# 1) World model (Blocks World)
# ----------------------------
# KEYWORDS FOR THE PARSER. THESE WORDS ARE THE ONES THAT ALLOW THE PARSER TO RECOGNISE PARTS OF THE TEXT AS ATTRIBUTES
SIZE_RANKS = {"small": 1, "medium": 2, "large": 3}
COLORS = {"red", "green", "blue", "yellow"}
SHAPES = {"cube", "pyramid", "block"}
SURFACES = {"table", "bed", "floor", "window"}

@dataclass(frozen=True)
class Obj:
    name: str
    color: str
    size: str
    shape: str  # "cube", "pyramid", "block" (generic)
    is_surface: bool = False
    is_container: bool = False

@dataclass
class World:
    objects: Dict[str, Obj]
    on: Dict[str, Optional[str]]  # on[x] = y means x is on y; None means on table
    holding: Optional[str] = None
    inside: Dict[str,str] = field(default_factory=dict)  # inside[x] = y means x is inside y
    next_to: Dict[str,Set[str]] = field(default_factory=dict)  # next_to[x] = y means x is next to y)

    def is_next_to(self, a: str, b: str) -> bool:
        """True if a is next to b."""
        return b in self.next_to.get(a, set())


    def put_next_to(self, item: str, other_obj: str) -> None:
        """Add a symetric next-to relationship"""
        if self.holding != item:
            raise RuntimeError(f"Not holding {item}.")
        self.next_to.setdefault(item, set()).add(other_obj)
        self.next_to.setdefault(other_obj, set()).add(item)
        self.holding = None

    def remove_next_to(self, a: str, b: str) -> None:
        """Remove a symetric next-to relationship"""#Maybe not used
        if a in self.next_to and b in self.next_to[a]:
            self.next_to[a].remove(b)
        if b in self.next_to and a in self.next_to[b]:
            self.next_to[b].remove(a)

    
    def is_inside(self, item: str, container: str) -> bool:
        """True if item is inside container."""
        return self.inside.get(item) == container
    
    def inside_of(self, container: str) -> Optional[str]:
        """Return a list of all items currently inside the given container."""
        return [item for item, cont in self.inside.items() if cont == container]

    def can_fit_inside(self, item: str, container: str) -> bool:
        """True if item can fit inside container based on size."""
        cont_obj = self.objects.get(container)
        item_obj = self.objects.get(item)

        if cont_obj.is_container is False:
            return False  # Container is not a container

        if not cont_obj or not item_obj:
            return False  # One of the objects doesn't exist

        item_size_rank = SIZE_RANKS.get(item_obj.size)
        container_size_rank = SIZE_RANKS.get(cont_obj.size)

        return item_size_rank < container_size_rank

    def put_inside(self, item: str, container: str) -> None:
        """Put item inside a container"""
        if self.holding != item:
            raise RuntimeError(f"Not holding {item}.")
        if not self.can_fit_inside(item, container):
            raise RuntimeError(f"{item} cannot fit inside {container}.")
        
        self.inside[item] = container
        self.holding = None

    def is_clear(self, obj_name: str) -> bool:
        """True if nothing is on top of obj_name."""
        if self.objects[obj_name].is_surface:
            return True

        return all(support != obj_name for support in self.on.values())

    def top_of(self, obj_name: str) -> Optional[str]:
        """Return object that is on top of obj_name (if any)."""
        for x, support in self.on.items():
            if support == obj_name:
                return x
        return None

    def put_on(self, x: str, y: str) -> None:
        """Place x on y"""
        if self.holding != x:
            raise RuntimeError(f"Not holding {x}.")
        if y in self.inside:
            raise RuntimeError(f"Cannot place on {y}: {y} is inside {self.inside[y]}. Take it out first.")
        if not self.is_clear(y):
            raise RuntimeError(f"Cannot place on {y}: {y} not clear.")

        self.on[x] = y
        if y in self.next_to:
            for neighbor in list(self.next_to[y]):
                self.next_to.setdefault(x, set()).add(neighbor)
                self.next_to.setdefault(neighbor, set()).add(x)
        self.holding = None


    def pickup(self, x: str) -> None:
        if self.holding is not None:
            raise RuntimeError("Already holding something.")
        if not self.is_clear(x):
            raise RuntimeError(f"Cannot pick up {x}: not clear.")

        #Remove inside relation
        if x in self.inside:
            container = self.inside[x]
            if not self.is_clear(container):
                raise RuntimeError(f"Cannot pick up {x}: container {container} not clear.")
            del self.inside[x]

        #Remove next to relation
        if x in self.next_to:
            for neighbor in list(self.next_to[x]):
                self.remove_next_to(x, neighbor)

        #Remove on relation
        self.on[x] = None

        self.holding = x        


    def describe(self) -> str:
        lines = []
        for name, obj in sorted(self.objects.items()):
            if getattr(obj, "is_surface", False):
                continue

            # 1. Primary Location
            if name == self.holding:
                loc = "is being HELD"
            elif name in self.inside:
                loc = f"is INSIDE {self.inside[name]}"
            elif self.on.get(name) is not None:
                loc = f"is ON {self.on[name]}"
            else:
                loc = "is on the table"

            size_str = f"{obj.size} " if hasattr(obj, "size") and obj.size else ""
            lines.append(f"{name} ({size_str}{obj.color} {obj.shape}) {loc}")

            # 2. Container Contents (if any items are inside this object)
            contained_items = self.inside_of(name)
            if contained_items:
                lines.append(f"   └─ contains: {', '.join(sorted(contained_items))}")

            # 3. Adjacency Relations
            neighbors = sorted(list(self.next_to.get(name, set())))
            if neighbors:
                lines.append(f"   └─ next to: {', '.join(neighbors)}")

        lines.append(f"Holding: {self.holding if self.holding else 'Nothing'}")
        return "\n".join(lines)

# ----------------------------------------
# 2) Reference grounding (simple semantics)
# ----------------------------------------

def resolve_ref(world: World,
    color: Optional[str] = None,
    shape: Optional[str] = None,
    size: Optional[str] = None,
    surface: Optional[str] = None) -> List[str]:
    """Return object names matching the requested properties."""

    if surface and surface in world.objects:
        return[surface]

    matches = []
    for name, obj in world.objects.items():
        if color is not None and obj.color != color:
            continue
        if size is not None and obj.size != size:
            continue
        if shape is not None and obj.shape != shape:
            continue
        matches.append(name)
    return matches
# ----------------------------------------
# 3) Planning / inference (toy planner)
# ----------------------------------------

def plan_pickup(world: World, x: str) -> List[Tuple[str, str, Optional[str]]]:
    """
    Plan to pick up x. If x is not clear, move blockers to table first.
    Returns a plan as a list of (action, obj, target).
    """
    plan = []
    blocker = world.top_of(x)
    if blocker is not None:
        # move blocker away first (to table)
        # if blocker itself isn't clear, recurse
        plan += plan_pickup(world, blocker)
        plan.append(("put_on", blocker, None))  # put on table
    plan.append(("pickup", x, None))
    return plan


def plan_put(world: World, x: str, y: str) -> List[Tuple[str, str, Optional[str]]]:
    """Plan to put x on y, ensuring both are feasible."""
    plan = []
    # Ensure y is clear
    blocker = world.top_of(y)
    if blocker is not None:
        plan += plan_pickup(world, blocker)
        plan.append(("put_on", blocker, None))
    # Ensure we can pick up x
    plan += plan_pickup(world, x)
    plan.append(("put_on", x, y))
    return plan


def execute_plan(world: World, plan: List[Tuple[str, str, Optional[str]]]) -> None:
    for action, obj, target in plan:
        if action == "pickup":
            world.pickup(obj)
        elif action == "put_on":
            if target is None:
                # put on table
                if world.holding != obj:
                    # if we planned it, we should be holding it; but safe-check
                    raise RuntimeError("Planner/executor mismatch.")
                world.on[obj] = None
                world.holding = None
            else:
                world.put_on(obj, target)
        else:
            raise ValueError(f"Unknown action: {action}")
# ----------------------------------------
# 1+2) "Parsing": minimal, rule-based parser
# ----------------------------------------



def parse_command(text: str) -> dict:
    """
    Extremely small rule-based parser for demo:
    - "pick up the red block"
    - "put the blue pyramid on the red cube"
    """
    t = text.lower().replace("?", "").strip()
    tokens = t.split()

    if t.startswith("pick up"):
        # find color + shape (if present)
        color = next((w for w in tokens if w in COLORS), None)
        shape = next((w for w in tokens if w in SHAPES), "block")
        return {"intent": "PICKUP", "ref": {"color": color, "shape": shape}}

    if t.startswith("put"):
        # "put the X on the Y"
        if "on" not in tokens:
            raise ValueError("Expected 'on' in put command.")
        on_i = tokens.index("on")
        left = tokens[1:on_i]     # description of X
        right = tokens[on_i+1:]   # description of Y

        color_x = next((w for w in left if w in COLORS), None)
        shape_x = next((w for w in left if w in SHAPES), "block")
        color_y = next((w for w in right if w in COLORS), None)
        shape_y = next((w for w in right if w in SHAPES), "block")

        return {
            "intent": "PUT_ON",
            "x": {"color": color_x, "shape": shape_x},
            "y": {"color": color_y, "shape": shape_y},
        }

    raise ValueError("Unknown command format.")
# ----------------------------------------
# Dialogue manager: clarification behavior
# ----------------------------------------

def choose_unique(matches: List[str], what: str) -> str:
    if not matches:
        raise ValueError(f"I can't find any {what} that matches your description.")
    if len(matches) > 1:
        # SHRDLU-like clarification
        raise ValueError(f"I don't understand which {what} you mean: {matches}")
    return matches[0]


def interpret_and_act(world: World, utterance: str) -> None:
    parsed = parse_command(utterance)

    # these are like actions, they could become a seperate class that does the work but im too lazy to think OOP
    #I think that like if we had an interface and that interface had a function called action which gets implemented
    #In each class that inherits from it. Then each action could be a class (bruh).
    if parsed["intent"] == "PICKUP":
        ref = parsed["ref"]
        matches = resolve_ref(world, ref["color"], ref["shape"])
        x = choose_unique(matches, f"{ref['color'] or ''} {ref['shape']}".strip())
        plan = plan_pickup(world, x)
        print("PLAN:", plan)
        execute_plan(world, plan)
        print(f"OK. Picked up {x}.")

    elif parsed["intent"] == "PUT_ON":
        mx = resolve_ref(world, parsed["x"]["color"], parsed["x"]["shape"])
        my = resolve_ref(world, parsed["y"]["color"], parsed["y"]["shape"])
        x = choose_unique(mx, f"{parsed['x']['color'] or ''} {parsed['x']['shape']}".strip())
        y = choose_unique(my, f"{parsed['y']['color'] or ''} {parsed['y']['shape']}".strip())
        plan = plan_put(world, x, y)
        print("PLAN:", plan)
        execute_plan(world, plan)
        print(f"OK. Put {x} on {y}.")

    else:
        raise ValueError("Unsupported intent.")
# ----------------------------
# Demo run
# ----------------------------
# class Obj:
#     name: str
#     color: str
#     size: str
#     shape: str  # "cube", "pyramid", "block" (generic)
#     is_surface: bool = False
#     is_container: bool = False

# @dataclass
# class World:
#     objects: Dict[str, Obj]
#     on: Dict[str, Optional[str]]  # on[x] = y means x is on y; None means on table
#     holding: Optional[str] = None
#     inside: Dict[str,str] = field(default_factory=dict)  # inside[x] = y means x is inside y
#     next_to: Dict[str,Set[str]] = field(default_factory=dict)  # next_to[x] = y means x is next to y)



if __name__ == "__main__":
    world = World(
        objects={
            "bed": Obj("bed", "brown", "large", "bed", is_surface=True),
            "table": Obj("table", "brown", "large", "table", is_surface=True),
            "cube_l": Obj("cube_l", "red", "large", "cube"),
            "cube_m": Obj("cube_m", "red", "medium", "cube"),
            "cube_s": Obj("cube_s", "red", "small", "cube"),
            "basket_s": Obj("basket_s", "brown", "small", "basket", is_container=True),
            "basket_l": Obj("basket_l", "brown", "large", "basket", is_container=True),

        },
        on={#cube_s on table
            "cube_s": "table",
            "cube_m": "cube_s",   
            "cube_l": "table",
            "basket_s": "table",
            "basket_l": "table",
        },
        # inside={
        # },
        next_to={
            "cube_s": {"cube_l","basket_s"},
            "cube_l": {"basket_l"}
            }
    )

    print("INITIAL WORLD:\n" + world.describe(), "\n")
    world.pickup("cube_m")
    world.put_on("cube_m", "bed")
    world.pickup("cube_s")
    world.put_inside("cube_s", "basket_l")
    print("\nWORLD AFTER PUTTING CUBE_S INSIDE BASKET_L:\n" + world.describe(), "\n")


    # # 1) Example that triggers clarification (two red cubes exist)
    # try:
    #     interpret_and_act(world, "put the blue pyramid on the red cube")
    # except ValueError as e:
    #     print("SHRDLU:", e)

    # # 2) Disambiguated request
    # print("\nDisambiguating by specifying the target name isn't supported by our tiny grammar,")
    # print("so we instead remove ambiguity by changing the world (like a controlled micro-domain).")

    # # Remove one red cube to resolve ambiguity
    # del world.objects["b2"]
    # del world.on["b2"]

    # print("\nWORLD NOW:\n" + world.describe(), "\n")

    # interpret_and_act(world, "put the red cube on the blue pyramid")

    # print("\nFINAL WORLD:\n" + world.describe())