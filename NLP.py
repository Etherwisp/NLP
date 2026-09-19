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

def resolve_ref(
    world,
    color: Optional[str] = None,
    shape: Optional[str] = None,
    size: Optional[str] = None,
    surface: Optional[str] = None,
    next_to_rel: Optional[str] = None,
    inside_rel: Optional[str] = None,
    **kwargs,  # Gracefully ignores 'raw_tokens' or other extra keys
) -> List[str]:
    """Return object names matching physical attributes and spatial relations."""

    # 1. Direct surface resolution (e.g., "table", "bed")
    if surface and surface in world.objects:
        return [surface]

    matches = []
    for name, obj in world.objects.items():
        # Ignore surfaces when filtering for movable blocks/containers
        if getattr(obj, "is_surface", False):
            continue

        # Physical trait filtering
        if color is not None and getattr(obj, "color", None) != color:
            continue
        if size is not None and getattr(obj, "size", None) != size:
            continue
        if (
            shape is not None
            and shape != "block"
            and getattr(obj, "shape", None) != shape
        ):
            continue

        # Spatial condition: "next to <target>"
        if next_to_rel is not None:
            target_matches = (
                resolve_ref(world, shape=next_to_rel)
                if next_to_rel in SHAPES or next_to_rel in SURFACES
                else [next_to_rel]
            )
            if not any(
                world.is_next_to(name, target)
                for target in target_matches
                if target in world.objects
            ):
                continue

        # Spatial condition: "inside <container>"
        if inside_rel is not None:
            container_matches = (
                resolve_ref(world, shape=inside_rel)
                if inside_rel in SHAPES or inside_rel in SURFACES
                else [inside_rel]
            )
            if not any(
                world.is_inside(name, cont)
                for cont in container_matches
                if cont in world.objects
            ):
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
# "Parser implementation"
# ----------------------------------------

SYNONYMS = {
    # create synonims for shapes and sizes
    "grab": "pick up",
    "take": "pick up",
    "lift": "pick up",
    "place": "put",
    "set": "put",
    "drop": "put",
    "beside": "next to",
    "near": "next to",
    "by": "next to",
    "onto": "on",
    "on top of": "on",
    "inside": "in",
    "into": "in",
}

PICKUP_PATTERN = re.compile(
    r"^(?:pick up|take|grab|lift)\s+(?:the\s+|a\s+|an\s+)?(?P<ref>.+)$",
    re.IGNORECASE,
)

# not used anymore, but could be useful for future reference
# PUT_PATTERN = re.compile(
#     r"^(?:put|place|set|drop|move)\s+(?:the\s+|a\s+|an\s+)?(?P<x>.+?)\s+(?P<prep>next to|beside|by|near|in|inside|into|on|onto)\s+(?:the\s+|a\s+|an\s+)?(?P<y>.+)$",
#     re.IGNORECASE,
# )

PUT_INSIDE_ACTION = re.compile(
    r"^(?:put|place|set|drop|move)\s+(?:the\s+|a\s+|an\s+)?(?P<x>.+?)\s+(?P<prep>in|inside|into)\s+(?:the\s+|a\s+|an\s+)?(?P<y>.+)$",
    re.IGNORECASE,
)

PUT_ON_ACTION = re.compile(
    r"^(?:put|place|set|drop|move)\s+(?:the\s+|a\s+|an\s+)?(?P<x>.+?)\s+(?P<prep>on|onto)\s+(?:the\s+|a\s+|an\s+)?(?P<y>.+)$",
    re.IGNORECASE,
)

PUT_NEXT_TO_ACTION = re.compile(
    r"^(?:put|place|set|drop|move)\s+(?:the\s+|a\s+|an\s+)?(?P<x>.+?)\s+(?P<prep>next to|beside|by|near)\s+(?:the\s+|a\s+|an\s+)?(?P<y>.+)$",
    re.IGNORECASE,
)



def normalize_text(text: str) -> str:
    """Cleans input string and replaces synonym variants with standard terms."""
    text = text.lower().replace("?", "").replace("!", "").strip()#lowercase, remove punctuation, and strip whitespace

#remove polite prefixes like "can you", "could you", or "please" from the beginning of the text
    text = re.sub(r"\b(?:can you|could you|please|hello|hey),?\s*","",text,flags=re.IGNORECASE,).strip()
    # Sort synonyms by length descending so multi-word phrases match before single words
    sorted_synonyms = sorted(
        SYNONYMS.items(), key=lambda item: len(item[0]), reverse=True
    )

    for phrase, canonical in sorted_synonyms:
        # Use \b to ensure whole-word replacement only
        text = re.sub(r"\b" + re.escape(phrase) + r"\b", canonical, text)

    return text

def parse_compound_command(world, sentence: str) -> List[dict]:
    # 1. Normalize text
    text = normalize_text(sentence)
    
    # 2. Split into atomic sub-commands on "and", "then", or commas
    sub_commands = [c.strip() for c in re.split(r'\b(?:and|then)\b|,', text) if c.strip()]
    parsed_commands = []
    last_mentioned_object = None  # Track "it" context across sub-commands

    for cmd in sub_commands:
        # Resolve 'it' if the sub-command references 'it'
        if last_mentioned_object and " it " in f" {cmd} ":
            cmd = re.sub(r'\bit\b', last_mentioned_object, cmd)#replaces it with the exact object mentioned
            
        # 3. Parse and execute single sub-command using your existing regex patterns
        parsed = parse_command(cmd)
        parsed_commands.append(parsed)
        
        # Track the object for the next "it" reference
        if "x" in parsed:
            last_mentioned_object = " ".join(parsed["x"]["raw_tokens"])
        elif "ref" in parsed:
            last_mentioned_object = " ".join(parsed["ref"]["raw_tokens"])
            
    return parsed_commands

INTENT_MAP = {
    "on": "PUT_ON",
    "in": "PUT_INSIDE",
    "next to": "PUT_NEXT_TO",
}

def parse_command(text: str) -> dict:
    """Parses an atomic command."""
    text = normalize_text(text)

    # 1. Match PICKUP commands
    m_pickup = PICKUP_PATTERN.match(text)
    if m_pickup:
        ref_str = m_pickup.group("ref")
        ref_tokens = re.sub(r"\b(the|a|an)\b", "", ref_str).split()
        return {"intent": "PICKUP", "ref": parse_descriptor(ref_tokens)}

    # 2. Match PUT commands (Check container 'in' and surface 'on' first)
    # This prevents 'next to' inside object X from stealing the main action intent
    m_put = (
        PUT_INSIDE_ACTION.match(text)
        or PUT_ON_ACTION.match(text)
        or PUT_NEXT_TO_ACTION.match(text)
    )

    if m_put:
        prep = m_put.group("prep")
        intent = INTENT_MAP[prep]

        x_str = re.sub(r"\b(the|a|an)\b", "", m_put.group("x"))
        y_str = re.sub(r"\b(the|a|an)\b", "", m_put.group("y"))

        return {
            "intent": intent,
            "x": parse_descriptor(x_str.split()),
            "y": parse_descriptor(y_str.split()),
        }

    raise ValueError(f"Could not parse command: '{text}'")

def parse_descriptor(tokens: List[str]) -> dict:
    original_tokens = list(tokens)  # Preserve original tokens before splitting
    text = " ".join(tokens)

    next_to_target = None
    inside_target = None

    # Detect container modifier: "inside <target>" or "in <target>"
    for prep in ["inside", "in"]:
        pattern = f" {prep} "
        if pattern in f" {text} ":
            parts = text.split(prep, 1)
            tokens = parts[0].split()  # Base object tokens before preposition
            inside_target = (
                parts[1].strip().replace("the ", "").replace("a ", "")
            )
            break

    # Detect spatial adjacency modifier: "next to <target>"
    if "next to" in text:
        parts = text.split("next to", 1)
        tokens = parts[0].split()
        next_to_target = parts[1].strip().replace("the ", "").replace("a ", "")

    surface = next((w for w in tokens if w in SURFACES), None)
    if surface:
        return {
            "color": None,
            "shape": "surface",
            "size": None,
            "surface": surface,
            "next_to_rel": next_to_target,
            "inside_rel": inside_target,
            "raw_tokens": original_tokens,  # Added key
        }

    color = next((w for w in tokens if w in COLORS), None)
    shape = next((w for w in tokens if w in SHAPES), "object")
    size = next((w for w in tokens if w in SIZE_RANKS), None)

    return {
        "color": color,
        "shape": shape,
        "size": size,
        "surface": None,
        "next_to_rel": next_to_target,
        "inside_rel": inside_target,
        "raw_tokens": original_tokens,  # Added key
    }
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

# ----------------------------------------
# INTERPRET AND ACT HAS NOT BEEN TESTED, TOGETHER WITH RESOLVE REF THEY NEED TO BE CHECKED AND WHAT IS LEFT IS TO MAKE THE PLANS
# FOR WORLD ACTIONS
# ----------------------------------------  
def interpret_and_act(world: World, utterance: str) -> None:
    commands = parse_compound_command(utterance)

    for parsed in commands:
        intent = parsed["intent"]

        if intent == "PICKUP":
            ref = parsed["ref"]
            # **ref automatically passes color, shape, size, next_to_rel, inside_rel
            matches = resolve_ref(world, **ref)
            x = choose_unique(
                matches, f"{ref['color'] or ''} {ref['shape']}".strip()
            )

            plan = plan_pickup(world, x)
            print("PLAN:", plan)
            execute_plan(world, plan)
            print(f"OK. Picked up {x}.")

        elif intent in ("PUT_ON", "PUT_INSIDE", "PUT_NEXT_TO"):
            # Unpack full descriptor dictionary for both X and Y
            mx = resolve_ref(world, **parsed["x"])
            my = resolve_ref(world, **parsed["y"])

            x_label = (
                f"{parsed['x']['color'] or ''} {parsed['x']['shape']}".strip()
            )
            y_label = (
                f"{parsed['y']['color'] or ''} {parsed['y']['shape']}".strip()
            )

            x = choose_unique(mx, x_label)
            y = choose_unique(my, y_label)

            plan = plan_put(world, x, y, intent)
            print("PLAN:", plan)
            execute_plan(world, plan)
            print(f"OK. Executed {intent}: placed {x} relative to {y}.")

        else:
            raise ValueError(f"Unsupported intent: '{intent}'")
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

#Sentence types
#take the blue pyramid and put it on the red cube
#please pick up the red block next to the chair and put it inside the basket
#move the small red cube from the table to the bed
#can you take the block inside the basket and put it on the table
#take the basket on the small red cube and put it next to the bed


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

    #print(normalize_text("hey, can you please grab the basket on the small red cube, then put it next to the bed"))
    print(parse_compound_command(world, "put the red cube next to the chair inside the container and then put it on the table"))
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