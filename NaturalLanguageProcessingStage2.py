from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import re
from IntentClassifier import predict_intent


# ----------------------------
# 1) World model (Blocks World)
# ----------------------------
# KEYWORDS FOR THE PARSER. THESE WORDS ARE THE ONES THAT ALLOW THE PARSER TO RECOGNISE PARTS OF THE TEXT AS OBJECT ATTRIBUTES
SIZE_RANKS = {"small": 1, "medium": 2, "large": 3}
COLORS = {"red", "green", "blue", "yellow", "purple", "baby_blue", "black"}
SHAPES = {"cube", "pyramid", "block", "blanket", "keyboard", "plate", "pillow","cup","bin","basketball"}
SURFACES = {"table", "bed", "floor"}

#THIS IS USED TO REPLACE WORDS IN THE USER INPUT WITH THEIR SYNONYMS SO THAT THE PARSER CAN RECOGNIZE THEM.
# FOR EXAMPLE, "grab" IS A SYNONYM FOR "pick up", SO IF THE USER SAYS "grab the red cube", IT WILL BE NORMALIZED TO 
# "pick up the red cube". ALSO BABY BLUE IS A TWO WORD COLOR. THE PARSER DOES NOT RECOGNISE TWO WORD COLORS, SO WE 
# REPLACE IT WITH BABY_BLUE WHICH IS LIKE A SINGLE WORD. THE PARSER WILL THEN RECOGNISE IT AS A COLOR.
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
    "baby blue": "baby_blue"
}


PICKUP_PATTERN = re.compile(
    r"^(?:pick up|take|grab|lift)\s+(?:the\s+|a\s+|an\s+)?(?P<ref>.+)$",
    re.IGNORECASE,
)

PUT_INSIDE_PATTERN = re.compile(
    r"^(?:put|place|set|drop|move)\s+(?:the\s+|a\s+|an\s+)?(?P<x>.+?)\s+(?P<prep>in|inside|into)\s+(?:the\s+|a\s+|an\s+)?(?P<y>.+)$",
    re.IGNORECASE,
)

PUT_ON_PATTERN = re.compile(
    r"^(?:put|place|set|drop|move)\s+(?:the\s+|a\s+|an\s+)?(?P<x>.+?)\s+(?P<prep>on|onto)\s+(?:the\s+|a\s+|an\s+)?(?P<y>.+)$",
    re.IGNORECASE,
)

PUT_NEXT_TO_PATTERN = re.compile(
    r"^(?:put|place|set|drop|move)\s+(?:the\s+|a\s+|an\s+)?(?P<x>.+?)\s+(?P<prep>next to|beside|by|near)\s+(?:the\s+|a\s+|an\s+)?(?P<y>.+)$",
    re.IGNORECASE,
)






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

    def get_base_surface(world: World, obj: str) -> str:
        """Recursively traces down through stacks and containers to find the
        bottom-most surface (e.g., 'table', 'tray') that obj rests on.
        """
        curr = obj
        visited = set()

        while curr and curr not in visited:
            visited.add(curr)

            if hasattr(world, "inside") and curr in world.inside and world.inside[curr]:
                curr = world.inside[curr] #If inside a container, step to the container itself
                continue

            if hasattr(world, "on") and curr in world.on and world.on[curr]:
                curr = world.on[curr] #If stacked on another object/surface, step down
                continue

            break

        return curr if curr else "table"
    

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

    def put_next_to(self, item: str, other_obj: str) -> None:
        """Add a symetric next-to relationship"""
        if self.holding != item:
            raise RuntimeError(f"Not holding {item}.")
        self.next_to.setdefault(item, set()).add(other_obj)
        self.next_to.setdefault(other_obj, set()).add(item)
        self.holding = None

    def put_inside(self, item: str, container: str) -> None:
        """Put item inside a container"""
        if self.holding != item:
            raise RuntimeError(f"Not holding {item}.")
        if not self.can_fit_inside(item, container):
            raise RuntimeError(f"{item} cannot fit inside {container}.")
        
        self.inside[item] = container
        self.holding = None

    def put_on(self, x: str, y: str) -> None:
        """Place x on y"""
        if self.holding != x:
            raise RuntimeError(f"Not holding {x}.")
        if y in self.inside:
            raise RuntimeError(f"Cannot place on {y}: {y} is inside {self.inside[y]}. Take it out first.")
        if not self.is_clear(y):
            raise RuntimeError(f"Cannot place on {y}: {y} not clear.")

        self.on[x] = y
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
    """Plan to pick up x.

    If x or its container is blocked, clears all blockers onto the base surface
    that x (or its container) is sitting on.
    """
    plan = []

    # If already holding x, no action needed
    if getattr(world, "holding", None) == x:
        return plan

    # 1. Identify the base surface where all cleared items should be placed
    base_surface = world.get_base_surface(x)

    # 2. If x is inside a container, clear any objects covering the top of that container
    if hasattr(world, "inside") and x in world.inside:
        container = world.inside[x]
        container_blocker = (
            world.top_of(container) if hasattr(world, "top_of") else None
        )
        if container_blocker is not None:
            plan += plan_pickup(world, container_blocker)
            plan.append(("put_on", container_blocker, base_surface))

    # 3. Clear any objects stacked directly on top of x
    blocker = world.top_of(x) if hasattr(world, "top_of") else None
    if blocker is not None:
        # Recursively clear items sitting on top of the blocker first
        plan += plan_pickup(world, blocker)
        # Place the immediate blocker onto the root base surface
        plan.append(("put_on", blocker, base_surface))

    # 4. Pick up target x
    plan.append(("pickup", x, None))
    return plan

## CHECK PLAN PUT AND IF TIS GOOD WE CHECK CHAIN COMMANDS. AND THEN IF THATS ALSO GOOD WE are clean????
def plan_put(
    world: World, x: str, y: str, intent: str = "PUT_ON"
) -> List[Tuple[str, str, Optional[str]]]:
    """Plans placing object x relative to object y based on the intent:

    PUT_ON, PUT_NEXT_TO, or PUT_INSIDE.
    """
    plan = []
    base_surface = world.get_base_surface(y)

    if intent == "PUT_ON":
        if(world.on[x] == y):
            raise RuntimeError(
                f"Cannot place on {y}: {x} is already on {y}."
            )
        # Check if y is trapped inside a container
        if (
            hasattr(world, "inside")
            and y in world.inside
            and world.inside[y] is not None
        ):
            raise RuntimeError(
                f"Cannot place on {y}: {y} is inside {world.inside[y]}. Take it out first."
            )

        # Check if y has an object on top of it; clear blocker to base surface
        blocker = (
            world.top_of(y) if hasattr(world, "top_of") and not world.objects[y].is_surface else None
        )

        if blocker is not None:
            plan += plan_pickup(world, blocker)
            plan.append(("put_on", blocker, base_surface))

        
        if(world.holding != x):# Pick up x if not already holding it
            plan += plan_pickup(world, x)
        plan.append(("put_on", x, y))

    elif intent == "PUT_NEXT_TO":
        # x is picked up and placed next to y on y's base surface
        plan += plan_pickup(world, x)
        plan.append(("put_next_to", x, y))

    elif intent == "PUT_INSIDE":
        # Check if y is a valid container
        if hasattr(world, "is_container") and not world.is_container(y):
            raise RuntimeError(
                f"Cannot place {x} inside {y}: {y} is not a container."
            )

        # Build full hierarchy chain of nested containers (y -> parent -> grandparent...)
        container_chain = []
        curr = y
        while curr:
            container_chain.append(curr)
            if (
                hasattr(world, "inside")
                and curr in world.inside
                and world.inside[curr] is not None
            ):
                curr = world.inside[curr]
            else:
                break

        # Traverse from outermost container down to y, clearing blockers covering any container
        for container in reversed(container_chain):
            blocker = (
                world.top_of(container)
                if hasattr(world, "top_of")
                else None
            )
            if blocker is not None:
                plan += plan_pickup(world, blocker)
                plan.append(("put_on", blocker, base_surface))

        # Check capacity/fit constraints on target container y
        if hasattr(world, "can_fit_inside") and not world.can_fit_inside(x, y):
            raise RuntimeError(
                f"Cannot place {x} inside {y}: {x} is too large."
            )
        if y in world.inside.values():  
            raise RuntimeError(
                f"Cannot place {x} inside {y}: {y} already has something inside."
            )

        # Pick up x and place inside y
        plan += plan_pickup(world, x)
        plan.append(("put_inside", x, y))

    else:
        raise ValueError(f"Unknown intent: {intent}")

    return plan


def execute_plan(
    world, plan: List[Tuple[str, str, Optional[str]]]
) -> None:
    """Executes planned actions against the world model sequentially."""
    for action, obj, target in plan:
        if action == "pickup":
            world.pickup(obj)

        elif action == "put_on":
            world.put_on(obj, target if target is not None else "wtf bro")

        elif action == "put_inside":
            world.put_inside(obj, target)

        elif action == "put_next_to":
            world.put_next_to(obj, target)

        else:
            raise ValueError(f"Unknown action: {action}")

# ----------------------------------------
# "Parser implementation"
# ----------------------------------------

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

def get_core_phrase(desc: dict) -> str:
    """Builds a basic object descriptor (e.g., 'the purple blanket') ignoring
    spatial modifiers.
    """
    parts = []
    if desc.get("size"):
        parts.append(desc["size"])
    if desc.get("color"):
        parts.append(desc["color"])
    if desc.get("shape"):
        parts.append(desc["shape"])

    phrase = " ".join(parts).strip()
    return f"the {phrase}" if phrase else "the object"

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
           # print(f"IT REPLACED, new command: {cmd}")
        # 3. Parse and execute single sub-command using your existing regex patterns
        parsed = parse_command(cmd)
        parsed_commands.append(parsed)
        
        # Track the object for the next "it" reference
        if "x" in parsed:
            last_mentioned_object = get_core_phrase(parsed["x"])
        elif "ref" in parsed:
            last_mentioned_object = get_core_phrase(parsed["ref"])
            
    return parsed_commands

INTENT_MAP = {
    "on": "PUT_ON",
    "in": "PUT_INSIDE",
    "next to": "PUT_NEXT_TO",
}

def parse_command(text: str) -> dict:
    """Parses an atomic command."""
    intent = predict_intent(text)#Usage of the trained intent classifier to predict the intent of the command
    #print("Heyo intent classified: " + intent)

    # 1. Handle PICKUP intent
    if intent == "PICKUP":
        match_pickup = PICKUP_PATTERN.search(text) or re.search(
            r"(?:pick\s+up|grab|lift|take|fetch)\s+(?P<ref>.+)", text, re.I
        )
        if match_pickup:
            ref_str = match_pickup.group("ref")
            ref_tokens = re.sub(r"\b(the|a|an)\b", "", ref_str).split()
            return {"intent": "PICKUP", "ref": parse_descriptor(ref_tokens)}

    # 2. Handle placement intents (PUT_ON, PUT_INSIDE, PUT_NEXT_TO)
    elif intent in ("PUT_ON", "PUT_INSIDE", "PUT_NEXT_TO"):
        pattern_map = {
            "PUT_INSIDE": PUT_INSIDE_PATTERN,
            "PUT_ON": PUT_ON_PATTERN,
            "PUT_NEXT_TO": PUT_NEXT_TO_PATTERN,
        }

        # First attempt entity extraction using the pattern corresponding to the predicted intent
        match_put = pattern_map[intent].search(text)

        # Fallback to alternative placement patterns if the primary pattern missed
        if not match_put:
            match_put = (
                PUT_INSIDE_PATTERN.search(text)
                or PUT_ON_PATTERN.search(text)
                or PUT_NEXT_TO_PATTERN.search(text)
            )

        if match_put:
            x_str = re.sub(r"\b(the|a|an)\b", "", match_put.group("x"))
            y_str = re.sub(r"\b(the|a|an)\b", "", match_put.group("y"))

            return {
                "intent": intent,  # Preserves the ML-classified intent
                "x": parse_descriptor(x_str.split()),
                "y": parse_descriptor(y_str.split()),
            }

    raise ValueError(f"Could not parse entities for classified intent '{intent}' in text: '{text}'")

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
            "raw_tokens": original_tokens,  
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
        "raw_tokens": original_tokens,  
    }
# ----------------------------------------
# Dialogue manager: clarification behavior
# ----------------------------------------

def choose_unique(matches: List[str], what: str) -> str:
    if not matches:
        raise ValueError(
            f"I can't find any {what} that matches your description."
        )

    if len(matches) > 1:
        choices_str = ", ".join(matches)
        while True:
            user_choice = input(
                f"Which {what} did you mean? Choices: [{choices_str}]. Please type the object name: "
            ).strip()

            if user_choice in matches:
                return user_choice

            print(
                f"'{user_choice}' does not exist in the available choices ({matches}). Please try again.\n"
            )

    return matches[0]

def interpret_and_act(world: World, utterance: str) -> None:
    commands = parse_compound_command(world,utterance)

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
            ##print("PLAN:", plan)
            execute_plan(world, plan)
            #print(f"OK. Picked up {x}.")

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
            #print("PLAN:", plan)
            execute_plan(world, plan)
            #print(f"OK. Executed {intent}: placed {x} relative to {y}.")

        else:
            raise ValueError(f"Unsupported intent: '{intent}'")
    print("command parsed and executed. New world: \n")
    print(world.describe())

#Sentence types
#take the blue pyramid and put it on the red cube
#please take the block and put it inside the basket
#move the small red cube from the table to the bed
#can you take the block inside the basket and put it on the table
#take the basket on the small red cube and put it next to the bed
#take the basket next to the small red cube and put it on the bed


if __name__ == "__main__":
    world = World(
        objects={
            "bed": Obj("bed", color=None, size="large", shape="bed", is_surface=True),
            "table": Obj("table", color="brown", size="large", shape="table", is_surface=True),
            "floor": Obj("floor", color="brown", size=None, shape="floor", is_surface=True),

            "cup": Obj("cup", color="green", size="small", shape="cup",is_container=True),
            "cube1": Obj("cube1", color="red", size="medium", shape="cube"),
            "cube2": Obj("cube2", color="red", size="large", shape="cube"),
            "keyboard": Obj("keyboard", color="black", size="small", shape="keyboard"),
            "plate": Obj("plate", color="black", size="medium", shape="plate", is_container = True),
            "pillow": Obj("pillow", color="baby_blue", size="small", shape="pillow"),
            "blanket": Obj("blanket", color="purple", size="big", shape="blanket"),    
            "medium bag": Obj("medium bag", color="red", size="medium", shape="bag", is_container=True),
            "basketball": Obj("basketball", color="brown", size="large", shape="basketball"),
            "trash bin": Obj("trash bin", color=None, size="large", shape="bin", is_container=True),

        },
        on={#translate to cup ON table etc.
            "cup": "table",
            "cube1": "bed",
            "cube2": "floor", 
            "keyboard": "table",
            "plate": "table",
            "pillow": "bed",
            "blanket": "bed",
            "medium bag": "floor",
            "basketball": "floor",
            "trash bin": "floor"
        },
         inside={
         },
        next_to={
            "medium bag": {"trash bin","basketball"},
            "plate": {"keyboard"}
            }
    )
    print("INITIAL WORLD:\n" + world.describe(), "\n")

    interpret_and_act(world,"can you pick up the baby blue pillow and put it on the table?")
    interpret_and_act(world,"put the cup in the plate")
    interpret_and_act(world,"take the red cube and put it on the table")#ambiguity
    interpret_and_act(world,"put the plate in the trash bin")# throws both plate and cup in the trash bin
    interpret_and_act(world,"take the basketball and put it on the keyboard")
    interpret_and_act(world,"put the keyboard on the bed and put the trash bin on the bed")
    interpret_and_act(world,"pick up the cup and put it on the table")
    interpret_and_act(world,"can you put the purple blanket next to the keyboard?")#Using the trained intent classifier,this gets missclassified
    interpret_and_act(world,"take the purple blanket next to the keyboard and put it on the floor")#Supports spatial relation in object description
    interpret_and_act(world,"put the basketball next to the pillow")

