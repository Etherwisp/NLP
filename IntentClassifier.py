#from NLP import World, Obj, resolve_ref, choose_unique, plan_pickup, plan_put, execute_plan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

train_data = [
    # =========================================================================
    # --- 1. PICK_UP Intents  ---
    # =========================================================================
    ("Pick up the green ball", "PICKUP"),
    ("Pick up the red block", "PICKUP"),
    ("grab the table", "PICKUP"),
    ("lift the blue cube", "PICKUP"),
    ("take the yellow pyramid", "PICKUP"),
    ("pick up the small red block", "PICKUP"),
    ("grab the big blue ball", "PICKUP"),
    ("lift up the wooden box", "PICKUP"),
    ("take the green cube", "PICKUP"),
    ("get the red object", "PICKUP"),
    ("pick up that blue pyramid", "PICKUP"),
    ("grab the yellow block", "PICKUP"),
    ("lift the small green ball", "PICKUP"),
    ("take the heavy iron box", "PICKUP"),
    ("pick up the object on the left", "PICKUP"),
    ("grab the sphere", "PICKUP"),
    ("lift the cylinder", "PICKUP"),
    ("take the item", "PICKUP"),
    ("pick up the black block", "PICKUP"),
    ("grab the white ball", "PICKUP"),
    ("lift the orange pyramid", "PICKUP"),
    ("take the tiny cube", "PICKUP"),
    ("pick up the object near the wall", "PICKUP"),
    ("grab the heavy cylinder", "PICKUP"),
    ("pick up the topmost block", "PICKUP"),
    ("lift the rightmost sphere", "PICKUP"),
    ("take the purple item", "PICKUP"),
    ("grab the plastic box", "PICKUP"),
    ("pick up this specific cube", "PICKUP"),
    ("lift the object quickly", "PICKUP"),
    ("fetch the green sphere", "PICKUP"),
    ("retrieve the red cube", "PICKUP"),
    ("seize the yellow block", "PICKUP"),
    ("clutch the blue pyramid", "PICKUP"),
    ("pick up the object in front of you", "PICKUP"),
    ("grab the leftmost object", "PICKUP"),
    ("lift the metallic box", "PICKUP"),
    ("take the glowing ball", "PICKUP"),
    ("pick up the item on the floor", "PICKUP"),
    ("grab the one on the desk", "PICKUP"),
    ("lift the second block", "PICKUP"),
    ("take the last available item", "PICKUP"),
    ("pick up whatever is there", "PICKUP"),
    ("grab that specific ball", "PICKUP"),
    ("lift the heavy pyramid", "PICKUP"),
    ("take the tiny red sphere", "PICKUP"),
    ("pick up the object from the corner", "PICKUP"),
    ("grab the shiny metal cylinder", "PICKUP"),
    ("lift the soft foam block", "PICKUP"),
    ("take the item located upstairs", "PICKUP"),

    # =========================================================================
    # --- 2. PUT_ON Intents ---
    # =========================================================================
    ("put the green ball on the table", "PUT_ON"),
    ("place the red block on the shelf", "PUT_ON"),
    ("put the blue cube on the chair", "PUT_ON"),
    ("set the yellow box on the floor", "PUT_ON"),
    ("place the small pyramid on the big block", "PUT_ON"),
    ("put the red ball on top of the desk", "PUT_ON"),
    ("drop the blue cube on the table", "PUT_ON"),
    ("position the green block on the platform", "PUT_ON"),
    ("put the yellow pyramid on the wooden box", "PUT_ON"),
    ("place the blue sphere on top of the table", "PUT_ON"),
    ("set the red block upon the shelf", "PUT_ON"),
    ("drop the green ball on the surface", "PUT_ON"),
    ("place the object on the counter", "PUT_ON"),
    ("put the cube onto the desk", "PUT_ON"),
    ("place the green ball over the table", "PUT_ON"),
    ("put the heavy box on the flat surface", "PUT_ON"),
    ("set the pyramid on top of the stand", "PUT_ON"),
    ("lay the sphere upon the table", "PUT_ON"),
    ("position the block on top of the box", "PUT_ON"),
    ("put the item on the wooden surface", "PUT_ON"),
    ("place the yellow cube on the upper desk", "PUT_ON"),
    ("put the small ball on the main table", "PUT_ON"),
    ("position the heavy block on the ground", "PUT_ON"),
    ("drop the white sphere on the wooden shelf", "PUT_ON"),
    ("place the object right on the platform", "PUT_ON"),
    ("set the pyramid onto the stage", "PUT_ON"),
    ("put the cyan block on the big desk", "PUT_ON"),
    ("place the item securely on the table", "PUT_ON"),
    ("put the red block on the glass surface", "PUT_ON"),
    ("place the blue sphere on the workbench", "PUT_ON"),
    ("set the green pyramid on top of the crate", "PUT_ON"),
    ("put the tiny cube on the metal plate", "PUT_ON"),
    ("drop the cylinder on the wooden board", "PUT_ON"),
    ("place the heavy box on the table top", "PUT_ON"),
    ("position the yellow item on the high shelf", "PUT_ON"),
    ("put the white ball on the main platform", "PUT_ON"),
    ("set the purple block upon the desk", "PUT_ON"),
    ("place the object on the display stand", "PUT_ON"),
    ("put the black pyramid on the floor tile", "PUT_ON"),
    ("drop the orange cube on the office table", "PUT_ON"),
    ("place the metallic sphere on the pedestal", "PUT_ON"),
    ("put the plastic box on the clean surface", "PUT_ON"),
    ("set the item right on the table edge", "PUT_ON"),
    ("place the red cylinder on top of the crate", "PUT_ON"),
    ("put the blue block on the center table", "PUT_ON"),
    ("position the small ball on the wooden desk", "PUT_ON"),
    ("drop the heavy object on the stage floor", "PUT_ON"),
    ("place the green item on the smooth surface", "PUT_ON"),
    ("put the yellow sphere upon the main counter", "PUT_ON"),
    ("set the final block on the shelf", "PUT_ON"),

    # =========================================================================
    # --- 3. PUT_INSIDE Intents  ---
    # =========================================================================
    ("put the green ball in the box", "PUT_INSIDE"),
    ("place the red block in the drawer", "PUT_INSIDE"),
    ("put the blue cube in the basket", "PUT_INSIDE"),
    ("set the yellow box in the cabinet", "PUT_INSIDE"),
    ("drop the green ball in the bin", "PUT_INSIDE"),
    ("insert the red block in the container", "PUT_INSIDE"),
    ("place the small cube inside the red box", "PUT_INSIDE"),
    ("put the yellow ball into the drawer", "PUT_INSIDE"),
    ("drop the blue cylinder within the basket", "PUT_INSIDE"),
    ("insert the green pyramid inside the cabinet", "PUT_INSIDE"),
    ("place the red item in the container", "PUT_INSIDE"),
    ("put the blue block into the bin", "PUT_INSIDE"),
    ("store the object inside the box", "PUT_INSIDE"),
    ("drop it in the drawer", "PUT_INSIDE"),
    ("put the sphere within the container", "PUT_INSIDE"),
    ("place the tiny block deep in the box", "PUT_INSIDE"),
    ("insert the item into the slot", "PUT_INSIDE"),
    ("slide the cube into the drawer", "PUT_INSIDE"),
    ("drop the ball inside the bin", "PUT_INSIDE"),
    ("store the pyramid within the container", "PUT_INSIDE"),
    ("place the green ball deep inside the container", "PUT_INSIDE"),
    ("put the red cube in the storage bin", "PUT_INSIDE"),
    ("drop the small item into the wooden box", "PUT_INSIDE"),
    ("insert the blue object within the cabinet drawer", "PUT_INSIDE"),
    ("store the pyramid inside the locked chest", "PUT_INSIDE"),
    ("put the cylinder in the metal container", "PUT_INSIDE"),
    ("place the block right into the slot", "PUT_INSIDE"),
    ("put the sphere inside the deep drawer", "PUT_INSIDE"),
    ("insert the small item into the plastic bin", "PUT_INSIDE"),
    ("store the blue block within the wooden box", "PUT_INSIDE"),
    ("drop the red ball into the dark container", "PUT_INSIDE"),
    ("place the yellow pyramid inside the vault", "PUT_INSIDE"),
    ("put the heavy box in the storage unit", "PUT_INSIDE"),
    ("insert the tiny cylinder into the pipe slot", "PUT_INSIDE"),
    ("store the white item inside the basket", "PUT_INSIDE"),
    ("put the black block in the secret compartment", "PUT_INSIDE"),
    ("drop the green cube into the open drawer", "PUT_INSIDE"),
    ("place the metal object within the case", "PUT_INSIDE"),
    ("put the plastic sphere in the garbage bin", "PUT_INSIDE"),
    ("insert the clean item into the testing box", "PUT_INSIDE"),
    ("store the final object inside the container", "PUT_INSIDE"),
    ("put the shiny block deep in the slot", "PUT_INSIDE"),
    ("drop the soft pyramid into the storage drawer", "PUT_INSIDE"),
    ("place the red item inside the metal box", "PUT_INSIDE"),
    ("put the blue object within the office bin", "PUT_INSIDE"),
    ("insert the small sphere into the wooden crate", "PUT_INSIDE"),
    ("store the heavy block inside the safe", "PUT_INSIDE"),
    ("put the green ball in the sorting basket", "PUT_INSIDE"),
    ("drop the yellow cube into the inner compartment", "PUT_INSIDE"),
    ("place the item deep inside the main box", "PUT_INSIDE"),

    # =========================================================================
    # --- 4. PUT_NEXT_TO Intents ---
    # =========================================================================
    ("put the green ball next to the red block", "PUT_NEXT_TO"),
    ("place the blue cube beside the yellow box", "PUT_NEXT_TO"),
    ("move the red block next to the chair", "PUT_NEXT_TO"),
    ("position the green cube beside the table", "PUT_NEXT_TO"),
    ("place the small ball close to the big box", "PUT_NEXT_TO"),
    ("put the yellow block near the red pyramid", "PUT_NEXT_TO"),
    ("place the blue ball beside the green cube", "PUT_NEXT_TO"),
    ("put the red block close to the wall", "PUT_NEXT_TO"),
    ("move the yellow pyramid near the table", "PUT_NEXT_TO"),
    ("position the green block next to the blue box", "PUT_NEXT_TO"),
    ("place it beside the chair", "PUT_NEXT_TO"),
    ("put the cylinder near the sphere", "PUT_NEXT_TO"),
    ("place the box adjacent to the shelf", "PUT_NEXT_TO"),
    ("move the pyramid close to the ball", "PUT_NEXT_TO"),
    ("position the block nearby the table", "PUT_NEXT_TO"),
    ("place the sphere next to the cube", "PUT_NEXT_TO"),
    ("move the item beside the container", "PUT_NEXT_TO"),
    ("set the object close to the desk", "PUT_NEXT_TO"),
    ("put the green ball right next to the red pyramid", "PUT_NEXT_TO"),
    ("place the heavy cube adjacent to the table", "PUT_NEXT_TO"),
    ("move the small sphere close to the wall", "PUT_NEXT_TO"),
    ("position the yellow item nearby the box", "PUT_NEXT_TO"),
    ("place the blue block next to the green sphere", "PUT_NEXT_TO"),
    ("put the cylinder beside the wooden chair", "PUT_NEXT_TO"),
    ("move the pyramid close to the metallic bin", "PUT_NEXT_TO"),
    ("put the red block next to the blue cylinder", "PUT_NEXT_TO"),
    ("place the white ball beside the office desk", "PUT_NEXT_TO"),
    ("move the small cube close to the metal shelf", "PUT_NEXT_TO"),
    ("position the yellow box nearby the armchair", "PUT_NEXT_TO"),
    ("place the green pyramid next to the brown table", "PUT_NEXT_TO"),
    ("put the heavy item beside the dark wall", "PUT_NEXT_TO"),
    ("move the tiny sphere close to the storage bin", "PUT_NEXT_TO"),
    ("position the block adjacent to the main pillar", "PUT_NEXT_TO"),
    ("place the cyan cube next to the glass table", "PUT_NEXT_TO"),
    ("put the purple ball beside the wooden crate", "PUT_NEXT_TO"),
    ("move the object close to the corner desk", "PUT_NEXT_TO"),
    ("position the red item nearby the blue box", "PUT_NEXT_TO"),
    ("place the cylinder next to the open drawer", "PUT_NEXT_TO"),
    ("put the small block beside the metal plate", "PUT_NEXT_TO"),
    ("move the heavy pyramid close to the shelf", "PUT_NEXT_TO"),
    ("position the white item adjacent to the chair", "PUT_NEXT_TO"),
    ("place the green ball next to the office board", "PUT_NEXT_TO"),
    ("put the yellow cube beside the floor lamp", "PUT_NEXT_TO"),
    ("move the dark sphere close to the table edge", "PUT_NEXT_TO"),
    ("position the plastic box nearby the wall", "PUT_NEXT_TO"),
    ("place the blue block next to the metal container", "PUT_NEXT_TO"),
    ("put the red sphere beside the wooden board", "PUT_NEXT_TO"),
    ("move the final item close to the desk", "PUT_NEXT_TO"),
    ("position the object adjacent to the basket", "PUT_NEXT_TO"),
    ("place the tiny cube next to the main box", "PUT_NEXT_TO")
]

Text_train = [item[0] for item in train_data]
Label_train = [item[1] for item in train_data]

# 2. Κατασκευή και εκπαίδευση του μοντέλου μηχανικής μάθησης (Stage 2 Requirement)
intent_model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(random_state=42)
)


intent_model.fit(Text_train, Label_train)

def predict_intent(utterance: str) -> str:
    """Predicts intent (PICKUP, PUT_ON, PUT_INSIDE, PUT_NEXT_TO) from command string."""
    return intent_model.predict([utterance.lower().strip()])[0]

# # 3. Δοκιμή/Αξιολόγηση (Evaluation για την αναφορά σου)
# test_data = [
#     ("pick up the red block", "PICKUP"),
#     ("put the ball on the table", "PUT_ON"),
#     ("place the cube in the box", "PUT_INSIDE")
# ]


test_data = [
    # =========================================================================
    # --- 1. PICKUP Edge Cases ---
    # Challenges: Distractor prepositions ("in", "on", "next to") in spatial descriptors, 
    # and novel verbs ("collect", "hoist", "remove").
    # =========================================================================
    ("pick up the ball inside the basket", "PICKUP"),       # Distractor preposition: "inside"
    ("collect the blue cube from the table", "PICKUP"),      # Novel verb: "collect", distractor: "table"
    ("hoist the heavy metal cylinder", "PICKUP"),            # Novel verb: "hoist"
    ("lift up the item sitting on the shelf", "PICKUP"),     # Distractor preposition: "on"
    ("gather the yellow blocks", "PICKUP"),                  # Novel verb: "gather"
    ("take the red pyramid off the desk", "PICKUP"),         # Distractor phrase: "off the desk"
    ("secure the green object", "PICKUP"),                   # Novel verb: "secure"
    ("grab the box next to the chair", "PICKUP"),           # Distractor preposition: "next to"
    ("remove the sphere from the container", "PICKUP"),     # Novel verb: "remove", distractor: "container"
    ("pick up what is on top of the counter", "PICKUP"),     # Distractor phrase: "on top of"

    # =========================================================================
    # --- 2. PUT_ON Edge Cases ---
    # Challenges: Tricky prepositions ("upon", "onto"), novel verbs ("rest", "mount", "balance"),
    # and target locations often associated with containers ("top of the box").
    # =========================================================================
    ("rest the red block on the platform", "PUT_ON"),        # Novel verb: "rest"
    ("mount the camera on the stand", "PUT_ON"),             # Novel verb: "mount"
    ("put the green ball upon the wooden board", "PUT_ON"),   # Variant preposition: "upon"
    ("place the cube on top of the basket", "PUT_ON"),       # Distractor target: "basket" (often implies INSIDE)
    ("set the sphere onto the shelf", "PUT_ON"),             # Preposition variant: "onto"
    ("leave the blue box on the desk", "PUT_ON"),            # Novel verb: "leave"
    ("balance the pyramid on the cylinder", "PUT_ON"),       # Novel verb: "balance"
    ("put the item over the cabinet", "PUT_ON"),             # Preposition variant: "over"
    ("lay the ball down on the surface", "PUT_ON"),          # Phrasal verb: "lay down on"
    ("drop the block on top of the red box", "PUT_ON"),      # Distractor target: "box" (often implies INSIDE)

    # =========================================================================
    # --- 3. PUT_INSIDE Edge Cases ---
    # Challenges: Novel verbs ("stash", "pack", "stuff", "tuck"), prepositions ("within"), 
    # and complex spatial modifiers ("inside the box next to the shelf").
    # =========================================================================
    ("stash the yellow cube in the drawer", "PUT_INSIDE"),   # Novel verb: "stash"
    ("pack the pyramid into the box", "PUT_INSIDE"),         # Novel verb: "pack"
    ("drop the red ball into the bucket", "PUT_INSIDE"),     # Unseen container: "bucket"
    ("put the object inside the blue desk drawer", "PUT_INSIDE"), # Compound target noun
    ("slide the item into the slot on the table", "PUT_INSIDE"),  # Distractor: "on the table"
    ("deposit the sphere into the container", "PUT_INSIDE"), # Novel verb: "deposit"
    ("stuff the soft block in the basket", "PUT_INSIDE"),    # Novel verb: "stuff"
    ("place the cylinder within the cabinet", "PUT_INSIDE"), # Variant preposition: "within"
    ("put the green item inside the box next to the shelf", "PUT_INSIDE"), # Distractor phrase: "next to"
    ("tuck the small cube into the bag", "PUT_INSIDE"),      # Novel verb: "tuck", unseen container: "bag"

    # =========================================================================
    # --- 4. PUT_NEXT_TO Edge Cases ---
    # Challenges: Prepositions like "by", "alongside", "close by", and verbs like "park", "align".
    # =========================================================================
    ("park the red block by the blue ball", "PUT_NEXT_TO"),  # Novel verb: "park", preposition: "by"
    ("align the cylinder alongside the box", "PUT_NEXT_TO"), # Novel verb: "align", preposition: "alongside"
    ("put the yellow sphere close to the drawer", "PUT_NEXT_TO"), # Distractor target: "drawer"
    ("place the cube right next to the container on the table", "PUT_NEXT_TO"), # Distractors: "container", "on"
    ("set the green block beside the red pyramid", "PUT_NEXT_TO"),
    ("position the object adjacent to the wall", "PUT_NEXT_TO"),
    ("leave the ball near the desk", "PUT_NEXT_TO"),         # Novel verb: "leave"
    ("put the pyramid by the edge of the shelf", "PUT_NEXT_TO"), # Preposition: "by"
    ("place the item alongside the wooden board", "PUT_NEXT_TO"),
    ("move the cylinder close by the cabinet", "PUT_NEXT_TO") # Distractor target: "cabinet"
]


X_test = [t[0] for t in test_data]
y_test = [t[1] for t in test_data]
y_pred = intent_model.predict(X_test)

# Explicitly define label order
labels = ["PICKUP", "PUT_ON", "PUT_INSIDE", "PUT_NEXT_TO"]

# Overall Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"-> Test Accuracy: {accuracy * 100:.1f}%\n")

# Precision, Recall, and F1-Score (Per-Class & Summary)
print("-> Classification Report:")
print(classification_report(y_test, y_pred, labels=labels, digits=2))

# Sentence Breakdown
print("-> Detailed Predictions:")
for txt, true_l, pred_l in zip(X_test, y_test, y_pred):
    print(f"  Φράση: '{txt}' | True: {true_l} | Pred: {pred_l}")