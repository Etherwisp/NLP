from NLP import World, Obj, resolve_ref, choose_unique, plan_pickup, plan_put, execute_plan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

train_data = [
    # Pick-Up Intents
    ("Pick up the green ball", "PICKUP"),
    ("Pick up the red block", "PICKUP"),
    ("grab the table", "PICKUP"),
    ("lift the blue cube", "PICKUP"),

    # Put-On Intents
    ("put the green ball on the table", "PUT_ON"),
    ("place the red block on the shelf", "PUT_ON"),
    ("put the blue cube on the chair", "PUT_ON"),
    ("set the yellow box on the floor", "PUT_ON"),

    # Put-Inside Intents
    ("put the green ball in the box", "PUT_INSIDE"),
    ("place the red block in the drawer", "PUT_INSIDE"),
    ("put the blue cube in the basket", "PUT_INSIDE"),
    ("set the yellow box in the cabinet", "PUT_INSIDE"),
    ("drop the green ball in the bin", "PUT_INSIDE"),
    ("insert the red block in the container", "PUT_INSIDE"),

    # Put Next-To Intents
    ("put the green ball next to the red block", "PUT_NEXT_TO"),
    ("place the blue cube beside the yellow box", "PUT_NEXT_TO")
]

Text_train = [item[0] for item in train_data]
Label_train = [item[1] for item in train_data]

# 2. Κατασκευή και εκπαίδευση του μοντέλου μηχανικής μάθησης (Stage 2 Requirement)
intent_model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(random_state=42)
)


intent_model.fit(Text_train, Label_train)


# 3. Δοκιμή/Αξιολόγηση (Evaluation για την αναφορά σου)
test_data = [
    ("pick up the red block", "PICKUP"),
    ("put the ball on the table", "PUT_ON"),
    ("place the cube in the box", "PUT_INSIDE")
]

X_test = [t[0] for t in test_data]
y_test = [t[1] for t in test_data]
y_pred = intent_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"-> Test Accuracy: {accuracy * 100:.1f}%")

for txt, true_l, pred_l in zip(X_test, y_test, y_pred):
    print(f"  Φράση: '{txt}' | True: {true_l} | Pred: {pred_l}")