from src.prompts import normalize_prompt, parse_classes, to_caption


DOT_PROMPT = (
    "person.face.hand.sofa.coffeetable.carpet.tv.tvcabinet.window.curtain."
    "wire.powerstrip.bed.nightstand.desklamp.diningtable.stool.cabinet.bowl."
    "door.shoes.chargingdock.socks.pantcuff.skirthem.migobody.pet.liquidcontainer.foot.trashcan"
)


def test_dot_glued_prompt_splits():
    classes = parse_classes(DOT_PROMPT)
    assert "person" in classes
    assert "face" in classes
    assert "coffeetable" in classes
    assert "tvcabinet" in classes
    assert "trashcan" in classes
    assert len(classes) == 30
    # Must NOT keep the whole string as one class.
    assert classes[0] == "person"
    assert all("." not in c for c in classes)


def test_comma_prompt():
    classes = parse_classes("person, helmet, safety vest")
    assert classes == ["person", "helmet", "safety vest"]


def test_caption_roundtrip():
    caption, classes = normalize_prompt(DOT_PROMPT)
    assert caption.endswith(".")
    assert caption == to_caption(classes)
    assert "person. face." in caption
