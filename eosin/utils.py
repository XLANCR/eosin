import dateparser
from dateparser.search import search_dates


def combine_text_objects(text_objects):
    index = min(x["index"] for x in text_objects)
    text = " ".join(x["text"] for x in text_objects)

    left_most = min(x["x0"] for x in text_objects)
    right_most = max(x["x1"] for x in text_objects)
    top = min(x["top"] for x in text_objects)
    doc_top = min(x["doctop"] for x in text_objects)
    bottom = min(x["bottom"] for x in text_objects)

    height = (
        bottom - top
    )  # Since (0,0) is the top left of the page, bottom value is higher than top
    width = (
        right_most - left_most
    )  # Same for this, righter value is greater than lefter

    return {
        "index": index,
        "text": text,
        "x0": left_most,
        "x1": right_most,
        "top": top,
        "doctop": doc_top,
        "bottom": bottom,
        "upright": True,
        "height": height,
        "width": width,
        "direction": "ltr",
    }


# TODO: Make this work better with multiline text
def group_adjacent_text(text_objects, expected_gap=3):
    nearby_text_objects = sorted(text_objects, key=lambda x: x["x0"])
    grouped_text = []

    nearest_x1 = nearby_text_objects[0]["x1"]
    words_list_grouping = [nearby_text_objects[0]]

    for i in range(1, len(nearby_text_objects)):
        if nearby_text_objects[i]["x0"] - nearest_x1 < expected_gap:
            words_list_grouping.append(nearby_text_objects[i])
            nearest_x1 = nearby_text_objects[i]["x1"]
        else:
            # here we are sorting so the header that is higher on the page ( has lower y value) gets shown first, otherwise it would get shown second
            words_list_grouping.sort(key=lambda x: x["bottom"])
            grouped_text.append(words_list_grouping)
            words_list_grouping = [nearby_text_objects[i]]
            nearest_x1 = nearby_text_objects[i]["x1"]

    grouped_text.append(
        words_list_grouping
    )  # The last grouping doesnt get appended and its required in both cases above so we append here

    output_text = [combine_text_objects(group) for group in grouped_text]
    return output_text


def is_valid_date(date_string):
    parsed = dateparser.parse(date_string)
    stat = "Invalid"
    if parsed:
        info = search_dates(
            date_string,
            settings={
                "REQUIRE_PARTS": ["day", "month", "year"],
                "STRICT_PARSING": True,
            },
        )

        if info:
            stat = "Valid"
        else:
            stat = "Incomplete"

    return stat
