def combine_text_objects(text_objects):
    index = min(x["index"] for x in text_objects)
    text = ' '.join(x["text"] for x in text_objects)

    left_most = min(x["x0"] for x in text_objects)
    right_most = max(x["x1"] for x in text_objects)
    top = min(x["top"] for x in text_objects)
    doc_top = min(x["doctop"] for x in text_objects)
    bottom = min(x["bottom"] for x in text_objects)

    height = bottom - top # Since (0,0) is the top left of the page, bottom value is higher than top
    width = right_most - left_most # Same for this, righter value is greater than lefter

    return {
        'index': index,
        'text': text,
        'x0': left_most,
        'x1': right_most,
        'top': top,
        'doctop': doc_top,
        'bottom': bottom,
        'upright': True,
        'height': height,
        'width': width,
        'direction': 'ltr'
    }

def group_adjacent_text(text_objects, expected_gap=3):
        nearby_text_objects = sorted(text_objects, key=lambda x: x["x0"])
        grouped_text = []
        # print([text["text"] for text in nearby_text_objects])

        nearest_x1 = nearby_text_objects[0]["x1"]
        words_list_grouping = [nearby_text_objects[0]]
        # for i in range(1, len(nearby_text_objects)):

  
        for i in range(1, len(nearby_text_objects)):
            if nearby_text_objects[i]["x0"] - nearest_x1 < expected_gap:
                # print("NEAREST X1", nearest_x1)
                # print(nearby_text_objects[i]["x0"])
                # print("gap is", nearby_text_objects[i]["x0"] - nearest_x1)
                words_list_grouping.append(nearby_text_objects[i])
                nearest_x1 = nearby_text_objects[i]["x1"]
            else:
                # here we are sorting so the header that is higher on the page ( has lower y value) gets shown first, otherwise it would get shown second
                words_list_grouping.sort(key=lambda x: x["bottom"])
                grouped_text.append(words_list_grouping)
                words_list_grouping = [nearby_text_objects[i]]
                nearest_x1 = nearby_text_objects[i]["x1"]

        grouped_text.append(words_list_grouping) # Append the last grouping since that doesn't get appended above

        output_text = [combine_text_objects(group) for group in grouped_text]
        # print(grouped_headers)
        return output_text