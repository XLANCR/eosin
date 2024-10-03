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