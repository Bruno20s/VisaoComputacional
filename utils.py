def merge_boxes(boxes, thresh):

    merged = []

    for x, y, w, h in boxes:

        cx = x + w // 2
        cy = y + h // 2

        merged_flag = False

        for i in range(len(merged)):

            mx, my, mw, mh = merged[i]

            mcx = mx + mw // 2
            mcy = my + mh // 2

            if abs(cx - mcx) < thresh and abs(cy - mcy) < thresh:

                nx = min(x, mx)
                ny = min(y, my)

                nw = max(x + w, mx + mw) - nx
                nh = max(y + h, my + mh) - ny

                merged[i] = (nx, ny, nw, nh)

                merged_flag = True
                break

        if not merged_flag:
            merged.append((x, y, w, h))

    return merged