import time
import sys
import matplotlib

matplotlib.use("Agg")


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ""
    for i in range(size):
        bar += "█" if i <= done else "░"
    return bar


def stream(message):
    sys.stdout.write(f"\r{message}")


def simple_table(item_tuples):
    border_pattern = "+---------------------------------------"
    whitespace = "                                            "

    headings, cells, = (
        [],
        [],
    )

    for item in item_tuples:

        heading, cell = str(item[0]), str(item[1])

        pad_head = True if len(heading) < len(cell) else False

        pad = abs(len(heading) - len(cell))
        pad = whitespace[:pad]

        pad_left = pad[: len(pad) // 2]
        pad_right = pad[len(pad) // 2 :]

        if pad_head:
            heading = pad_left + heading + pad_right
        else:
            cell = pad_left + cell + pad_right

        headings += [heading]
        cells += [cell]

    border, head, body = "", "", ""

    for i in range(len(item_tuples)):

        temp_head = f"| {headings[i]} "
        temp_body = f"| {cells[i]} "

        border += border_pattern[: len(temp_head)]
        head += temp_head
        body += temp_body

        if i == len(item_tuples) - 1:
            head += "|"
            body += "|"
            border += "+"

    print(border)
    print(head)
    print(border)
    print(body)
    print(border)
    print(" ")


def time_since(started):
    elapsed = time.time() - started
    m = int(elapsed // 60)
    s = int(elapsed % 60)
    if m >= 60:
        h = int(m // 60)
        m = m % 60
        return f"{h}h {m}m {s}s"
    else:
        return f"{m}m {s}s"
