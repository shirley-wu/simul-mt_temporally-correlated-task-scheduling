import os


def log_str(logging):
    return ", ".join([(k + ": " + ("{:.4f}".format(v) if isinstance(v, float) else str(v)))
                      for k, v in logging.items()])


def verify_dir(path):
    d = os.path.dirname(path)
    if not os.path.exists(d):
        os.makedirs(d)


def format_seconds(s):
    min = int(s // 60)
    if min > 0:
        s = s % 60
    else:
        return "{:f} seconds".format(s)
    h = int(min // 60)
    if h > 0:
        min = int(min % 60)
    else:
        return "{:d} min {:f} seconds".format(min, s)
    d = int(h // 24)
    if d > 0:
        return "{:d} day {:d} hour {:d} min {:f} seconds".format(d, h, min, s)
    else:
        return "{:d} hour {:d} min {:f} seconds".format(h, min, s)
