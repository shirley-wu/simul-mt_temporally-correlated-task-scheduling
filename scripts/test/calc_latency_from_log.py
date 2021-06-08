import sys
from metrics import RW2AP, RW2AL


def parse_log(fname):
    results = []
    src = None
    with open(fname, encoding="utf-8") as f:
        for line in f:
            if line.startswith("S-"):
                assert src is None
                src = line.split("\t")[-1].strip()
            elif line.startswith("H-"):
                assert src is not None
                tgt = line.split("\t")[-1].strip()
                results.append((src, tgt))
                src = None
    assert src is None
    return results


def convert_result_to_RW(src, tgt, k):
    n_src = len(src.strip().split(" "))
    n_tgt = len(tgt.strip().split(" "))
    out = []
    # read prefix first
    read = min(n_src, k - 1)
    out += ["R", ] * read
    # read & write iteratively
    written = 0
    while read < n_src and written < n_tgt:
        out += ["R", "W", ]
        read += 1
        written += 1
    # finalize
    if written < n_tgt:
        out += ["W", ] * (n_tgt - written)
    # elif read < n_src:
    #     out += ["R", ] * (n_src - read)
    return " ".join(out)


if __name__ == "__main__":
    log = sys.argv[1]
    k = int(sys.argv[2])
    results = parse_log(log)
    aps = []
    als = []
    for src, tgt in results:
        rw = convert_result_to_RW(src, tgt, k)
        aps.append(RW2AP(rw, add_eos=True))
        als.append(RW2AL(rw, add_eos=True))
    print("AP: %.6f" % (sum(aps) / len(aps)))
    print("AL: %.6f" % (sum(als) / len(als)))
