"""Usage:
    <file-name> --in=INPUT_FILE --out=OUTPUT_FILE
"""
from docopt import docopt

def run(inf, out):
    tab_output = open(out, 'w')

    with open(inf, 'r') as f:
        for line in f:
            items = line.strip().split("\t")
            sent = items[0]
            conf = items[-1]
            args = []
            for arg in items[1:-1]:
                if arg.startswith("V"):
                    pred = arg[arg.index(':') + 1:]
                elif arg.startswith("ARG"):
                    args.append(arg[arg.index(':') + 1:])
            newline = "{}\t{}\t{}\t{}\n".format(sent, conf, pred, "\t".join(args))
            tab_output.write(newline)

    tab_output.close()

if __name__=="__main__":
    args = docopt(__doc__)
    inf = args['--in']
    out = args['--out']
    run(inf, out)
