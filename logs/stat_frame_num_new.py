import json
import argparse

def stat_record(dirpath, frame_num):
    frame_num_stat = [0] * (frame_num + 1)
    frame_aux_num_stat = [0] * (frame_num + 1)
    infile = dirpath + '/' + 'stdout.log'
    with open(infile, 'r') as rf:
        for line in rf.readlines():
            if 'Test with config:' in line:
                # reset counter
                frame_num_stat = [0] * (frame_num + 1)
                frame_aux_num_stat = [0] * (frame_num + 1)
                continue
            identifier = 'logging.py:  95: json_stats:'
            if identifier in line:
                data = json.loads(line[52:])
                if "avg_frame_num" in data.keys():
                    continue    # last line.
                if "frame_num" in data.keys():
                    frame_str = str(data["frame_num"])
                    selection = [int(x) for x in frame_str.split('+')[0][1:-1].split(", ")]
                    frame_num_stat[selection[0]] += len(selection)
                    if '+' in frame_str:
                        selection_aux = [int(x) for x in frame_str.split('+')[1][1:-1].split(", ")]
                        selection_aux = [1 if x > 0 else 0 for x in selection_aux]
                    else:
                        selection_aux = [0] * len(selection)
                    frame_aux_num_stat[selection[0]] += sum(selection_aux)

    outfile = dirpath + '/' + 'stat.dat'
    with open(outfile, 'w') as wf:
        wf.write("framenum count fallback selframe auxframe\n")
        for i in range(1, frame_num + 1):
            wf.write("{} {} {} {} {}\n".format(
                i, frame_num_stat[i], frame_aux_num_stat[i], frame_num_stat[i] * i, frame_aux_num_stat[i] * i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirpath",
        type=str,
        metavar="d",
        help="directory contains stdout.log",
    )
    parser.add_argument(
        "--framenum",
        type=int,
        default=8,
        metavar="N",
        help="frame number"
    )
    args = parser.parse_args()

    stat_record(args.dirpath, args.framenum)
