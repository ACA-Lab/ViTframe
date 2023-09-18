import json

MAX_FRAME_NUM = 8


def stat_record(dirpath):
    global MAX_FRAME_NUM
    frame_num_stat = [0] * (MAX_FRAME_NUM + 1)
    infile = dirpath + '/' + 'stdout.log'
    with open(infile, 'r') as rf:
        for line in rf.readlines():
            if 'logging.py:  95: json_stats:' in line:
                data = json.loads(line[52:])
                if "avg_frame_num" in data.keys():
                    continue    # last line.
                if "frame_num_overall" in data.keys():
                    frame_num_stat[int(data["frame_num_overall"])] += 1
                else:
                    if "frame_num" in data.keys():
                        frame_str = str(data["frame_num"])
                        if '+' in frame_str:
                            frame_num_stat[int(frame_str.split('+')[0])] += 1
                        else:
                            frame_num_stat[int(frame_str)] += 1

    outfile = dirpath + '/' + 'stat.dat'
    with open(outfile, 'w') as wf:
        wf.write("framenum count\n")
        for i in range(MAX_FRAME_NUM + 1):
            wf.write(f"{i} {frame_num_stat[i]}\n")

stat_record("k400/ssim_8_direct_0.3")
stat_record("k400/ssim_8_direct_0.4")
stat_record("k400/ssim_8_direct")
stat_record("k400/ssim_8_direct_0.6")
stat_record("k400/ssim_8_direct_0.7")
stat_record("k400/ssim_8_direct_0.8")
stat_record("k400/ssim_8_direct_0.9")
