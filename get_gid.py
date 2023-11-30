import os
import glob


def find_gid(ib_dir):
    for port_fp in glob.glob(os.path.join(ib_dir, "*", "ports", "*")):
        gids_dir = os.path.join(port_fp, "gids")
        gid_attrs_types_dir = os.path.join(port_fp, "gid_attrs", "types")
        for gid in os.listdir(gids_dir):
            gid_fp = os.path.join(gids_dir, gid)
            with open(gid_fp, 'r') as f:
                c = f.read().split(":")
            if c[0] == "0000" and any(x != "0000" for x in c[1:]):
                version_fp = os.path.join(gid_attrs_types_dir, gid)
                if os.path.isfile(version_fp):
                    try:
                        with open(version_fp, 'r') as f:
                            for line in f:
                                if "v2" in line.lower():
                                    return gid
                    except OSError:
                        pass
    # Not found
    return -1


print(find_gid("/sys/class/infiniband/"))
