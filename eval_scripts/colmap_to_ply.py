from incremental_sfm.colmap_helpers import read_colmap_points3D

def write_ply(path, points):
    n = len(points)

    with open(path, "w") as f:
        # header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar r\n")
        f.write("property uchar g\n")
        f.write("property uchar b\n")
        f.write("end_header\n")

        # data
        for p in points:
            f.write(f"{p[0]} {p[1]} {p[2]} {int(p[3])} {int(p[4])} {int(p[5])}\n")

if __name__ == "__main__":
    root_name = "../eval_data/eval3_sparse/incremental/result/points3D"
    pc_txt = root_name + ".txt"
    np_points = read_colmap_points3D(pc_txt)
    write_ply(root_name + ".ply", np_points)