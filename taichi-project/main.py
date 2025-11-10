import taichi as ti
import taichi.math as tm
import trimesh
import numpy as np

ti.init(arch='cpu')

n_itr = 32  # Change for the quality
dt = 4e-2 / n_itr
substeps = int(1 / 60 // dt)
solver_itr = 4

gravity = ti.Vector([0, -9.8, 0])
k_damping = 0.1
k_stretching = 1.0
k_bending = 0.1

mesh = trimesh.load('taichi-project/clothMesh.obj', process=False)

vertices = mesh.vertices.astype(np.float32)
faces = mesh.faces
unique_edges = mesh.edges_unique

n = len(vertices)
num_faces = len(faces)
num_edges = len(unique_edges)

# This function is build with GPT-5
def build_other_nodes():
    vertex_faces = [[] for _ in range(n)]
    for fi, (a, b, c) in enumerate(faces):
        vertex_faces[a].append(fi)
        vertex_faces[b].append(fi)
        vertex_faces[c].append(fi)
    for v in range(n):
        vertex_faces[v].sort()
    num_edges = len(unique_edges)
    other_nodes = np.full((num_edges, 2), -1, dtype=np.int32)
    for ei in range(num_edges):
        a, b = unique_edges[ei]
        fa = vertex_faces[a]
        fb = vertex_faces[b]
        ia = ib = 0
        collected = []
        while ia < len(fa) and ib < len(fb):
            if fa[ia] == fb[ib]:
                fi = fa[ia]
                tri = faces[fi]
                for v in tri:
                    if v != a and v != b:
                        collected.append(int(v))
                ia += 1
                ib += 1
            elif fa[ia] < fb[ib]:
                ia += 1
            else:
                ib += 1
        if len(collected) > 0:
            other_nodes[ei, 0] = collected[0]
        if len(collected) > 1:
            other_nodes[ei, 1] = collected[1]
    return other_nodes

other_nodes_np = build_other_nodes()

x = ti.Vector.field(3, dtype=ti.f32, shape=n)
v = ti.Vector.field(3, dtype=ti.f32, shape=n)
w = ti.field(dtype=ti.f32, shape=n)
p = ti.Vector.field(3, dtype=ti.f32, shape=n)
r = ti.Vector.field(3, dtype=ti.f32, shape=n)
r_mat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n)

indices = ti.field(int, shape=num_faces * 3)
edges = ti.Vector.field(2, dtype=int, shape=num_edges)
edge_length = ti.field(dtype=ti.f32, shape=num_edges)
other_nodes = ti.Vector.field(n=2, dtype=int, shape=num_edges)
rest_angles = ti.field(dtype=ti.f32, shape=num_edges)

x.from_numpy(vertices)
indices.from_numpy(faces.flatten())
edges.from_numpy(unique_edges.astype(np.int32))
other_nodes.from_numpy(other_nodes_np.astype(np.int32))

#fix points for debugging
#fixnum = 56  # horizontal
fixnum = 111  # vertical
fix = ti.Vector.field(3, dtype=ti.f32, shape=fixnum)
@ti.kernel
def f():
    for i in range(fixnum):
        fix[i] = x[i]
f()

@ti.kernel
def initialize():
    for i in range(n):
        v[i] = tm.vec3(0.0)
        w[i] = 500.0
    for j in range(num_edges):
        node1, node2 = edges[j]
        edge_length[j] = tm.distance(x[node1], x[node2])

        node1, node2 = edges[j]
        node3, node4 = other_nodes[j]
        if node4 == -1: continue
        p2 = x[node2] - x[node1]
        p3 = x[node3] - x[node1]
        p4 = x[node4] - x[node1]
        p2p3 = tm.cross(p2, p3)
        len_p2p3 = tm.length(p2p3)
        if j==1: print(len_p2p3)
        n1 = p2p3 / len_p2p3
        p2p4 = tm.cross(p2, p4)
        len_p2p4 = tm.length(p2p4)
        n2 = p2p4 / len_p2p4
        d = tm.clamp(n1.dot(n2), -1, 1)
        rest_angles[j] = tm.acos(d)

@ti.kernel
def damp_velocity():
    x_cm = tm.vec3(0.0)
    v_cm = tm.vec3(0.0)
    m_total = 0.0
    for i in range(n):
        x_cm += x[i] / w[i]
        v_cm += v[i] / w[i]
        m_total += 1.0 / w[i]
    x_cm /= m_total
    v_cm /= m_total

    for i in range(n):
        r[i] = x[i] - x_cm
        r_mat[i] = ti.Matrix([[0, -r[i][2], r[i][1]],
                              [r[i][2], 0, -r[i][0]],
                              [-r[i][1], r[i][0], 0] ])
    
    L = tm.vec3(0.0)
    for i in range(n):
        L += tm.cross(r[i], v[i] / w[i])

    I = tm.mat3(0.0)
    for i in range(n):
        I += r_mat[i] @ r_mat[i].transpose() / w[i]
    
    w = I.inverse() @ L
    for i in range(n):
        v[i] += k_damping * (v_cm + tm.cross(w, r[i]) - v[i])


@ti.kernel
def stretching_constraint():
    for j in range(num_edges):
        node1, node2 = edges[j]
        disp = p[node1] - p[node2]
        length = tm.length(disp)
        d = edge_length[j]
        s = 1.0 / (w[node1] + w[node2]) * (length - d) * disp / length
        p[node1] += - w[node1] * s * k_stretching
        p[node2] += + w[node2] * s * k_stretching

@ti.kernel
def bending_constraint():
    for j in range(num_edges):
        node1, node2 = edges[j]
        node3, node4 = other_nodes[j]
        if node4 == -1: 
            continue

        p2 = p[node2] - p[node1]
        p3 = p[node3] - p[node1]
        p4 = p[node4] - p[node1]

        p2p3 = tm.cross(p2, p3)
        len_p2p3 = tm.length(p2p3)
        if len_p2p3 < 1e-6: continue
        n1 = p2p3 / len_p2p3
        p2p4 = tm.cross(p2, p4)
        len_p2p4 = tm.length(p2p4)
        if len_p2p4 < 1e-6: continue
        n2 = p2p4 / len_p2p4
        d = tm.clamp(n1.dot(n2), -1, 1)

        q3 = ( tm.cross(p2, n2) + d * tm.cross(n1, p2) ) / len_p2p3
        q4 = ( tm.cross(p2, n1) + d * tm.cross(n2, p2) ) / len_p2p4
        q2 = -( tm.cross(p3, n2) + d * tm.cross(n1, p3) ) / len_p2p3 - ( tm.cross(p4, n1) + d * tm.cross(n2, p4) ) / len_p2p4
        q1 = -q2 -q3 -q4

        s = ti.sqrt(1 - d**2) * (ti.acos(d) - rest_angles[j]) * k_bending
        w_sum = w[node1] * tm.length(q1)**2 + w[node2] * tm.length(q2)**2 + w[node3] * tm.length(q3)**2 + w[node4] * tm.length(q4)**2
        if w_sum < 1e-6: continue
        p[node1] += -w[node1] * s / w_sum * q1
        p[node2] += -w[node2] * s / w_sum * q2
        p[node3] += -w[node3] * s / w_sum * q3
        p[node4] += -w[node4] * s / w_sum * q4


@ti.kernel
def update_velocity():
    ti.loop_config(parallelize=True)
    for i in range(n):
        v[i] += dt * w[i] * gravity

@ti.kernel
def calculate_position():
    ti.loop_config(parallelize=True)
    for i in range(n):
        p[i] = x[i] + dt * v[i]

@ti.kernel
def update_position():
    ti.loop_config(parallelize=True)
    for i in range(n):
        v[i] = (p[i] - x[i]) / dt
        x[i] = p[i]

@ti.kernel
def fix_points():
    for i in range(fixnum):
        x[i] = fix[i]
        v[i] = tm.vec3(0.0)



def substep():
    update_velocity()
    damp_velocity()
    calculate_position()

    # generate collision constraints here

    for _ in range(solver_itr):
        stretching_constraint()
        bending_constraint()

    update_position()

    # velocity update here


def main():
    window = ti.ui.Window("window", (512, 512), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = window.get_scene()
    camera = ti.ui.Camera()

    current_t = 0.0
    initialize()

    while window.running:

        for _ in range(substeps):
            substep()
            fix_points()
            current_t += dt

        camera.position(0.0, 50.0, 100.0)
        camera.lookat(0.0, 0.0, 0.0)

        #camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        scene.set_camera(camera)

        scene.point_light(pos=(100.0, 100.0, 100.0), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        scene.mesh(x, indices=indices, color=(0.5, 0.5, 0.5), two_sided=True)
        
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
