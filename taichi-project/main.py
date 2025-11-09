import taichi as ti
import taichi.math as tm
import trimesh
import numpy as np

ti.init(arch='cpu')

n_itr = 128
dt = 4e-2 / n_itr
substeps = int(1 / 60 // dt)
solver_itr = 4

gravity = ti.Vector([0, -9.8, 0])

mesh = trimesh.load('taichi-project/clothMesh.obj', process=False)

vertices = mesh.vertices.astype(np.float32)
faces = mesh.faces
unique_edges = mesh.edges_unique

n = len(vertices)
num_faces = len(faces)
num_edges = len(unique_edges)

x = ti.Vector.field(3, dtype=ti.f32, shape=n)
v = ti.Vector.field(3, dtype=ti.f32, shape=n)
w = ti.field(dtype=ti.f32, shape=n)
p = ti.Vector.field(3, dtype=ti.f32, shape=n)
r = ti.Vector.field(3, dtype=ti.f32, shape=n)
r_mat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n)

indices = ti.field(int, shape=num_faces * 3)
edges = ti.Vector.field(2, dtype=int, shape=num_edges)
edge_length = ti.field(dtype=ti.f32, shape=num_edges)

x.from_numpy(vertices)
indices.from_numpy(faces.flatten().astype(np.int32))
edges.from_numpy(unique_edges.astype(np.int32))

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
    k_damping = 0.01
    for i in range(n):
        v[i] += k_damping * (v_cm + tm.cross(w, r[i]) - v[i])


@ti.kernel
def stretch_constraint():
    ti.loop_config(parallelize=True)
    for j in range(num_edges):
        node1, node2 = edges[j]
        disp = p[node1] - p[node2]
        length = tm.length(disp)
        d = edge_length[j]
        s = 1.0 / (w[node1] + w[node2]) * (length - d) * disp / length
        p[node1] += - w[node1] * s
        p[node2] += + w[node2] * s

@ti.kernel
def bend_constraint():
    pass

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
        stretch_constraint()

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
