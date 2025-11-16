import taichi as ti
import taichi.math as tm
import trimesh
import numpy as np

ti.init(arch='cpu')

n_itr = 16  # Change for the quality
dt = 4e-2 / n_itr
substeps = int(1 / 60 // dt)  # fps
solver_itr = 4

gravity = ti.Vector([0, -9.8, 0])

# This function is built with GPT-5
def build_other_nodes(num_vertices, faces, unique_edges):
    vertex_faces = [[] for _ in range(num_vertices)]
    for fi, (a, b, c) in enumerate(faces):
        vertex_faces[a].append(fi)
        vertex_faces[b].append(fi)
        vertex_faces[c].append(fi)
    for v in range(num_vertices):
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


@ti.data_oriented
class Object:
    def __init__(self, mesh, is_rigid = False, point_indices_to_fix = []):
        self.mesh = mesh
        self.is_rigid = is_rigid

        # material properties
        self.weight = 500.0
        self.k_damping = 0.5
        self.k_stretching = 1.0
        self.k_bending = 0.2

        # mesh properties
        self.vertices = mesh.vertices.astype(np.float32)
        self.faces = mesh.faces
        self.unique_edges = mesh.edges_unique

        self.num_vertices = len(self.vertices)
        self.num_faces = len(self.faces)
        self.num_edges = len(self.unique_edges)

        self.other_nodes_np = build_other_nodes(self.num_vertices, self.faces, self.unique_edges)

        # init taichi fields
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.w = ti.field(dtype=ti.f32, shape=self.num_vertices)
        self.p = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.r = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.r_mat = ti.Matrix.field(3, 3, dtype=ti.f32, shape=self.num_vertices)

        self.indices = ti.field(int, shape=self.num_faces * 3)
        self.edges = ti.Vector.field(2, dtype=int, shape=self.num_edges)
        self.edge_length = ti.field(dtype=ti.f32, shape=self.num_edges)
        self.other_nodes = ti.Vector.field(n=2, dtype=int, shape=self.num_edges)
        self.rest_angles = ti.field(dtype=ti.f32, shape=self.num_edges)

        # fill in fields
        self.x.from_numpy(self.vertices)
        self.indices.from_numpy(self.faces.flatten())
        self.edges.from_numpy(self.unique_edges.astype(np.int32))
        self.other_nodes.from_numpy(self.other_nodes_np.astype(np.int32))

        # fix points
        self.fixed_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.num_vertices)
        self.is_fixed = ti.field(dtype=int, shape=self.num_vertices)
        self.is_fixed.fill(0)
        for idx in point_indices_to_fix:
            self.is_fixed[idx] = 1
            self.fixed_pos[idx] = self.vertices[idx]

    @ti.kernel
    def initialize(self):
        for i in range(self.num_vertices):
            self.v[i] = tm.vec3(0.0)
            self.w[i] = self.weight
        for j in range(self.num_edges):
            node1, node2 = self.edges[j]
            self.edge_length[j] = tm.distance(self.x[node1], self.x[node2])
            node1, node2 = self.edges[j]
            node3, node4 = self.other_nodes[j]
            if node4 == -1: continue
            p2 = self.x[node2] - self.x[node1]
            p3 = self.x[node3] - self.x[node1]
            p4 = self.x[node4] - self.x[node1]
            p2p3 = tm.cross(p2, p3)
            len_p2p3 = tm.length(p2p3)
            n1 = p2p3 / len_p2p3
            p2p4 = tm.cross(p2, p4)
            len_p2p4 = tm.length(p2p4)
            n2 = p2p4 / len_p2p4
            d = tm.clamp(n1.dot(n2), -1, 1)
            self.rest_angles[j] = tm.acos(d)

    @ti.kernel
    def damp_velocity(self):
        x_cm = tm.vec3(0.0)
        v_cm = tm.vec3(0.0)
        m_total = 0.0
        for i in range(self.num_vertices):
            x_cm += self.x[i] / self.w[i]
            v_cm += self.v[i] / self.w[i]
            m_total += 1.0 / self.w[i]
        x_cm /= m_total
        v_cm /= m_total

        for i in range(self.num_vertices):
            self.r[i] = self.x[i] - x_cm
            self.r_mat[i] = ti.Matrix([[0, -self.r[i][2], self.r[i][1]],
                                [self.r[i][2], 0, -self.r[i][0]],
                                [-self.r[i][1], self.r[i][0], 0] ])
        
        L = tm.vec3(0.0)
        for i in range(self.num_vertices):
            L += tm.cross(self.r[i], self.v[i] / self.w[i])

        I = tm.mat3(0.0)
        for i in range(self.num_vertices):
            I += self.r_mat[i] @ self.r_mat[i].transpose() / self.w[i]
        
        omega = I.inverse() @ L
        for i in range(self.num_vertices):
            self.v[i] += self.k_damping * (v_cm + tm.cross(omega, self.r[i]) - self.v[i])

    @ti.kernel
    def stretching_constraint(self):
        for j in range(self.num_edges):
            node1, node2 = self.edges[j]
            disp = self.p[node1] - self.p[node2]
            length = tm.length(disp)
            d = self.edge_length[j]
            s = 1.0 / (self.w[node1] + self.w[node2]) * (length - d) * disp / length
            self.p[node1] += - self.w[node1] * s * self.k_stretching
            self.p[node2] += + self.w[node2] * s * self.k_stretching

    @ti.kernel
    def bending_constraint(self):
        for j in range(self.num_edges):
            node1, node2 = self.edges[j]
            node3, node4 = self.other_nodes[j]
            if node4 == -1: 
                continue

            p2 = self.p[node2] - self.p[node1]
            p3 = self.p[node3] - self.p[node1]
            p4 = self.p[node4] - self.p[node1]

            p2p3 = tm.cross(p2, p3)
            len_p2p3 = tm.length(p2p3)
            n1 = p2p3 / len_p2p3
            p2p4 = tm.cross(p2, p4)
            len_p2p4 = tm.length(p2p4)
            n2 = p2p4 / len_p2p4
            d = tm.clamp(n1.dot(n2), -1, 1)

            q3 = ( tm.cross(p2, n2) + d * tm.cross(n1, p2) ) / len_p2p3
            q4 = ( tm.cross(p2, n1) + d * tm.cross(n2, p2) ) / len_p2p4
            q2 = -( tm.cross(p3, n2) + d * tm.cross(n1, p3) ) / len_p2p3 - ( tm.cross(p4, n1) + d * tm.cross(n2, p4) ) / len_p2p4
            q1 = -q2 -q3 -q4

            s = ti.sqrt(1 - d**2) * (ti.acos(d) - self.rest_angles[j]) * self.k_bending
            w_sum = self.w[node1] * tm.length(q1)**2 + self.w[node2] * tm.length(q2)**2 + self.w[node3] * tm.length(q3)**2 + self.w[node4] * tm.length(q4)**2
            if w_sum < 1e-6: continue
            self.p[node1] += - self.w[node1] * s / w_sum * q1
            self.p[node2] += - self.w[node2] * s / w_sum * q2
            self.p[node3] += - self.w[node3] * s / w_sum * q3
            self.p[node4] += - self.w[node4] * s / w_sum * q4

    @ti.kernel
    def update_velocity(self):
        for i in range(self.num_vertices):
            self.v[i] += dt * self.w[i] * gravity

    @ti.kernel
    def calculate_position(self):
        for i in range(self.num_vertices):
            self.p[i] = self.x[i] + dt * self.v[i]

    @ti.kernel
    def update_position(self):
        for i in range(self.num_vertices):
            self.v[i] = (self.p[i] - self.x[i]) / dt
            self.x[i] = self.p[i]

    @ti.kernel
    def fix_points(self):
        for i in range(self.num_vertices):
            if self.is_fixed[i] == 1:
                self.x[i] = self.fixed_pos[i]
                self.v[i] = tm.vec3(0.0)

    @ti.kernel
    def solve_ground_collision(self):
        for i in range(self.num_vertices):
            if self.p[i].y < 0.0:
                self.p[i].y = 0.0

    def predict_position(self):
        self.update_velocity()
        self.damp_velocity()
        self.calculate_position()

    def solve_self_constraints(self):
        if not self.is_rigid:
            self.stretching_constraint()
            self.bending_constraint()
        else:
            pass
            # TBA: rigit body constraint


@ti.kernel
def solve_collision_constraints(obj1:ti.template(), obj2:ti.template()): # type: ignore
    pass



def main():
    window = ti.ui.Window("window", (512, 512), vsync=True)
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))
    scene = window.get_scene()
    camera = ti.ui.Camera()

    current_t = 0.0

    # create and initialize objects
    cloth_mesh = trimesh.load('taichi-project/clothMesh.obj', process=False)
    # horizontal 56 / 90 deg 111
    cloth_fixed_points = list(range(56)) + list(range(150,166))
    cloth_obj = Object(cloth_mesh, is_rigid=False, point_indices_to_fix=cloth_fixed_points)
    cloth_obj.initialize()

    bunny_mesh = trimesh.load('taichi-project/bunny.obj', process=False)
    bunny_mesh.vertices *= 200.0
    bunny_mesh.vertices += [0.0, -25.0, 0.0]
    bunny_obj = Object(bunny_mesh, is_rigid=True)
    bunny_obj.initialize()

    # main loop
    while window.running:
        for _ in range(substeps):
            cloth_obj.predict_position()
            #bunny_obj.predict_position()

            # constraint solver
            for _ in range(solver_itr):
                cloth_obj.solve_self_constraints()
                #bunny_obj.solve_self_constraints()

                #cloth_obj.solve_ground_collision()
                #bunny_obj.solve_ground_collision()

                solve_collision_constraints(cloth_obj, bunny_obj)

            cloth_obj.update_position()
            #bunny_obj.update_position()

            # velocity update here: friction (not doing this for now)

            cloth_obj.fix_points()
            #bunny_obj.fix_points()

            # 위치 하나땡겨보기
            current_t += dt


        #camera settings
        camera.position(0.0, 50.0, 100.0)
        #camera.position(0.0, 1.0, 1.0)
        camera.lookat(0.0, 0.0, 0.0)
        camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        scene.set_camera(camera)
        scene.point_light(pos=(100.0, 100.0, 100.0), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))

        # render objects
        scene.mesh(cloth_obj.x, indices=cloth_obj.indices, color=(0.5, 0.5, 0.5), two_sided=True)
        scene.mesh(bunny_obj.x, indices=bunny_obj.indices, color=(0.5, 0.0, 0.0), two_sided=True)

        # render scene
        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    main()
