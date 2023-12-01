@ti.kernel
    def initials_OT_3D(self):
        sq_pi = ti.sqrt(4 * ti.math.pi)
        for idx in ti.grouped(self.rho[0]):
            x, y, z = idx

            self.rho[0][idx] = self.RHO0
            self.u[0][idx] = vec3(
                [
                    -(1 + self.eps_p * ti.math.sin(self.h[2] * z))
                    * self.U0
                    * ti.math.sin(self.h[1] * y),
                    (1 + self.eps_p * ti.math.sin(self.h[2] * z))
                    * self.U0
                    * ti.math.sin(self.h[0] * x),
                    self.eps_p * ti.math.sin(self.h[2] * z),
                ]
            )
            self.B[0][idx] = (
                vec3(
                    [
                        -self.B0 * ti.math.sin(self.h[1] * y),
                        self.B0 * ti.math.sin(2.0 * self.h[0] * x),
                        0,
                    ]
                )
                / sq_pi
            )

    

    @ti.kernel
    def initials_SOD(self):
        sq_pi = ti.sqrt(4 * ti.math.pi)
        for idx in ti.grouped(self.rho[0]):
            x, y, z = idx

            rho_ = 1.0
            if x > 0.5 * self.shape[0]:
                rho_ = 0.1

            self.rho[0][idx] = rho_
            self.u[0][idx] = vec3(0)
            self.B[0][idx] = vec3(0)

            self.B[0][idx][0] = 3.0 / sq_pi
            if x < 0.5 * self.shape[0]:
                self.B[0][idx][1] = 5.0 / sq_pi
            else:
                self.B[0][idx][1] = 2.0 / sq_pi

    @ti.kernel
    def initials_rand(self):
        sq_pi = ti.sqrt(4 * ti.math.pi)
        for idx in ti.grouped(self.rho[0]):
            x, y, z = idx

            _rho = self.RHO0 * (1 + 1e-2 * ti.randn(dt=double))
            self.rho[0][idx] = _rho
            self.u[0][idx] = self.U0 * (
                vec3(1)
                + 1e-2
                * vec3([ti.randn(dt=double), ti.randn(dt=double), ti.randn(dt=double)])
            )
            self.B[0][idx] = (
                self.B0
                * (
                    vec3(1)
                    + 1e-2
                    * vec3(
                        [ti.randn(dt=double), ti.randn(dt=double), ti.randn(dt=double)]
                    )
                )
                / sq_pi
            )

    def update_B_staggered_call(self, j=0):
        self.staggered_idx = j
        self.update_B_staggered()
        self.fv_computer.ghosts_periodic_foo_call(self.B_staggered[self.staggered_idx])

    def update_B_call(self, j=0):
        self.staggered_idx = j
        self.update_B()
        self.fv_computer.ghosts_periodic_foo_call(self.B[self.staggered_idx])

    @ti.kernel
    def update_B_staggered_3D(self):
        for idx in ti.grouped(self.B[self.staggered_idx]):
            if not self.fv_computer.check_ghost_idx(idx):
                result = vec3(0)
                for i in ti.ndrange(3):
                    idx_left = idx
                    idx_right = idx + get_basis(i)

                    result[i] = 0.5 * (
                        self.B[self.staggered_idx][idx_left][i]
                        + self.B[self.staggered_idx][idx_right][i]
                    )

                self.B_staggered[self.staggered_idx][idx] = result

    @ti.kernel
    def update_B_3D(self):
        for idx in ti.grouped(self.B[self.staggered_idx]):
            if not self.fv_computer.check_ghost_idx(idx):
                result = vec3(0)
                for i in ti.ndrange(3):
                    idx_left = idx
                    idx_right = idx - get_basis(i)

                    result[i] = 0.5 * (
                        self.B_staggered[self.staggered_idx][idx_left][i]
                        + self.B_staggered[self.staggered_idx][idx_right][i]
                    )

                self.B[self.staggered_idx][idx] = result

    @ti.kernel
    def update_B_staggered_2D(self):
        for idx in ti.grouped(self.B[self.staggered_idx]):
            if not self.fv_computer.check_ghost_idx(idx):
                result = vec3(0)
                for i in ti.ndrange(2):
                    idx_left = idx
                    idx_right = idx + get_basis(i)

                    result[i] = 0.5 * (
                        self.B[self.staggered_idx][idx_left][i]
                        + self.B[self.staggered_idx][idx_right][i]
                    )

                result[2] = 0
                self.B_staggered[self.staggered_idx][idx] = result

    @ti.kernel
    def update_B_2D(self):
        for idx in ti.grouped(self.B[self.staggered_idx]):
            if not self.fv_computer.check_ghost_idx(idx):
                result = vec3(0)
                for i in ti.ndrange(2):
                    idx_left = idx
                    idx_right = idx - get_basis(i)

                    result[i] = 0.5 * (
                        self.B_staggered[self.staggered_idx][idx_left][i]
                        + self.B_staggered[self.staggered_idx][idx_right][i]
                    )

                result[2] = self.B[self.staggered_idx][idx][2]
                self.B[self.staggered_idx][idx] = result



# compute_flux corenrs:
 # corner = idx - vec3i([1, 1, 0]) + get_dx_st_2D(self.axes, j, 0, left=False)
        # v_rho = V_plus_sc_2D(self.v_rho, corner)
        # v_u = V_plus_vec_2D(self.v_u, corner)
        # v_b = V_plus_vec_2D(self.v_b, corner)

        # v_corner = vec7(0)

        # v_corner[0] = v_rho
        # v_corner[1:4] = v_u
        # v_corner[4:] = v_b

        # if ti.static(self.ideal == False):
        #     gradU = self.grad_U(corner)
        #     gradB = self.grad_B(corner)

        #     result[1:4] -= 0.5 * get_mat_col(
        #         self.u_computer.flux_viscous(v_corner, gradU, gradB), self.axes
        #     )

        #     result[4:] -= 0.5 * get_mat_col(
        #         self.B_computer.flux_viscous(v_corner, gradU, gradB), self.axes
        #     )

        # if ti.static(self.hall):
        #     result[4:] -= 0.5 * get_mat_col(
        #         self.B_computer.flux_hall(
        #             v_rho, v_u, v_b, self.grad_B(corner), self.rot_B(corner)
        #         ),
        #         self.axes,
        #     )

        # if ti.static(self.les != NonHallLES.DNS):
        #     gradU = self.grad_U(corner)
        #     gradB = self.grad_B(corner)
        #     rotU = self.rot_U(corner)
        #     rotB = self.rot_B(corner)

        #     result[1:4] -= 0.5 * get_mat_col(
        #         self.u_computer.flux_les(v_corner, gradU, gradB, rotU, rotB),
        #         idx=self.axes,
        #     )

        #     result[4:] -= 0.5 * get_mat_col(
        #         self.B_computer.flux_les(v_corner, gradU, gradB, rotU, rotB),
        #         idx=self.axes,
        #     )