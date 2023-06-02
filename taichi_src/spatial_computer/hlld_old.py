    @ti.func
    def HLLD(self, flux_rho: ti.template(), flux_u: ti.template(), flux_B: ti.template(), 
        Q_rho_L, Q_u_L, Q_B_L, Q_rho_R, Q_u_R, Q_B_R, D_u, D_v, D_W, i):
        c_f_L = self.get_c_fast(Q_rho_L, Q_u_L, Q_B_L, i)
        c_f_R = self.get_c_fast(Q_rho_R, Q_u_R, Q_B_R, i)
        c_f_max = ti.max(c_f_L, c_f_R)

        yz = get_idx_to_basis(i)
        x = i
        y = yz[0]
        z = yz[1]

        u_R = Q_u_R[x] / Q_rho_R
        v_R = Q_u_R[y] / Q_rho_R
        w_R = Q_u_R[z] / Q_rho_R

        u_L = Q_u_L[x] / Q_rho_L
        v_L = Q_u_L[y] / Q_rho_L
        w_L = Q_u_L[z] / Q_rho_L

        Bx_R = Q_B_R[x]
        By_R = Q_B_R[y]
        Bz_R = Q_B_R[z]

        Bx_L = Q_B_L[x]
        By_L = Q_B_L[y]
        Bz_L = Q_B_L[z]
        

        p_T_L = ti.pow(Q_rho_L, self.gamma) + Q_B_L.norm_sqr() / (2.0 * self.Ma**2)
        p_T_R = ti.pow(Q_rho_R, self.gamma) + Q_B_R.norm_sqr() / (2.0 * self.Ma**2)

        S_L = ti.min(0, ti.min(u_L, u_R) - c_f_max)
        S_R = ti.max(0, ti.max(u_L, u_R) + c_f_max)

        Bx_hll = (S_R*Bx_R-S_L*Bx_L)/(S_R-S_L)

        Theta = ti.min(1, 
            (c_f_max - ti.min(0, D_u)) / (c_f_max - ti.min(D_v, D_w, 0))
        )**4

        S_m = (
            (S_R - u_R)*Q_rho_R*u_R - (S_L - u_L)*Q_rho_L*u_L - Theta*(p_T_R - p_T_L) - Bx_L**2 + Bx_R**2
        ) / (
            (S_R - u_R)*Q_rho_R - (S_L - u_L)*Q_rho_L
        )

        # p_T_star = (
        #     (S_R - u_R)*Q_rho_R*p_T_L - (S_L - u_L)*Q_rho_L*p_T_R + Q_rho_L*Q_rho_R*(S_R - u_R)*(S_L - u_L)*(u_R - u_L)
        # ) / (
        #     (S_R - u_R)*Q_rho_R - (S_L - u_L)*Q_rho_L
        # )

        rho_L_star = Q_rho_L*(S_L - u_L)/(S_L-S_m)
        rho_R_star = Q_rho_R*(S_R - u_R)/(S_R-S_m)

        v_L_star = v_L - Bx_L*By_L*(S_m - u_L)/(Q_rho_L*(S_L - u_L)*(S_L-S_m) - Bx_L**2 + 1e-6)
        w_L_star = w_L - Bx_L*Bz_L*(S_m - u_L)/(Q_rho_L*(S_L - u_L)*(S_L-S_m) - Bx_L**2 + 1e-6)
        By_L_star = By_L*(Q_rho_L*(S_L-u_L)**2 - Bx_L**2)/(Q_rho_L*(S_L - u_L)*(S_L - S_m) - Bx_L**2 + 1e-6)
        Bz_L_star = Bz_L*(Q_rho_L*(S_L-u_L)**2 - Bx_L**2)/(Q_rho_L*(S_L - u_L)*(S_L - S_m) - Bx_L**2 + 1e-6)

        v_R_star = v_R - Bx_R*By_R*(S_m - u_R)/(Q_rho_R*(S_R - u_R)*(S_R-S_m) - Bx_R**2 + 1e-6)
        w_R_star = w_R - Bx_R*Bz_R*(S_m - u_R)/(Q_rho_R*(S_R - u_R)*(S_R-S_m) - Bx_R**2 + 1e-6)
        By_R_star = By_R*(Q_rho_R*(S_R-u_R)**2 - Bx_R**2)/(Q_rho_R*(S_R - u_R)*(S_R - S_m) - Bx_R**2 + 1e-6)
        Bz_R_star = Bz_R*(Q_rho_R*(S_R-u_R)**2 - Bx_R**2)/(Q_rho_R*(S_R - u_R)*(S_R - S_m) - Bx_R**2 + 1e-6)

        sq_rho_L_star = ti.sqrt(rho_L_star)
        sq_rho_R_star = ti.sqrt(rho_R_star)
        signBx = ti.math.sign(Bx_hll)

        S_L_star = S_m - ti.abs(Bx_hll)/sq_rho_L_star
        S_R_star = S_m + ti.abs(Bx_hll)/sq_rho_R_star

        v_star_star = (
            sq_rho_L_star * v_L_star + sq_rho_R_star * v_R_star + (By_R_star - By_L_star)*signBx
        ) / (
            sq_rho_L_star + sq_rho_R_star
        )
        w_star_star = (
            sq_rho_L_star * w_L_star + sq_rho_R_star * w_R_star + (Bz_R_star - Bz_L_star)*signBx
        ) / (
            sq_rho_L_star + sq_rho_R_star
        )

        By_star_star = (
            sq_rho_L_star * By_R_star + sq_rho_R_star * By_L_star 
            + sq_rho_L_star*sq_rho_R_star*(v_R_star - v_L_star)*signBx
        ) / (
            sq_rho_L_star + sq_rho_R_star
        )

        Bz_star_star = (
            sq_rho_L_star * Bz_R_star + sq_rho_R_star * Bz_L_star 
            + sq_rho_L_star*sq_rho_R_star*(w_R_star - w_L_star)*signBx
        ) / (
            sq_rho_L_star + sq_rho_R_star
        )

        result = mat3x3(0)

        Q_u_L_star = vec3(0)
        Q_u_L_star[x] = rho_L_star*S_m
        Q_u_L_star[y] = rho_L_star*v_L_star
        Q_u_L_star[z] = rho_L_star*w_L_star

        Q_B_L_star = vec3(0)
        Q_B_L_star[x] = 0
        Q_B_L_star[y] = By_L_star
        Q_B_L_star[z] = Bz_L_star
        
        Q_u_R_star = vec3(0)
        Q_u_R_star[x] = rho_R_star*S_m
        Q_u_R_star[y] = rho_R_star*v_R_star
        Q_u_R_star[z] = rho_R_star*w_R_star

        Q_B_R_star = vec3(0)
        Q_B_L_star[x] = 0
        Q_B_L_star[y] = By_L_star
        Q_B_L_star[z] = Bz_L_star

        Q_u_L_star_star = vec3(0)
        Q_u_L_star_star[x] = rho_L_star*S_m
        Q_u_L_star_star[y] = rho_L_star*v_star_star
        Q_u_L_star_star[z] = rho_L_star*w_star_star

        Q_u_R_star_star = vec3(0)
        Q_u_R_star_star[x] = rho_R_star*S_m
        Q_u_R_star_star[y] = rho_R_star*v_star_star
        Q_u_R_star_star[z] = rho_R_star*w_star_star

        Q_B_star_star = vec3(0)
        Q_B_star_star[x] = 0
        Q_B_star_star[y] = By_star_star
        Q_B_star_star[z] = Bz_star_star
        
        Q_B_L_old = Q_B_L
        Q_B_L_old[x] = 0
        
        Q_B_R_old = Q_B_R
        Q_B_R_old[x] = 0

        if S_L > 0:
            result[0, 0] = get_vec_col(flux_rho(Q_rho_L, Q_u_L, Q_B_L), i)

            result[:, 1] = get_mat_col(flux_u(Q_rho_L, Q_u_L, Q_B_L), i)
            result[:, 2] = get_mat_col(flux_B(Q_rho_L, Q_u_L, Q_B_L), i)
        elif S_L <= 0 and S_L_star >= 0:
            result[0, 0] = (get_vec_col(flux_rho(Q_rho_L, Q_u_L, Q_B_L), i)
                + S_L*(rho_L_star - Q_rho_L)
            )

            result[:, 1] = (get_mat_col(flux_u(Q_rho_L, Q_u_L, Q_B_L), i)
                + S_L*(Q_u_L_star - Q_u_L)
            )
            result[:, 2] = (get_mat_col(flux_B(Q_rho_L, Q_u_L, Q_B_L), i)
                + S_L*(Q_B_L_star - Q_B_L_old)
            )
        elif S_L_star <= 0 and S_m >= 0:
            result[0, 0] = (get_vec_col(flux_rho(Q_rho_L, Q_u_L, Q_B_L), i)
                + S_L_star*rho_L_star - (S_L_star-S_L)*rho_L_star
                - Q_rho_L*S_L
            )

            result[:, 1] = (get_mat_col(flux_u(Q_rho_L, Q_u_L, Q_B_L), i)
                + S_L_star*Q_u_L_star_star - (S_L_star-S_L)*Q_u_L_star
                - Q_u_L*S_L
            )
            result[:, 2] = (get_mat_col(flux_B(Q_rho_L, Q_u_L, Q_B_L), i)
                + S_L_star*Q_B_star_star - (S_L_star-S_L)*Q_B_L_star
                - Q_B_L_old*S_L
            )
        elif S_m <= 0 and S_R_star >= 0:
            result[0, 0] = (get_vec_col(flux_rho(Q_rho_R, Q_u_R, Q_B_R), i)
                + S_R_star*rho_R_star - (S_R_star-S_R)*rho_R_star
                - Q_rho_R*S_R
            )

            result[:, 1] = (get_mat_col(flux_u(Q_rho_R, Q_u_R, Q_B_R), i)
                + S_R_star*Q_u_R_star_star - (S_R_star-S_R)*Q_u_R_star
                - Q_u_R*S_R
            )
            result[:, 2] = (get_mat_col(flux_B(Q_rho_R, Q_u_R, Q_B_R), i)
                + S_R_star*Q_B_star_star - (S_R_star-S_R)*Q_B_R_star
                - Q_B_R_old*S_R
            )
        elif S_R_star <= 0 and S_R >= 0:
            result[0, 0] = (get_vec_col(flux_rho(Q_rho_R, Q_u_R, Q_B_R), i)
                + S_R*(rho_R_star - Q_rho_R)
            )

            result[:, 1] = (get_mat_col(flux_u(Q_rho_R, Q_u_R, Q_B_R), i)
                + S_R*(Q_u_R_star - Q_u_R)
            )
            result[:, 2] = (get_mat_col(flux_B(Q_rho_R, Q_u_R, Q_B_R), i)
                + S_R*(Q_B_R_star - Q_B_R_old)
            )
        elif S_R <= 0:
            result[0, 0] = get_vec_col(flux_rho(Q_rho_R, Q_u_R, Q_B_R), i)

            result[:, 1] = get_mat_col(flux_u(Q_rho_R, Q_u_R, Q_B_R), i)
            result[:, 2] = get_mat_col(flux_B(Q_rho_R, Q_u_R, Q_B_R), i)

        return result