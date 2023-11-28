 def get_filtered_shape(self, filter_size: vec3i):
        new_shape = [0, 0, 0]
        for i in range(len(filter_size)):
            new_shape[i] = max(1, int(self.shape[i] // filter_size[i]))
        
        new_h = vec3(0)

        for i in range(len(new_h)):
            new_h[i] = self.shape[i] * self.h[i] / new_shape[i]

        return tuple(new_shape), new_h

@ti.func
    def foo_filter_1d(self, h, new_h, i, shape, idx):
        idx_i = get_elem_1d(idx, i)
        shape_i = get_elem_1d(shape, i)
        h_i = get_elem_1d(h, i)
        new_h_i = get_elem_1d(new_h, i)

        idx_i_left = idx_i
        idx_i_right = 0
        if (idx_i + 1) < shape_i:
            idx_i_right = idx_i + 1
        else:
            idx_i_right = idx_i

        left_i_new = ti.floor(idx_i_left * h_i / new_h_i)
        right_i_new = ti.floor(idx_i_right * h_i / new_h_i)

        left_delta = h_i
        right_delta = double(0.0)
        if left_i_new != right_i_new:
            left_delta = right_i_new*new_h_i - idx_i*h_i
            right_delta = h_i - left_delta

        return vec4([left_i_new, right_i_new, left_delta, right_delta])

    @ti.kernel
    def knl_foo_filter(self, foo: ti.template(), out: ti.template(), h: vec3, new_h: vec3):
        dV_new = new_h[0]*new_h[1]*new_h[2]
        for i, j, k in ti.ndrange(self.filter_old_shape[0], 
            self.filter_old_shape[1], self.filter_old_shape[2]):
            idx = [i, j, k]

            x_vec = self.foo_filter_1d(h, new_h, 0, self.filter_old_shape, idx)
            y_vec = self.foo_filter_1d(h, new_h, 1, self.filter_old_shape, idx)
            z_vec = self.foo_filter_1d(h, new_h, 2, self.filter_old_shape, idx)

            for i, j ,k in ti.static(ti.ndrange(2, 2, 2)):
                idx_new = [0, 0, 0]
                dV = double(1.0)
                idx_new[0] = ti.cast(x_vec[i], int)
                dV *= x_vec[i+2]
                idx_new[1] =ti.cast(y_vec[j], int)
                dV *= y_vec[j+2]
                idx_new[2] = ti.cast(z_vec[k], int)
                dV *= z_vec[k+2]
                
                out[idx_new] += (foo(idx) * dV )/ dV_new

    def foo_filter(self, foo, out, shape, new_h, h):
        self.filter_old_shape = shape
        self.knl_foo_filter(foo, out, h, new_h)

    def create_filtered_sc(self, filter_size):
        new_shape, new_h = self.get_filtered_shape(filter_size)

        new_field = ti.field(dtype=double, shape=tuple(new_shape))
        new_field.fill(0)
        return new_field

    def filter_sc(self, foo, filter_size, out):
        new_shape, new_h = self.get_filtered_shape(filter_size)

        self.foo_filter(foo, out, self.shape, new_h, self.h)
    
    def create_filtered_vec(self, filter_size):
        new_shape, new_h = self.get_filtered_shape(filter_size)

        new_field = ti.Vector.field(n=3, dtype=double, shape=tuple(new_shape))
        new_field.fill(0)
        return new_field

    def filter_vec(self, foo, filter_size, out):
        new_shape, new_h = self.get_filtered_shape(filter_size)

        self.foo_filter(foo, out, self.shape, new_h, self.h)
    
    def create_filtered_mat(self, filter_size):
        new_shape, new_h = self.get_filtered_shape(filter_size)

        new_field = ti.Matrix.field(n=3, m=3, dtype=double, shape=tuple(new_shape))
        new_field.fill(0)
        return new_field

    def filter_mat(self, foo, filter_size, out):
        new_shape, new_h = self.get_filtered_shape(filter_size)
        self.foo_filter(foo, out, self.shape, new_h, self.h)

    def filter_favre_sc(self, foo, rho_filtered, filter_size, out):
        self.filter_sc(foo, filter_size, out)

        field_div(out, rho_filtered)

    def filter_favre_vec(self, foo, rho_filtered, filter_size, out):
        self.filter_vec(foo, filter_size, out)

        field_div(out, rho_filtered)
    
    def filter_favre_mat(self, foo, rho_filtered, filter_size, out):
        self.filter_mat(foo, filter_size, out)

        field_div(out, rho_filtered)