import imageio
import vispy.plot as vp

fig = vp.Fig(size=(600, 500), show=False)
plotwidget = fig[0, 0]

fig.title = "bollu"
plotwidget.plot([(x, x**2) for x in range(0, 100000)], title="y = x^2")
plotwidget.colorbar(position="top", cmap="autumn")

if __name__ == '__main__':
    writer = imageio.get_writer('animation.gif')
    fig.show(run=True)
    im = fig.render(alpha=True)
    writer.append_data(im)
    writer.close()