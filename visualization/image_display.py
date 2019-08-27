import matplotlib.pyplot as plt

def _show_image(ax, image):
    ax.imshow(image, interpolation='none', extent=(0, image.shape[1], 0, image.shape[0]))

def _add_image_axes(fig, pos, title, font_size, image=None):
    # pos is a tuple of (left, bottom, width, height) in term of actual size (not #pixels)
    ax = fig.add_axes(pos)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if title is not None:
        ax.set_title(title, fontsize=font_size)
    if image is not None:
        _show_image(ax, image)
    return ax

class ImageDisplay(object):
    """
    Parameters
    ----------
    fit_mode: str or int
        "equal" for the case where all images have the same size.
        "max" for the case where the max height and width are fitted.
        int for the case where all images are resized to the given size
    """
    def __init__(self, dpi=96., margin=4, font_size=12, n_cols=None, fit_mode="equal"):
        try:
            import matplotlib
            self.dpi = matplotlib.rcParams["figure.dpi"]
        except ImportError:
            self.dpi = dpi
        self.px_margin = margin
        self.font_size = font_size
        self.n_cols = n_cols
        self.fit_mode = fit_mode
        self.px_h_img = 0
        self.px_w_img = 0
        self.images = list()

    def add_image(self, image, title=None):
        self.images.append((image, title))
        if self.fit_mode == "equal":
            if self.px_h_img == 0:
                self.px_h_img = image.shape[0]
                self.px_w_img = image.shape[1]
            else:
                assert self.px_h_img == image.shape[0] \
                    and self.px_w_img == image.shape[1], \
                        "(%d, %d) vs (%d, %d)" % (self.px_h_img, self.px_w_img, image.shape[0], image.shape[1])
        elif self.fit_mode == "max":
            self.px_h_img = max(image.shape[0], self.px_h_img)
            self.px_w_img = max(image.shape[1], self.px_w_img)
        else:
            raise NotImplementedError

    def show(self):
        if self.n_cols is None:
            n_cols = len(self.images)
            n_rows = 1
        else:
            n_cols = self.n_cols
            n_rows = (len(self.images)-1) // self.n_cols + 1
        px_h_title = self.font_size * 3 / 2 # emperically convert font size to px
        px_h_fig = (px_h_title + self.px_margin + self.px_h_img) * n_rows + self.px_margin
        px_w_fig = (self.px_margin + self.px_w_img) * n_cols + self.px_margin
        px2sz = 1./self.dpi
        fig = plt.figure(figsize=(px_w_fig*px2sz, px_h_fig*px2sz), dpi=self.dpi)
        # add image axes
        axes = list()
        bottom = 1.
        for r in range(n_rows):
            bottom -= (px_h_title + self.px_margin + self.px_h_img) * 1./px_h_fig
            left = self.px_margin * 1./px_w_fig
            for c in range(n_cols):
                i_img = r*n_cols + c
                if i_img >= len(self.images):
                    break
                img, title = self.images[i_img]
                axes.append(_add_image_axes(fig,
                        (left, bottom, img.shape[1]*1./px_w_fig, img.shape[0]*1./px_h_fig),
                        title, self.font_size, img))
                left += (self.px_margin + self.px_w_img) * 1./px_w_fig
        fig.show()
        return fig, axes

    def show_images(self, images, wait=True):
        for image, title in images:
            self.add_image(image, title)
        fig, axes = self.show()
        if wait:
            plt.waitforbuttonpress()
        return fig, axes


            